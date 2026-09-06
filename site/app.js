/* Minutes — the parent's screen, as a hash-routed single page.

   No build step, no framework. Every data call is one POST to the API base
   (default /api, the proxy on this origin) with a JSON body that is the
   runtime's own payload, and the key from the URL fragment (#key=...) sent as
   x-minutes-key. Responses are the runtime's JSON:
     {"status": "done" | "error" | "awaiting_approval" | "accepted", ...}

   Runtime actions used here, and nothing else: ingest_iep, wake, answer,
   list_evidence, add_note, add_correspondence, statement, outbox, declined,
   mark_received, requests, deadlines, audit. Where the runtime's reply lacks a
   field, the view degrades and a comment beside it says what was missing. */

(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // Small helpers
  // -------------------------------------------------------------------------

  var app = document.getElementById('app');
  var SAMPLE_CASE = 'maya-demo';
  var SAMPLE_REPLAY = '2026-12-01';
  var DISCLAIMER =
    'Minutes compiles documentation from the IEP and the family’s own records. It is not a law firm, ' +
    'does not give legal advice, and never posts anything: an approved letter waits in your outbox for you ' +
    'to send by a method that proves delivery.';

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function attr(s) { return esc(s); }

  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  function splitISO(iso) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    return m ? { y: +m[1], m: +m[2], d: +m[3] } : null;
  }
  function fmtDate(iso) {
    var p = splitISO(iso);
    return p ? p.d + ' ' + MONTHS[p.m - 1] + ' ' + p.y : (iso || '');
  }
  function fmtLongDate(iso) {
    var p = splitISO(iso);
    return p ? p.d + ' ' + MONTHS_LONG[p.m - 1] + ' ' + p.y : (iso || '');
  }
  function todayISO() {
    var d = new Date();
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  }
  function schoolYearStart() {
    // Absent a ledger, the first of September of the current school year.
    var d = new Date();
    var y = d.getMonth() >= 7 ? d.getFullYear() : d.getFullYear() - 1;
    return y + '-09-01';
  }
  function fmtMin(n) {
    n = Number(n) || 0;
    return n.toLocaleString('en-US');
  }
  function roundHalfEven(x) {
    var f = Math.floor(x), diff = x - f;
    if (diff > 0.5) return f + 1;
    if (diff < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }
  function hrs(n) {
    var h = roundHalfEven((Number(n) || 0) / 6) / 10;
    return (h % 1 === 0 ? String(h) : h.toFixed(1)) + 'h';
  }
  function minCell(n, opts) {
    opts = opts || {};
    n = Number(n) || 0;
    var cls = opts.undoc && n > 0 ? ' class="undoc"' : '';
    var small = !opts.undoc && n >= 180 ? '<span class="hrs">' + hrs(n) + '</span>' : '';
    return '<td' + cls + '>' + fmtMin(n) + small + '</td>';
  }
  function plural(n, one, many) { return n === 1 ? one : (many || one + 's'); }
  function newCaseId() {
    var bytes = new Uint8Array(6);
    (window.crypto || window.msCrypto).getRandomValues(bytes);
    var hex = '';
    for (var i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, '0');
    return 'case-' + hex;
  }
  function toast(text) {
    var el = document.createElement('div');
    el.className = 'toast'; el.setAttribute('role', 'status'); el.textContent = text;
    document.body.appendChild(el);
    setTimeout(function () { el.remove(); }, 2400);
  }

  // -------------------------------------------------------------------------
  // Storage: the key and API base live in this tab's session; per-case context
  // (alias, ledger start, services) is cached so a header can paint at once.
  // -------------------------------------------------------------------------

  var store = {
    get: function (k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { sessionStorage.setItem(k, v); } catch (e) {} },
    del: function (k) { try { sessionStorage.removeItem(k); } catch (e) {} }
  };
  function apiBase() { return store.get('minutes.apiBase') || '/api'; }
  function apiKey() { return store.get('minutes.key') || ''; }
  function caseCtx(id) {
    try { return JSON.parse(store.get('minutes.case.' + id) || '{}'); } catch (e) { return {}; }
  }
  function saveCtx(id, patch) {
    var ctx = caseCtx(id);
    Object.keys(patch).forEach(function (k) { if (patch[k] != null) ctx[k] = patch[k]; });
    store.set('minutes.case.' + id, JSON.stringify(ctx));
    return ctx;
  }
  function asOf(id) { return id === SAMPLE_CASE ? (store.get('minutes.asof.' + id) || '') : ''; }
  function setAsOf(id, v) { if (v) store.set('minutes.asof.' + id, v); else store.del('minutes.asof.' + id); }
  function theme() { try { return localStorage.getItem('minutes.theme') || 'system'; } catch (e) { return 'system'; } }
  function setTheme(v) {
    try { if (v === 'system') localStorage.removeItem('minutes.theme'); else localStorage.setItem('minutes.theme', v); } catch (e) {}
    if (v === 'light' || v === 'dark') document.documentElement.setAttribute('data-theme', v);
    else document.documentElement.removeAttribute('data-theme');
  }

  // The key arrives once, in the fragment, and is then kept out of the URL.
  // A replay date for the sample (#/case/maya-demo?asof=2026-12-01) is
  // absorbed the same way, so a judge can be handed a link to 1 December.
  function absorbKey() {
    var h = location.hash || '';
    var k = /[#?&]key=([^&]+)/.exec(h);
    var a = /[?&]asof=(\d{4}-\d{2}-\d{2})/.exec(h);
    if (!k && !a) return;
    if (k) store.set('minutes.key', decodeURIComponent(k[1]));
    if (a) { var c = /#\/case\/([^/?&]+)/.exec(h); if (c && decodeURIComponent(c[1]) === SAMPLE_CASE) setAsOf(SAMPLE_CASE, a[1]); }
    var rest = h.replace(/[?&](key|asof)=[^&]+/g, '').replace(/^#key=[^&]+&?/, '#').replace(/\?$/, '');
    if (rest === '#' || rest === '') rest = '#/';
    history.replaceState(null, '', location.pathname + location.search + rest);
  }

  // -------------------------------------------------------------------------
  // API
  // -------------------------------------------------------------------------

  function ApiError(message, what) { this.message = message; this.what = what || ''; }

  function call(payload, opts) {
    opts = opts || {};
    var headers = { 'Content-Type': 'application/json' };
    if (apiKey()) headers['x-minutes-key'] = apiKey();
    var base = apiBase();
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = null;
    if (controller && opts.timeoutMs) timer = setTimeout(function () { controller.abort(); }, opts.timeoutMs);
    return fetch(base, { method: 'POST', headers: headers, body: JSON.stringify(payload), signal: controller ? controller.signal : undefined })
      .catch(function (e) {
        if (e && e.name === 'AbortError') {
          var err = new ApiError('Minutes did not answer within ' + Math.round(opts.timeoutMs / 1000) + ' seconds.', 'The work may still have been recorded. What Minutes did shows the trail.');
          err.timedOut = true;
          throw err;
        }
        throw new ApiError('Could not reach Minutes at ' + base + '.',
          'If you are running it on this machine, start `python app.py` and `python scripts/dev_web.py`, then open the address dev_web prints. Otherwise check the API base under Settings.');
      })
      .finally(function () { if (timer) clearTimeout(timer); })
      .then(function (res) {
        return res.text().then(function (text) {
          var data = null;
          try { data = JSON.parse(text); } catch (e) {}
          if (res.status === 401 || res.status === 403) {
            throw new ApiError('The key was refused (HTTP ' + res.status + ').',
              'Open the link you were given again — it carries the key — or paste the key under Settings.');
          }
          if (data == null) {
            throw new ApiError('Minutes answered with something that is not JSON (HTTP ' + res.status + ').',
              'The API base may point at the wrong place. Check it under Settings.');
          }
          if (!res.ok && !data.status) {
            throw new ApiError('Minutes answered HTTP ' + res.status + '.', data.error || data.message || '');
          }
          return data;
        });
      });
  }
  function isUnknownAction(res) {
    return res && res.status === 'error' && /unknown action/i.test(String(res.error || ''));
  }
  function errorFrom(res, action) {
    if (res instanceof ApiError) return res;
    if (isUnknownAction(res)) {
      return new ApiError('This runtime does not answer `' + action + '` yet.',
        'The deployed runtime lists these actions: ' + String(res.error).replace(/^.*expected one of /, '') + '. This screen needs `' + action + '`.');
    }
    var text = String((res && res.error) || 'Minutes reported an error and did not say what.');
    var what = '';
    if (/read-only/i.test(text)) what = 'The sample case cannot be written to. Start your own case from the front page to add to a file.';
    else if (/unknown case/i.test(text)) what = 'Paste the IEP first; that is what creates a case.';
    return new ApiError(text, what);
  }

  // The case itself: alias, ledger start and service names, read once per
  // tab from the runtime's `case` action and kept so every header can paint.
  function ensureCase(id) {
    var ctx = caseCtx(id);
    if (ctx.loaded) return Promise.resolve(ctx);
    return call({ action: 'case', case_id: id }).then(function (res) {
      if (res.status === 'error') throw errorFrom(res, 'case');
      var starts = (res.obligations || []).map(function (o) { return o.start_date; }).filter(Boolean).sort();
      return saveCtx(id, {
        loaded: true, exists: res.exists !== false, sample: !!res.sample,
        alias: res.student || null, ledgerStart: starts[0] || null,
        services: (res.obligations || []).map(function (o) { return o.service; }),
        counts: res.counts || null
      });
    });
  }
  function refreshMasthead(id) {
    var old = app.querySelector('.masthead');
    if (!old) return;
    var tmp = document.createElement('div'); tmp.innerHTML = masthead(id);
    old.replaceWith(tmp.firstElementChild);
  }
  function noCaseHtml(id) {
    return '<section class="tight"><div class="eyebrow">Case ' + esc(id) + '</div><h1>There is no case here yet.</h1>' +
      '<p class="lede">A case begins when its IEP is read into a ledger. Nothing has been pasted for this id.</p>' +
      '<div class="cta"><a href="#/case/new/' + attr(encodeURIComponent(id)) + '"><button type="button" class="primary">Paste the IEP for this case</button></a>' +
      '<a href="#/case/' + SAMPLE_CASE + '"><button type="button">Open the sample instead</button></a></div></section>';
  }
  // Every case route starts here: paint the shell, look the case up, then run the view.
  function withCase(id, sub, title, loadingText, guard, view) {
    shell({ caseId: id, sub: sub, title: title, body: loadingText });
    ensureCase(id).then(function (ctx) {
      if (!guard()) return;
      refreshMasthead(id);
      if (!ctx.exists) { setMain(noCaseHtml(id)); return; }
      view(ctx);
    }).catch(function (err) { if (guard()) setMain('<section class="tight"><div class="eyebrow">Case ' + esc(id) + '</div>' + errorHtml(err) + '</section>'); });
  }

  // -------------------------------------------------------------------------
  // Shell: masthead, case nav, footer
  // -------------------------------------------------------------------------

  var NAV = [
    ['', 'This week'], ['evidence', 'Evidence'], ['statement', 'Statement'],
    ['letters', 'Letters'], ['dates', 'Dates'], ['audit', 'What Minutes did']
  ];

  function masthead(caseId) {
    var right = '';
    if (caseId) {
      var ctx = caseCtx(caseId);
      var on = asOf(caseId) || todayISO();
      right = '<div class="asof">' +
        '<span>Case ' + (ctx.alias ? '<b>' + esc(ctx.alias) + '</b> · ' : '') + '<span class="mono">' + esc(caseId) + '</span></span>' +
        '<span>as of <b>' + esc(fmtLongDate(on)) + '</b></span>' +
        '<button type="button" class="iconbtn" data-open-settings>Settings</button></div>';
    } else {
      right = '<div class="asof"><button type="button" class="iconbtn" data-open-settings>Settings</button></div>';
    }
    return '<header class="masthead"><a class="wordmark" href="#/">Minutes <small>the statement the school never sends</small></a>' + right + '</header>';
  }
  function caseNav(caseId, sub) {
    return '<nav class="casenav" aria-label="This case">' + NAV.map(function (n) {
      var href = '#/case/' + encodeURIComponent(caseId) + (n[0] ? '/' + n[0] : '');
      var cur = (sub || '') === n[0] ? ' aria-current="page"' : '';
      return '<a href="' + attr(href) + '"' + cur + '>' + n[1] + '</a>';
    }).join('') + '</nav>';
  }
  function caseFooter(caseId) {
    var first = caseId === SAMPLE_CASE
      ? 'Sample case. Every person, school and document here is fictional.'
      : 'Case ' + esc(caseId) + '. Minutes keeps this case in its own session; nothing here is shared with anyone.';
    return '<footer><div>' + first + '</div><div>' + esc(DISCLAIMER) + '</div></footer>';
  }
  function shell(opts) {
    var html = masthead(opts.caseId);
    if (opts.caseId) html += caseNav(opts.caseId, opts.sub);
    html += '<main id="main">' + opts.body + '</main>';
    html += opts.caseId ? caseFooter(opts.caseId) : '<footer><div>' + esc(DISCLAIMER) + '</div></footer>';
    app.innerHTML = html;
    document.title = opts.title ? opts.title + ' · Minutes' : 'Minutes';
    var h = app.querySelector('main h1');
    if (h) { h.setAttribute('tabindex', '-1'); h.style.outline = 'none'; h.focus({ preventScroll: true }); }
    window.scrollTo(0, 0);
  }
  function setMain(html) { document.getElementById('main').innerHTML = html; }

  function loadingHtml(text, eyebrow) {
    return '<div class="week"><div class="eyebrow">' + esc(eyebrow || 'Reading') + '</div>' +
      '<p class="loading" role="status">' + esc(text) + '<span class="bar"></span></p></div>';
  }
  function errorHtml(err) {
    return '<div class="state-box error" role="alert"><b>' + esc(err.message) + '</b>' +
      (err.what ? '<div class="what">' + esc(err.what) + '</div>' : '') + '</div>';
  }
  function emptyHtml(text, what) {
    return '<div class="state-box"><div>' + esc(text) + '</div>' + (what ? '<div class="what">' + esc(what) + '</div>' : '') + '</div>';
  }

  // -------------------------------------------------------------------------
  // Letters: body with footnote markers and parent blanks, footnotes, basis
  // -------------------------------------------------------------------------

  function letterBodyHtml(body) {
    return esc(body)
      .replace(/\[\[PARENT: ([^\]]*)\]\]/g, '<span class="fill">[PARENT: $1]</span>')
      .replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
  }
  function provChip(provenance) {
    var p = String(provenance || '');
    if (p === 'school_confirmed') return '<span class="prov">School record</span>';
    if (p === 'parent_observed') return '<span class="prov parent">Your note</span>';
    if (p === 'documented_silence') return '<span class="prov silence">Documented silence</span>';
    return '<span class="prov neutral">' + esc(p || 'record') + '</span>';
  }
  function footnotesHtml(citations, count) {
    if (Array.isArray(citations) && citations.length) {
      return '<ol class="footnotes">' + citations.map(function (c) {
        var ev = c.evidence || {};
        return '<li><span class="m">' + esc(c.marker || '') + '</span>' + provChip(ev.provenance) +
          '<span>' + esc(ev.source ? sourceLabel(ev.source) + ': ' : '') + esc(ev.detail || c.claim || '') + '</span></li>';
      }).join('') + '</ol>';
    }
    // The interrupt reason carries only a count of citations; the full list
    // lives on the wake's new_cards[].draft, which we join by subject when it
    // is there. When it is not, say how many facts are cited and stop.
    if (count) return '<p class="basis">' + count + ' cited ' + plural(count, 'fact') + ' stand behind this letter; the runtime sent the count, not the list.</p>';
    return '';
  }
  function sourceLabel(source) {
    var m = /^iep-(\d{4}-\d{2}-\d{2})$/.exec(source);
    return m ? 'IEP of ' + m[1] : source;
  }
  function basisHtml(list) {
    if (!Array.isArray(list) || !list.length) return '';
    return '<p class="basis"><b>Authorities cited:</b> ' + list.map(esc).join(' · ') + '</p>';
  }
  function letterDetails(opts) {
    // opts: subject, body, citations, citationCount, basis, open
    return '<details class="letter"' + (opts.open ? ' open' : '') + '>' +
      '<summary><span>The letter · ' + esc(opts.subject || '') + '</span><span class="toggle"></span></summary>' +
      '<div class="paper">' + letterBodyHtml(opts.body || '') + '</div>' +
      footnotesHtml(opts.citations, opts.citationCount) + basisHtml(opts.basis) + '</details>';
  }
  function kindLabel(kind) {
    return { records_request: 'Records request', shortfall_notice: 'Shortfall notice',
      compensatory_request: 'Compensatory request', deadline_reminder: 'Deadline reminder' }[kind] || (kind || 'Letter');
  }
  function printLetter(subject, body) {
    var sheet = document.getElementById('print-sheet');
    sheet.innerHTML = '<div class="paper">' + letterBodyHtml(body) + '</div>';
    document.body.classList.add('print-letter');
    var done = function () { document.body.classList.remove('print-letter'); sheet.innerHTML = ''; window.removeEventListener('afterprint', done); };
    window.addEventListener('afterprint', done);
    window.print();
    setTimeout(done, 2000);
  }
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(function () { toast('Copied.'); }, function () { toast('Could not copy; select the letter and copy it by hand.'); });
    }
    toast('Copying is not available here; select the letter and copy it by hand.');
    return Promise.resolve();
  }

  // -------------------------------------------------------------------------
  // Markdown: the statement's rendering, and the tables inside it
  // -------------------------------------------------------------------------

  function inlineMd(s) {
    return esc(s).replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|\s)--(\s|$)/g, '$1—$2');
  }
  function mdToHtml(md) {
    var lines = String(md || '').replace(/\r/g, '').split('\n');
    var out = [], i = 0;
    while (i < lines.length) {
      var line = lines[i];
      if (!line.trim()) { i++; continue; }
      var h = /^(#{1,3})\s+(.*)$/.exec(line);
      if (h) { var lv = h[1].length; out.push('<h' + lv + '>' + inlineMd(h[2]) + '</h' + lv + '>'); i++; continue; }
      if (/^\|/.test(line)) {
        var rows = [];
        while (i < lines.length && /^\|/.test(lines[i])) { rows.push(lines[i]); i++; }
        out.push(mdTableHtml(rows));
        continue;
      }
      if (/^[-*]\s+/.test(line)) {
        var items = [];
        while (i < lines.length && /^[-*]\s+/.test(lines[i])) { items.push('<li>' + inlineMd(lines[i].replace(/^[-*]\s+/, '')) + '</li>'); i++; }
        out.push('<ul>' + items.join('') + '</ul>');
        continue;
      }
      var para = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,3}\s|\||[-*]\s)/.test(lines[i])) { para.push(lines[i]); i++; }
      out.push('<p>' + inlineMd(para.join(' ')) + '</p>');
    }
    return out.join('\n');
  }
  function splitRow(row) {
    return row.replace(/^\|/, '').replace(/\|\s*$/, '').split('|').map(function (c) { return c.trim(); });
  }
  function mdTableHtml(rows) {
    var head = splitRow(rows[0]);
    var align = rows[1] ? splitRow(rows[1]).map(function (c) { return /:\s*$/.test(c) ? 'num' : ''; }) : [];
    var body = rows.slice(2).map(function (r) {
      var cells = splitRow(r);
      var total = /^total$/i.test(cells[0]);
      return '<tr' + (total ? ' class="total"' : '') + '>' + cells.map(function (c, j) {
        return '<td class="' + (align[j] || '') + '">' + inlineMd(c) + '</td>';
      }).join('') + '</tr>';
    });
    return '<div class="tablewrap"><table class="text"><thead><tr>' + head.map(function (c, j) {
      return '<th class="' + (align[j] || '') + '">' + inlineMd(c) + '</th>';
    }).join('') + '</tr></thead><tbody>' + body.join('') + '</tbody></table></div>';
  }
  function mdTables(md) {
    // Every table in the markdown, keyed by the nearest heading above it.
    var lines = String(md || '').replace(/\r/g, '').split('\n');
    var tables = {}, heading = '', i = 0;
    while (i < lines.length) {
      var h = /^#{1,3}\s+(.*)$/.exec(lines[i]);
      if (h) { heading = h[1].trim(); i++; continue; }
      if (/^\|/.test(lines[i])) {
        var rows = [];
        while (i < lines.length && /^\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
        tables[heading] = rows.filter(function (r, k) { return k !== 1; }); // drop the alignment row
        continue;
      }
      i++;
    }
    return tables;
  }
  function mdSectionText(md, heading) {
    // The paragraphs directly under a heading, up to the next heading.
    var lines = String(md || '').replace(/\r/g, '').split('\n');
    var out = [], on = false;
    for (var i = 0; i < lines.length; i++) {
      var h = /^#{1,3}\s+(.*)$/.exec(lines[i]);
      if (h) { if (on) break; on = h[1].trim() === heading; continue; }
      if (on && lines[i].trim()) out.push(lines[i].trim());
    }
    return out;
  }
  function cellInt(c) { var m = /^-?[\d,]+/.exec(String(c || '').trim()); return m ? parseInt(m[0].replace(/,/g, ''), 10) : 0; }

  // -------------------------------------------------------------------------
  // Router
  // -------------------------------------------------------------------------

  function parseHash() {
    var h = (location.hash || '#/').replace(/^#/, '');
    h = h.split('?')[0];
    var parts = h.split('/').filter(Boolean);
    if (!parts.length) return { name: 'landing' };
    if (parts[0] === 'case' && parts[1] === 'new') return { name: 'new', id: parts[2] ? decodeURIComponent(parts[2]) : '' };
    if (parts[0] === 'case' && parts[1]) {
      var id = decodeURIComponent(parts[1]);
      var sub = parts[2] || '';
      if (['', 'evidence', 'statement', 'letters', 'dates', 'audit'].indexOf(sub) === -1) return { name: 'missing' };
      return { name: 'case', id: id, sub: sub };
    }
    return { name: 'missing' };
  }
  var renderSeq = 0;
  function render() {
    absorbKey();
    var r = parseHash();
    var seq = ++renderSeq;
    var guard = function () { return seq === renderSeq; }; // a stale fetch must not paint over a newer route
    if (r.name === 'landing') return viewLanding();
    if (r.name === 'new') return viewCaseNew(r.id);
    if (r.name === 'case') {
      if (r.sub === '') return viewWeek(r.id, guard);
      if (r.sub === 'evidence') return viewEvidence(r.id, guard);
      if (r.sub === 'statement') return viewStatement(r.id, guard);
      if (r.sub === 'letters') return viewLetters(r.id, guard);
      if (r.sub === 'dates') return viewDates(r.id, guard);
      if (r.sub === 'audit') return viewAudit(r.id, guard);
    }
    shell({ title: 'Not found', body: '<section class="tight"><h1>There is no page here.</h1><p class="lede">Go back to <a href="#/">the start</a>, or open a case by its id: <code>#/case/&lt;id&gt;</code>.</p></section>' });
  }

  // -------------------------------------------------------------------------
  // Landing
  // -------------------------------------------------------------------------

  function viewLanding() {
    var body =
      '<div class="pitch">' +
      '<div class="eyebrow">A background agent for parents, built with Strands Agents on Amazon Bedrock AgentCore</div>' +
      '<h1>Schools send report cards about your child. Nobody sends a statement about the school.</h1>' +
      '<p class="sub">A child’s IEP is a legal promise written in numbers — <b>300 minutes of speech therapy a month, occupational therapy twice a week, an annual review by March 12.</b> Whether those minutes are delivered is a question almost nobody can answer, because answering it means reconciling a year of scattered emails and half-remembered Tuesdays against a document in a drawer. Minutes keeps that ledger, stays quiet, and interrupts only when a decision is yours to make.</p>' +
      '<div class="cta"><a href="#/case/' + SAMPLE_CASE + '"><button type="button" class="primary">Try the sample case</button></a>' +
      '<a href="#/case/new"><button type="button">Start your case</button></a>' +
      '<span class="note">The sample is a fictional third-grader’s Fall 2026 semester.</span></div>' +
      '</div>' +

      '<section><h2>How it works</h2><ol class="how">' +
      '<li><span><b>The promise becomes a ledger.</b> The IEP is read once into typed obligations — minutes per session, sessions per period, provider, setting, dates — and every deadline it names. Each fact keeps the sentence it came from.</span></li>' +
      '<li><span><b>Evidence is graded, never assumed.</b> A school email, a service log, a progress report, your own note: each becomes a dated fact carrying its provenance.</span></li>' +
      '<li><span><b>Active discovery.</b> On a cadence, Minutes compiles a request for the district’s own service records. You release it and post it yourself; the response clock runs from the date you say it was received. Silence past that date becomes evidence.</span></li>' +
      '<li><span><b>Four buckets, kept apart.</b> owed = delivered + excused + documented misses + undocumented. Minutes with no record either way are a gap in the evidence, not an accusation.</span></li>' +
      '<li><span><b>Letters are compiled, not written.</b> Every factual sentence carries a footnote bound to a piece of evidence. A claim without evidence is omitted, and nothing leaves without you: the framework, not the prompt, enforces that.</span></li>' +
      '<li><span><b>The Statement.</b> Once a month, one artifact: owed, delivered, excused, short, and where every figure came from. The rest of the month the agent is quiet. That silence is the feature.</span></li>' +
      '</ol></section>' +

      '<section><h2>Evidence grades</h2><dl class="grade">' +
      '<dt>school_confirmed</dt><dd>The district’s own record or written statement</dd>' +
      '<dt>parent_observed</dt><dd>The family’s log — dated, but not the school’s record</dd>' +
      '<dt>documented_silence</dt><dd>Records were properly requested and not produced</dd>' +
      '</dl></section>' +

      '<section><h2>From a Statement</h2><div class="excerpt">' +
      '<p class="headline-figure">7,050 minutes (117.5 hours) short this period, <span class="undoc-n">6,825 minutes of it with no record either way.</span></p>' +
      '<p>8,520 minutes owed. 1,365 minutes documented as delivered. 105 minutes excluded as falling on dates a record notes your child was absent. Nobody has recorded 6,825 minutes of that shortfall either way, which is a gap in the evidence rather than a record of non-delivery.</p>' +
      '</div><p class="lede" style="margin-top:14px">The sample semester, 8 September 2026 to 29 January 2027. Every figure is arithmetic on the IEP’s own numbers against the records on file; the model reads text and writes connective prose, and never decides a number.</p></section>' +

      '<section><h2>What Minutes is not</h2><ul class="facts">' +
      '<li>Not a chatbot. There is nothing to open and nothing to converse with.</li>' +
      '<li>Not legal advice. It compiles documentation; what to do with it is yours to decide, with your advocate or attorney if you have one.</li>' +
      '<li>Not a mail client. It cannot post anything. You send the letter by a method that proves delivery and tell Minutes the date it arrived.</li>' +
      '</ul></section>';
    shell({ title: 'Minutes', body: body });
  }

  // -------------------------------------------------------------------------
  // New case: paste the IEP, confirm the ledger
  // -------------------------------------------------------------------------

  function viewCaseNew(presetId) {
    var body =
      '<section class="tight"><div class="eyebrow">A new case' + (presetId ? ' · ' + esc(presetId) : '') + '</div>' +
      '<h1>Paste the IEP.</h1>' +
      '<p class="lede">Minutes reads it once into a ledger: each service as minutes, sessions and dates, and each deadline the document sets. You will see the ledger and confirm it before anything else happens. Use a pseudonym for your child if the document names them; the ledger keeps an alias, never the full name.</p>' +
      '<form class="sheet" id="iep-form"><label class="field"><span>The IEP, as text</span>' +
      '<textarea class="iep" name="iep" id="iep-text" required minlength="200" placeholder="Paste the services and the dates the IEP sets. The whole document is fine." spellcheck="false"></textarea>' +
      '<span class="counter" id="iep-count">0 characters · at least 200 needed</span></label>' +
      '<div class="formfoot"><button type="submit" class="primary" id="iep-submit">Read the IEP</button>' +
      '<span class="note">This is the one step that reads the document with a model. It can take a minute.</span></div></form>' +
      '<div id="iep-result"></div></section>';
    shell({ title: 'A new case', body: body });

    var ta = document.getElementById('iep-text'), count = document.getElementById('iep-count');
    ta.addEventListener('input', function () {
      var n = ta.value.length;
      count.textContent = fmtMin(n) + ' characters' + (n < 200 ? ' · at least 200 needed' : '');
      count.className = 'counter' + (n < 200 ? ' short' : '');
    });
    document.getElementById('iep-form').addEventListener('submit', function (e) {
      e.preventDefault();
      var text = ta.value.trim();
      if (text.length < 200) { count.className = 'counter short'; ta.focus(); return; }
      var id = presetId || newCaseId();
      var result = document.getElementById('iep-result');
      var btn = document.getElementById('iep-submit');
      btn.disabled = true;
      result.innerHTML = loadingHtml('Reading the IEP into a ledger. Case ' + id + '.', 'Extracting');
      call({ action: 'ingest_iep', case_id: id, text: text }).then(function (res) {
        btn.disabled = false;
        if (res.status === 'error') { result.innerHTML = errorHtml(errorFrom(res, 'ingest_iep')); return; }
        // The reply is the runtime's `case` shape: student, school_year,
        // iep_date, obligations (with source quotes), deadlines (kind, due,
        // description — no quote), accommodations (strings).
        if (!Array.isArray(res.obligations)) {
          result.innerHTML = errorHtml(new ApiError('The IEP was read, but the reply carried no ledger to confirm.',
            'The response had these fields: ' + Object.keys(res).join(', ') + '. Expected `obligations` and `deadlines`.'));
          return;
        }
        result.innerHTML = ledgerHtml(id, res);
        document.getElementById('confirm-ledger').addEventListener('click', function () {
          var starts = res.obligations.map(function (o) { return o.start_date; }).filter(Boolean).sort();
          saveCtx(id, {
            loaded: true, exists: true, sample: false,
            alias: res.student || null, ledgerStart: starts[0] || null,
            services: res.obligations.map(function (o) { return o.service; })
          });
          location.hash = '#/case/' + encodeURIComponent(id);
        });
        document.getElementById('confirm-ledger').focus();
      }).catch(function (err) { btn.disabled = false; result.innerHTML = errorHtml(err); });
    });
  }

  function ledgerHtml(id, ledger) {
    var obligations = ledger.obligations || [], deadlines = ledger.deadlines || [], acc = ledger.accommodations || [];
    var alias = ledger.student || ledger.student_alias;
    var head = '<div class="ledger-head">' +
      (alias ? '<span>Student <b>' + esc(alias) + '</b></span>' : '') +
      (ledger.school_year ? '<span>School year <b>' + esc(ledger.school_year) + '</b></span>' : '') +
      (ledger.iep_date ? '<span>IEP dated <b>' + esc(fmtDate(ledger.iep_date)) + '</b></span>' : '') +
      '<span>Case <b class="mono">' + esc(id) + '</b></span></div>';
    var obl = obligations.length
      ? '<div class="tablewrap"><table class="text"><thead><tr><th>Service</th><th class="num">Minutes</th><th class="num">Sessions</th><th>Provider · setting</th><th>From – to</th></tr></thead><tbody>' +
        obligations.map(function (o) {
          return '<tr><td>' + esc(o.service) + '<span class="sub quote">“' + esc(o.source_quote || '') + '”</span></td>' +
            '<td class="num">' + fmtMin(o.minutes_per_session) + '</td>' +
            '<td class="num">' + esc(o.sessions_per_period) + ' / ' + esc(o.period) + '</td>' +
            '<td>' + esc(o.provider_role || '') + (o.setting ? ' · ' + esc(o.setting) : '') + '</td>' +
            '<td>' + esc(fmtDate(o.start_date)) + ' – ' + esc(fmtDate(o.end_date)) + '</td></tr>';
        }).join('') + '</tbody></table></div>'
      : emptyHtml('No service obligations were found in this text.', 'Check that the services section, with minutes and frequency, is in what you pasted.');
    var dl = deadlines.length
      ? '<ul class="rows">' + deadlines.slice().sort(function (a, b) { return String(a.due).localeCompare(String(b.due)); }).map(function (d) {
          return '<li><span class="d">' + esc(fmtDate(d.due)) + '</span><span>' + esc(d.description) +
            (d.source_quote ? '<span class="sub quote">“' + esc(d.source_quote) + '”</span>' : '') + '</span><span class="state">' + esc(String(d.kind || '').replace(/_/g, ' ')) + '</span></li>';
        }).join('') + '</ul>'
      : emptyHtml('No dates were found in this text.');
    var ac = acc.length ? '<ul class="facts">' + acc.map(function (a) { return '<li>' + esc(typeof a === 'string' ? a : a.description) + '</li>'; }).join('') + '</ul>' : '';
    return '<section><h2>The ledger Minutes read</h2><p class="lede">Each line keeps the sentence it came from. If something is wrong here, it will be wrong in every statement, so read it once now.</p>' + head + '</section>' +
      '<section class="tight"><h2>Services promised</h2>' + obl + '</section>' +
      '<section class="tight"><h2>Dates the IEP sets</h2>' + dl + '</section>' +
      (ac ? '<section class="tight"><h2>Accommodations</h2>' + ac + '</section>' : '') +
      '<section class="tight"><div class="formfoot"><button type="button" class="primary" id="confirm-ledger">This is right — open the case</button>' +
      '<span class="note">The case is ' + esc(id) + '. Keep the id; it is how you come back.</span></div></section>';
  }

  // -------------------------------------------------------------------------
  // This week: the wake, the decision cards, release or decline
  // -------------------------------------------------------------------------

  function asOfControls(id) {
    if (id !== SAMPLE_CASE) return '';
    var v = asOf(id);
    return '<div class="controls"><label class="field"><span>As of</span><input type="date" id="asof" value="' + attr(v) + '"><span class="hint">Leave empty for today.</span></label>' +
      '<button type="button" class="small" id="asof-apply">Check as of this date</button>' +
      (v !== SAMPLE_REPLAY ? '<button type="button" class="small link" id="asof-replay">Replay 1 December 2026</button>' : '') +
      (v ? '<button type="button" class="small link" id="asof-clear">Use today</button>' : '') + '</div>';
  }
  function bindAsOf(id) {
    var input = document.getElementById('asof');
    if (!input) return;
    document.getElementById('asof-apply').addEventListener('click', function () { setAsOf(id, input.value); render(); });
    var r = document.getElementById('asof-replay'); if (r) r.addEventListener('click', function () { setAsOf(id, SAMPLE_REPLAY); render(); });
    var c = document.getElementById('asof-clear'); if (c) c.addEventListener('click', function () { setAsOf(id, ''); render(); });
  }

  function headlineHtml(headline, quiet) {
    if (quiet) return '<h1 class="quiet">Nothing needs you. Minutes checked.</h1>';
    var m = /^(\d+) (decisions? needs?) you\.$/.exec(String(headline || ''));
    if (m) return '<h1><span class="needs">' + m[1] + ' ' + m[2].split(' ')[0] + '</span> ' + m[2].split(' ')[1] + ' you.</h1>';
    return '<h1>' + esc(headline || 'Minutes checked.') + '</h1>';
  }

  function viewWeek(id, guard, force) {
    var on = asOf(id);
    var waiting = asOfControls(id) + loadingHtml('Checking the case' + (on ? ' as of ' + fmtLongDate(on) : '') + '. If this week found a letter, Minutes compiles it now; that step calls a model and can take a minute.', 'This week’s check');
    withCase(id, '', 'This week', waiting, guard, function () {
      // While a letter this tab was shown is still unanswered, the check it
      // came from is shown again rather than woken again: a new wake would
      // not re-put the letter (the runtime suppresses cards it raised), and
      // the parent's decision is the thing waiting, not a fresh check.
      var cached = readPending(id);
      if (!force && cached && cached.wake && (cached.interrupts || []).some(function (i) { return !(cached.settled || {})[i.id]; })) {
        renderWeek(id, cached.wake, cached);
        return;
      }
      setMain(waiting);
      bindAsOf(id);
      // First: is a letter from an earlier check still waiting? The runtime keeps
      // that on the case (action `pending`), so a fresh tab or another device
      // sees it without waking, and without the runtime suppressing the card
      // as one it already raised.
      call({ action: 'pending', case_id: id }).catch(function () { return null; }).then(function (pend) {
        if (!guard()) return;
        var open = pend && pend.status === 'awaiting_approval' ? (pend.interrupts || []) : [];
        if (open.length && !force) {
          renderWeek(id, {
            today: on || null,
            headline: open.length + (open.length === 1 ? ' decision needs you.' : ' decisions need you.'),
            checked: ['A letter Minutes compiled at an earlier check is still waiting for your answer.'],
            new_cards: [],
            approval: { status: 'awaiting_approval', interrupts: open }
          });
          return;
        }
        var payload = { action: 'wake', case_id: id };
        if (on) payload.today = on; // omitted otherwise, so the runtime uses the real date
        return call(payload).then(function (res) {
        if (!guard()) return;
        if (res.status === 'error') { setMain(asOfControls(id) + '<div class="week"><div class="eyebrow">This week’s check</div>' + errorHtml(errorFrom(res, 'wake')) + '</div>'); bindAsOf(id); return; }
        if (res.status === 'accepted') {
          setMain(asOfControls(id) + '<div class="week"><div class="eyebrow">This week’s check</div><h1 class="quiet">Minutes is checking in the background.</h1><p class="reassure">Run ' + esc(res.run_id || '') + ' was accepted. Come back to this page in a minute and it will show what the check found.</p></div>');
          bindAsOf(id); return;
        }
        if (res.period && res.period[0]) saveCtx(id, { ledgerStart: res.period[0] });
        renderWeek(id, res);
        });
      }).catch(function (err) { if (guard()) { setMain(asOfControls(id) + '<div class="week"><div class="eyebrow">This week’s check</div>' + errorHtml(err) + '</div>'); bindAsOf(id); } });
    });
  }

  // The runtime lists a waiting approval (action `pending`), so a fresh tab can
  // see it. What this tab was shown is also kept locally until it is answered,
  // so a page reload mid-decision costs nothing.
  function pendingKey(id) { return 'minutes.pending.' + id; }
  function readPending(id) { try { return JSON.parse(store.get(pendingKey(id)) || 'null'); } catch (e) { return null; } }
  function writePending(state) {
    var open = state.interrupts.some(function (i) { return !state.settled[i.id]; });
    if (!open) { store.del(pendingKey(state.id)); return; }
    var wake = Object.assign({}, state.wake, { approval: null }); // the interrupts are kept beside it, once
    store.set(pendingKey(state.id), JSON.stringify({ today: state.wake.today, wake: wake, interrupts: state.interrupts, new_cards: state.wake.new_cards || [], settled: state.settled }));
  }

  function renderWeek(id, wake, cached) {
    var state = { id: id, wake: wake, interrupts: [], settled: {}, lastMessage: '' };
    var approval = wake.approval;
    var checked = (wake.checked || []).map(function (c) { return '<li>' + esc(c) + '</li>'; });
    var fromCache = false;
    if (approval && approval.status === 'awaiting_approval') {
      state.interrupts = approval.interrupts || [];
      writePending(state);
    } else if (cached && (cached.interrupts || []).some(function (i) { return !(cached.settled || {})[i.id]; })) {
      fromCache = true;
      state.interrupts = cached.interrupts; state.settled = cached.settled || {};
      var seen = (wake.new_cards || []).map(function (c) { return c.card_id; });
      state.wake = Object.assign({}, wake, { new_cards: (wake.new_cards || []).concat((cached.new_cards || []).filter(function (c) { return seen.indexOf(c.card_id) === -1; })) });
      var open = state.interrupts.filter(function (i) { return !state.settled[i.id]; }).length;
      checked.push('<li id="waiting-line">' + open + ' ' + plural(open, 'letter', 'letters') + ' put to you on ' + esc(fmtDate(cached.today)) + ' ' + (open === 1 ? 'is' : 'are') + ' still waiting for your answer.</li>');
      wake = state.wake;
    }
    if (approval && approval.status === 'done') state.lastMessage = approval.message || '';

    var anyOpen = state.interrupts.some(function (i) { return !state.settled[i.id]; });
    if (wake.suppressed && !state.interrupts.length) checked.push('<li>' + wake.suppressed + ' ' + plural(wake.suppressed, 'decision was', 'decisions were') + ' already put to you by an earlier check and ' + (wake.suppressed === 1 ? 'is' : 'are') + ' not repeated. Nothing about ' + (wake.suppressed === 1 ? 'it' : 'them') + ' has got worse.</li>');
    if (!checked.length) checked.push('<li>The runtime did not say what it checked.</li>'); // wake.checked missing

    var html = asOfControls(id) +
      '<div class="week"><div class="eyebrow">This week’s check' + (wake.today ? ' · ' + esc(fmtDate(wake.today)) : '') + (fromCache ? ' · <button type="button" class="link small" id="recheck">Check again</button>' : '') + '</div>' +
      '<div id="headline">' + headlineHtml(wake.headline, wake.quiet && !anyOpen) + '</div>' +
      (wake.quiet && !anyOpen ? '<p class="reassure">Every service was reconciled, every date was read, and nothing needs a decision from you. Minutes will check again next week.</p>' : '') +
      '<ul class="checked">' + checked.join('') + '</ul></div>' +
      '<div id="decisions"></div>';
    setMain(html);
    bindAsOf(id);
    var re = document.getElementById('recheck');
    if (re) re.addEventListener('click', function () { viewWeek(id, function () { return true; }, true); });
    renderDecisions(state);
    var h = document.querySelector('main h1'); if (h) { h.setAttribute('tabindex', '-1'); h.style.outline = 'none'; h.focus({ preventScroll: true }); }
  }

  function cardFor(state, interrupt) {
    var reason = (interrupt && interrupt.reason) || {};
    var cards = state.wake.new_cards || [];
    var card = cards.filter(function (c) { return c.draft && reason.subject && c.draft.subject === reason.subject; })[0];
    if (!card) {
      var sameKind = cards.filter(function (c) { return c.draft && c.draft.kind === reason.letter_kind; });
      if (sameKind.length === 1) card = sameKind[0];
    }
    return card || null;
  }
  function urgencyChip(urgency, fallback) {
    if (urgency === 'time_sensitive') return '<span class="chip needs">Time-sensitive</span>';
    if (urgency === 'deadline_imminent') return '<span class="chip needs">Deadline imminent</span>';
    if (urgency === 'routine') return '<span class="chip">Routine</span>';
    return '<span class="chip">' + esc(fallback || 'Decision') + '</span>';
  }

  function decisionCardHtml(state, interrupt, index) {
    var reason = interrupt.reason || {};
    var card = cardFor(state, interrupt);
    var draft = card && card.draft ? card.draft : {};
    var title = (card && card.title) || reason.subject || reason.question || interrupt.name || 'A decision';
    var why = (card && card.why_now) || (card ? '' : reason.question) || '';
    var facts = (card && card.facts) || [];
    var action = card && card.recommended_action;
    var blanks = Array.isArray(reason.blanks_only_you_can_fill) ? reason.blanks_only_you_can_fill : [];
    var blocking = Array.isArray(reason.blocking_before_send) ? reason.blocking_before_send : [];
    var settled = state.settled[interrupt.id];
    var key = attr(interrupt.id);

    var blanksHtml = blanks.length
      ? '<div class="blanks"><span class="eyebrow">Only you can fill these</span>' + blanks.map(function (p, k) {
          return '<label>' + esc(p) + '<input type="text" data-fill="' + key + '" data-k="' + k + '" placeholder="Your words, in the letter as written here"></label>';
        }).join('') + '<span class="counter">Left empty, the blank stays in the letter and it is held from sending until it is filled.</span></div>'
      : '';
    var blockingHtml = blocking.length && !blanks.length
      ? '<p class="basis"><b>Before it can be sent:</b> ' + blocking.map(esc).join(' · ') + '</p>' : '';

    var stamp = '';
    if (settled === 'approve') stamp = 'Released to your outbox. Post it by a method that proves delivery, then tell Minutes the date the district received it — that is the date the response clock runs from. <a href="#/case/' + attr(encodeURIComponent(state.id)) + '/letters">Open the outbox.</a>';
    if (settled === 'decline') stamp = 'Declined. The letter is kept whole on the record, nothing has left, and Minutes will not ask again about this one.';

    return '<article class="card' + (settled ? ' done' : '') + '" data-card="' + key + '">' +
      '<div class="card-head"><h3>' + esc(title) + '</h3>' + urgencyChip(card && card.urgency, kindLabel(reason.letter_kind)) + '</div>' +
      (why ? '<p class="why">' + esc(why) + '</p>' : '') +
      (facts.length ? '<ul class="facts">' + facts.map(function (f) { return '<li>' + esc(f) + '</li>'; }).join('') + '</ul>' : '') +
      (action ? '<p class="action"><b>Recommended:</b> ' + esc(action) + '</p>' : '') +
      (reason.body || draft.body
        ? letterDetails({ subject: reason.subject || draft.subject, body: reason.body || draft.body,
            citations: draft.citations, citationCount: typeof reason.citations === 'number' ? reason.citations : 0,
            basis: reason.legal_basis || draft.legal_basis, open: index === 0 })
        : '<p class="basis">The runtime sent no letter text with this decision.</p>') + // reason.body missing
      blanksHtml + blockingHtml +
      '<div class="decide"><button type="button" class="primary" data-approve="' + key + '">Release to my outbox</button>' +
      '<button type="button" data-decline="' + key + '">Decline</button>' +
      '<span class="note">Releasing does not send. You post it.</span></div>' +
      '<div class="state-box error card-error" hidden></div>' +
      '<div class="stamp' + (settled === 'decline' ? ' declined' : '') + '">' + stamp + '</div></article>';
  }

  function infoCardHtml(card, note) {
    // A card the wake raised that is not (or no longer) waiting on an answer.
    var d = card.draft;
    return '<article class="card done">' +
      '<div class="card-head"><h3>' + esc(card.title) + '</h3>' + urgencyChip(card.urgency) + '</div>' +
      (card.why_now ? '<p class="why">' + esc(card.why_now) + '</p>' : '') +
      ((card.facts || []).length ? '<ul class="facts">' + card.facts.map(function (f) { return '<li>' + esc(f) + '</li>'; }).join('') + '</ul>' : '') +
      (card.recommended_action ? '<p class="action"><b>Recommended:</b> ' + esc(card.recommended_action) + '</p>' : '') +
      (d ? letterDetails({ subject: d.subject, body: d.body, citations: d.citations, basis: d.legal_basis, open: false }) : '') +
      '<div class="stamp">' + esc(note) + '</div></article>';
  }

  function renderDecisions(state) {
    var box = document.getElementById('decisions');
    var pending = state.interrupts.filter(function (i) { return !state.settled[i.id]; });
    var answered = state.interrupts.filter(function (i) { return state.settled[i.id]; });
    var cards = state.wake.new_cards || [];
    var matched = state.interrupts.map(function (i) { return cardFor(state, i); }).filter(Boolean);
    var unmatched = cards.filter(function (c) { return matched.indexOf(c) === -1; });

    var parts = [];
    if (pending.length || answered.length) {
      parts.push('<section><h2>Decisions</h2><p class="lede">Each one is a letter Minutes compiled from the IEP and your records. Nothing leaves until you say so, and you post it yourself, by a method that proves delivery.</p><div class="cards">' +
        pending.concat(answered).map(function (i, k) { return decisionCardHtml(state, i, k); }).join('') + '</div></section>');
    }
    if (unmatched.length) {
      // A card with a draft that no interrupt carries: either its letter is
      // next in line (the caseworker puts one letter to you at a time), or
      // the caseworker finished without putting it. A card without a draft
      // never carried a letter.
      var note;
      if (pending.length) note = 'Next in line. Minutes puts one letter to you at a time; answer the one above and this one is compiled and put to you.';
      else if (state.finished || (state.wake.approval && state.wake.approval.status === 'done')) note = 'The caseworker finished without putting this letter to you' + (state.lastMessage ? ': “' + state.lastMessage + '”' : '.') + ' Minutes does not repeat a card it already raised; this one comes back when a later check finds it has got worse.';
      else note = 'Its letter has not been compiled yet. Check again and Minutes puts it to you.';
      parts.push('<section><h2>' + (pending.length || answered.length ? 'Also this week' : 'This week') + '</h2><div class="cards">' +
        unmatched.map(function (c) { return infoCardHtml(c, c.draft ? note : 'No letter goes with this; it is here so that you know.'); }).join('') + '</div></section>');
    }
    box.innerHTML = parts.join('');

    // The headline follows what is still open.
    var open = pending.length;
    var head = document.getElementById('headline');
    if (state.interrupts.length) {
      head.innerHTML = open ? headlineHtml(open + ' ' + (open === 1 ? 'decision needs' : 'decisions need') + ' you.') : headlineHtml('', true);
    }
    var waitingLine = document.getElementById('waiting-line');
    if (waitingLine && !open) waitingLine.textContent = 'Every letter put to you has been answered.';

    box.querySelectorAll('[data-approve]').forEach(function (b) { b.addEventListener('click', function () { answer(state, b.getAttribute('data-approve'), 'approve'); }); });
    box.querySelectorAll('[data-decline]').forEach(function (b) { b.addEventListener('click', function () { answer(state, b.getAttribute('data-decline'), 'decline'); }); });
  }

  function answer(state, interruptId, decision) {
    var card = document.querySelector('[data-card="' + CSS.escape(interruptId) + '"]');
    var errBox = card.querySelector('.card-error');
    var buttons = card.querySelectorAll('.decide button');
    buttons.forEach(function (b) { b.disabled = true; });
    errBox.hidden = true;

    var response = decision;
    if (decision === 'approve') {
      var fills = Array.prototype.map.call(card.querySelectorAll('[data-fill]'), function (i) { return i.value.trim(); });
      if (fills.some(Boolean)) response = { answer: 'approve', fill: fills }; // the runtime's dict form, for the blanks
    }
    var answers = {}; answers[interruptId] = response;
    var subject = (function () { var i = state.interrupts.filter(function (x) { return x.id === interruptId; })[0]; return i && i.reason ? i.reason.subject : ''; })();
    // The runtime has been seen to record an answer and then not return
    // (the release is persisted before the caseworker goes on to the next
    // letter). After the timeout the outbox and the declined list say what
    // was recorded, and the card follows them.
    call({ action: 'answer', case_id: state.id, answers: answers }, { timeoutMs: 150000 }).catch(function (err) {
      if (!err.timedOut || !subject) throw err;
      return Promise.all([
        call({ action: 'outbox', case_id: state.id }), call({ action: 'declined', case_id: state.id })
      ]).then(function (both) {
        var released = (both[0].outbox || []).some(function (l) { return l.subject === subject; });
        var declined = (both[1].declined || []).some(function (l) { return l.subject === subject; });
        if (!released && !declined) throw err;
        return { status: 'done', recovered: true, message: 'Minutes recorded your answer but did not come back to this page in time. If the next letter is ready, checking again will show it.' };
      });
    }).then(function (res) {
      if (res.status === 'error') {
        errBox.innerHTML = '<b>' + esc(res.error || 'Minutes could not record the answer.') + '</b><div class="what">Nothing was released. Try once more; if it fails again, check again from the top of this page.</div>';
        errBox.hidden = false; buttons.forEach(function (b) { b.disabled = false; }); return;
      }
      state.settled[interruptId] = decision;
      if (res.status === 'awaiting_approval') {
        // The next letter, or the same ones re-put. Keep what was answered;
        // adopt whatever is pending now.
        var known = state.interrupts.slice();
        (res.interrupts || []).forEach(function (i) {
          if (!known.some(function (k) { return k.id === i.id; })) known.push(i);
        });
        var stillPending = (res.interrupts || []).map(function (i) { return i.id; });
        // Anything neither answered nor re-put has settled on the runtime's side.
        state.interrupts = known.filter(function (i) { return state.settled[i.id] || stillPending.indexOf(i.id) !== -1; });
      } else {
        state.lastMessage = res.message || '';
        state.finished = !res.recovered;
        // done: every pending gate closed. Unanswered ones were declined by
        // the runtime's own rule (anything not an approval is a decline).
        state.interrupts.forEach(function (i) { if (!state.settled[i.id]) state.settled[i.id] = 'decline'; });
      }
      writePending(state);
      renderDecisions(state);
      if (res.recovered) toast(res.message);
      toast(decision === 'approve' ? 'Released to your outbox. Nothing has been sent.' : 'Declined and kept on the record.');
    }).catch(function (err) {
      errBox.innerHTML = '<b>' + esc(err.message) + '</b>' + (err.what ? '<div class="what">' + esc(err.what) + '</div>' : '');
      errBox.hidden = false; buttons.forEach(function (b) { b.disabled = false; });
    });
  }

  // -------------------------------------------------------------------------
  // Evidence
  // -------------------------------------------------------------------------

  function viewEvidence(id, guard) {
    var body = '<section class="tight"><div class="eyebrow">Evidence</div><h1>What is on file.</h1>' +
      '<p class="lede">Every dated fact this case rests on, newest first, each with where it came from. The district’s own records are marked as such; your notes are yours, dated, and never passed off as the school’s.</p>' +
      '<div id="evidence-list">' + loadingHtml('Reading the file.', 'Evidence') + '</div></section>' +
      '<section><h2>Items on file</h2><p class="lede">What each fact above was read from: school messages, returned logs, progress reports and your own notes.</p><div id="items-list"></div></section>' +
      '<section><h2>Add to the file</h2><div id="forms"></div></section>';
    withCase(id, 'evidence', 'Evidence', body, guard, function (ctx) {
      setMain(body);
      loadEvidence(id, guard);
      renderEvidenceForms(id, guard, ctx);
    });
  }

  function loadEvidence(id, guard) {
    call({ action: 'list_evidence', case_id: id }).then(function (res) {
      if (!guard()) return;
      var box = document.getElementById('evidence-list');
      if (res.status === 'error') { box.innerHTML = errorHtml(errorFrom(res, 'list_evidence')); return; }
      // {events: [ServiceEvent], correspondence: [Correspondence]}, both newest first.
      var list = res.events;
      if (!Array.isArray(list)) { box.innerHTML = errorHtml(new ApiError('The reply carried no evidence list.', 'Fields received: ' + Object.keys(res).join(', ') + '. Expected `events`.')); return; }
      if (!list.length) box.innerHTML = emptyHtml('Nothing on file yet.', 'A quick note below takes ten seconds and is on the record at once.');
      else {
        list = list.slice().sort(function (a, b) { return String(b.event_date).localeCompare(String(a.event_date)); });
        box.innerHTML = '<ul class="rows evidence">' + list.map(function (e) {
          var what = e.delivered ? fmtMin(e.minutes) + ' minutes delivered' : 'session missed';
          var tags = provChip(e.provenance) + (e.attribution === 'student_absence' ? '<span class="prov neutral">Student absent</span>' : '');
          return '<li><span class="d">' + esc(fmtDate(e.event_date)) + '</span><span>' + esc(e.service || '') + ' · ' + esc(what) +
            (e.source ? '<span class="sub">' + esc(sourceLabel(e.source)) + '</span>' : '') + '</span><span class="tags">' + tags + '</span></li>';
        }).join('') + '</ul>' +
          '<div class="legend"><span><i class="sw" style="background:var(--slate-tint);border-color:transparent"></i>School record: the district’s own record or written statement</span><span><i class="sw"></i>Your note: dated, but not the school’s record</span><span><i class="sw" style="border-color:var(--ink-3)"></i>Documented silence: records were asked for and not produced</span></div>';
      }
      var items = Array.isArray(res.correspondence) ? res.correspondence : []; // absent on an older runtime: section stays empty
      var ibox = document.getElementById('items-list');
      if (!items.length) ibox.innerHTML = emptyHtml('No items on file.');
      else {
        var KIND = { school_email: 'School email', parent_log: 'Your note', progress_report: 'Progress report', service_log: 'Service log' };
        ibox.innerHTML = '<ul class="rows evidence">' + items.map(function (it) {
          var chip = it.kind === 'parent_log' ? '<span class="prov parent">Your note</span>' : '<span class="prov">' + esc(KIND[it.kind] || it.kind) + '</span>';
          return '<li><span class="d">' + esc(fmtDate(it.received)) + '</span><span>' + esc(it.subject || KIND[it.kind] || '') + '<span class="sub">' + esc(it.sender || '') + (it.item_id ? ' · ' + esc(it.item_id) : '') + '</span>' +
            '<details class="kept"><summary>The text</summary><div class="paper" style="font-size:15px;padding:16px 18px">' + esc(it.body || '') + '</div></details></span><span class="tags">' + chip + '</span></li>';
        }).join('') + '</ul>';
      }
    }).catch(function (err) { if (guard()) document.getElementById('evidence-list').innerHTML = errorHtml(err); });
  }

  function fillServiceSelect(sel, services) {
    sel.innerHTML = services.map(function (s) { return '<option value="' + attr(s) + '">' + esc(s) + '</option>'; }).join('');
  }

  function renderEvidenceForms(id, guard, ctx) {
    var services = ctx.services || [];
    if (ctx.sample) {
      // The runtime refuses writes to the sample; say so instead of offering a form that will fail.
      document.getElementById('forms').innerHTML = '<div class="state-box"><div>The sample case is read-only.</div><div class="what">Its file is a fictional semester that ships with Minutes. To add notes and correspondence, <a href="#/case/new">start your own case</a> — it takes the IEP as text.</div></div>';
      return;
    }
    var serviceField = services.length
      ? '<select id="note-service" name="service" required></select>'
      : '<input id="note-service" name="service" required placeholder="As the IEP names it, e.g. Speech-Language Therapy">'; // no ledger services known
    document.getElementById('forms').innerHTML =
      '<form class="sheet" id="note-form"><h3>A quick note</h3>' +
      '<p class="lede" style="margin:0">What you saw, dated. It goes on file as your own observation the moment you save it; nothing here is read by a model.</p>' +
      '<div class="fieldrow"><label class="field"><span>Date</span><input type="date" name="date" required value="' + attr(todayISO()) + '"></label>' +
      '<label class="field"><span>Service</span>' + serviceField + '</label></div>' +
      '<div class="radio" role="radiogroup" aria-label="What happened"><label><input type="radio" name="delivered" value="true" checked> It happened</label><label><input type="radio" name="delivered" value="false"> It was missed</label></div>' +
      '<div class="fieldrow"><label class="field"><span>Minutes <span class="hint">optional</span></span><input type="number" name="minutes" min="0" step="1" placeholder="as delivered"></label></div>' +
      '<label class="field"><span>What you saw <span class="hint">optional</span></span><textarea name="text" placeholder="e.g. Came home with the speech folder; the session was held."></textarea></label>' +
      '<div class="formfoot"><button type="submit" class="primary">Put it on file</button><span class="note" id="note-status"></span></div></form>' +

      '<form class="sheet" id="corr-form" style="margin-top:22px"><h3>Paste correspondence</h3>' +
      '<p class="lede" style="margin:0">An email from the school, a returned service log, a progress report. Minutes reads it and turns it into dated facts, each marked as the district’s own record. That reading uses a model and can take a minute.</p>' +
      '<div class="fieldrow"><label class="field"><span>Received on</span><input type="date" name="received" required value="' + attr(todayISO()) + '"></label>' +
      '<label class="field"><span>Kind</span><select name="kind"><option value="school_email">School email</option><option value="service_log">Service log</option><option value="progress_report">Progress report</option><option value="parent_log">Parent log</option></select></label></div>' +
      '<div class="fieldrow"><label class="field"><span>From</span><input name="sender" required placeholder="e.g. Case manager"></label>' +
      '<label class="field"><span>Subject</span><input name="subject" required placeholder="As it was titled"></label></div>' +
      '<label class="field"><span>The text</span><textarea name="body" required minlength="20" placeholder="Paste the message or the log as text."></textarea></label>' +
      '<div class="formfoot"><button type="submit" class="primary">Read it into the file</button><span class="note" id="corr-status"></span></div></form>' +
      '<div id="corr-result"></div>';

    var sel = document.getElementById('note-service');
    if (sel.tagName === 'SELECT') fillServiceSelect(sel, services);

    document.getElementById('note-form').addEventListener('submit', function (e) {
      e.preventDefault();
      var f = e.target, status = document.getElementById('note-status'), btn = f.querySelector('button');
      var payload = { action: 'add_note', case_id: id, date: f.date.value, service: f.service.value.trim(), delivered: f.delivered.value === 'true' };
      if (f.minutes.value !== '') payload.minutes = parseInt(f.minutes.value, 10);
      if (f.text.value.trim()) payload.text = f.text.value.trim();
      btn.disabled = true; status.textContent = 'Saving…';
      call(payload).then(function (res) {
        btn.disabled = false;
        if (res.status === 'error') { status.textContent = ''; showFormError(f, errorFrom(res, 'add_note')); return; }
        // {event, item}: the fact reconciliation will count, and the note as written.
        var ev = res.event || {};
        status.textContent = 'On file' + (ev.source ? ' as ' + ev.source : '') + '.'; clearFormError(f); f.text.value = ''; f.minutes.value = '';
        toast('On file as your note.');
        loadEvidence(id, guard);
      }).catch(function (err) { btn.disabled = false; status.textContent = ''; showFormError(f, err); });
    });

    document.getElementById('corr-form').addEventListener('submit', function (e) {
      e.preventDefault();
      var f = e.target, status = document.getElementById('corr-status'), btn = f.querySelector('button'), result = document.getElementById('corr-result');
      var payload = { action: 'add_correspondence', case_id: id, received: f.received.value, kind: f.kind.value, sender: f.sender.value.trim(), subject: f.subject.value.trim(), body: f.body.value.trim() };
      btn.disabled = true; status.textContent = 'Reading… this can take a minute.'; result.innerHTML = '';
      call(payload).then(function (res) {
        btn.disabled = false; status.textContent = '';
        if (res.status === 'error') { showFormError(f, errorFrom(res, 'add_correspondence')); return; }
        clearFormError(f);
        // {item, events}: the item as kept, and the dated facts read out of it (often none).
        var events = res.events || [];
        result.innerHTML = '<div class="state-box" style="margin-top:12px"><div>Read. ' + (events.length ? events.length + ' dated ' + plural(events.length, 'fact') + ' went on file:' : 'No dated service fact was found in it; the item itself is kept.') + '</div>' +
          (events.length ? '<ul class="facts">' + events.map(function (ev) { return '<li>' + esc(fmtDate(ev.event_date)) + ' · ' + esc(ev.service) + ' · ' + (ev.delivered ? fmtMin(ev.minutes) + ' minutes' : 'missed') + '</li>'; }).join('') + '</ul>' : '') + '</div>';
        f.body.value = ''; f.subject.value = '';
        loadEvidence(id, guard);
      }).catch(function (err) { btn.disabled = false; status.textContent = ''; showFormError(f, err); });
    });
  }
  function showFormError(form, err) {
    clearFormError(form);
    var box = document.createElement('div'); box.className = 'state-box error form-error'; box.setAttribute('role', 'alert');
    box.innerHTML = '<b>' + esc(err.message) + '</b>' + (err.what ? '<div class="what">' + esc(err.what) + '</div>' : '');
    form.appendChild(box);
  }
  function clearFormError(form) { var b = form.querySelector('.form-error'); if (b) b.remove(); }

  // -------------------------------------------------------------------------
  // Statement
  // -------------------------------------------------------------------------

  function viewStatement(id, guard) {
    var waiting = '<section class="tight"><div class="eyebrow">Statement</div><h1>What was owed, what was delivered, and how each minute is evidenced.</h1>' + loadingHtml('Reading the ledger.', 'Statement') + '</section>';
    withCase(id, 'statement', 'Statement', waiting, guard, function (ctx) {
      var start = ctx.ledgerStart || schoolYearStart();
      var end = asOf(id) || todayISO();
      if (id === SAMPLE_CASE && !asOf(id) && end < start) end = SAMPLE_REPLAY; // the sample's term has not started on the real date
      if (end < start) end = start; // a case whose services have not begun: a one-day period, which the statement reports as nothing scheduled
      setMain('<section class="tight"><div class="eyebrow">Statement</div><h1>What was owed, what was delivered, and how each minute is evidenced.</h1>' +
        '<div class="controls"><label class="field"><span>From</span><input type="date" id="st-start" value="' + attr(start) + '"></label>' +
        '<label class="field"><span>To</span><input type="date" id="st-end" value="' + attr(end) + '"></label>' +
        '<button type="button" class="small" id="st-build">Build the statement</button>' +
        '<button type="button" class="small" id="st-print" hidden>Print</button></div>' +
        '<div id="statement">' + loadingHtml('Reconciling the ledger against the file. No model is involved.', 'Statement') + '</div></section>');
      var build = function () { loadStatement(id, document.getElementById('st-start').value, document.getElementById('st-end').value, guard); };
      document.getElementById('st-build').addEventListener('click', build);
      document.getElementById('st-print').addEventListener('click', function () { window.print(); });
      build();
    });
  }

  function loadStatement(id, start, end, guard) {
    var box = document.getElementById('statement');
    box.innerHTML = loadingHtml('Reconciling the ledger against the file. No model is involved.', 'Statement');
    if (!start || !end) { box.innerHTML = errorHtml(new ApiError('Both dates are needed.', 'Pick the first and last day of the period.')); return; }
    if (end < start) { box.innerHTML = errorHtml(new ApiError('The period ends before it starts.', 'Pick a "to" date on or after ' + fmtDate(start) + '.')); return; }
    var payload = { action: 'statement', case_id: id, start: start, end: end };
    if (asOf(id)) payload.today = asOf(id);
    call(payload).then(function (res) {
      if (!guard()) return;
      if (res.status === 'error') { box.innerHTML = errorHtml(errorFrom(res, 'statement')); return; }
      var st = res.statement || res;
      if (st.student) saveCtx(id, { alias: st.student });
      box.innerHTML = statementHtml(st, start, end);
      document.getElementById('st-print').hidden = false;
    }).catch(function (err) { if (guard()) box.innerHTML = errorHtml(err); });
  }

  function statementHtml(st, start, end) {
    var md = st.markdown || '';
    var totals = st.totals || {};
    // The runtime's statement JSON carries totals and the rendered markdown,
    // not the per-service lines; the ruled table below is parsed out of the
    // markdown's own tables so the two can never disagree.
    var tables = mdTables(md);
    var services = (tables['Services'] || []).slice(1);
    var evidence = (tables['Where the evidence comes from'] || []).slice(1);
    var byService = {};
    evidence.forEach(function (r) { byService[r[0]] = { confirmed: cellInt(r[1]), parent: cellInt(r[2]), undoc: cellInt(r[3]) }; });

    var period = mdSectionText(md, 'This period');
    var headline = (period[0] || '').replace(/^\*\*|\*\*$/g, '');
    var hm = /^(.*?short this period),\s*(.*)$/.exec(headline);
    var headlineHtmlStr = hm ? esc(hm[1]) + ' — <span class="undoc-n">' + esc(hm[2]) + '</span>' : esc(headline || (fmtMin(totals.shortfall_minutes) + ' minutes short this period.'));
    var lede = period.slice(1).join(' ');

    var rows = services.filter(function (r) { return !/^total$/i.test(r[0]); }).map(function (r) {
      var ev = byService[r[0]] || { undoc: 0 };
      return '<tr><td>' + esc(r[0]) + '</td>' + minCell(cellInt(r[1])) + minCell(cellInt(r[2])) + minCell(cellInt(r[3])) + minCell(cellInt(r[4])) + minCell(ev.undoc, { undoc: true }) + '</tr>';
    });
    var totalRow = services.filter(function (r) { return /^total$/i.test(r[0]); })[0];
    var evTotal = byService['TOTAL'] || byService['Total'] || { undoc: 0 };
    if (totalRow) {
      rows.push('<tr class="total"><td>Total</td>' + minCell(totals.owed_minutes != null ? totals.owed_minutes : cellInt(totalRow[1])) +
        minCell(totals.delivered_minutes != null ? totals.delivered_minutes : cellInt(totalRow[2])) + minCell(cellInt(totalRow[3])) +
        minCell(totals.shortfall_minutes != null ? totals.shortfall_minutes : cellInt(totalRow[4])) + minCell(evTotal.undoc, { undoc: true }) + '</tr>');
    }

    var table = rows.length
      ? '<div class="tablewrap"><table><thead><tr><th>Service</th><th>Owed</th><th>Delivered</th><th>Excused</th><th>Short</th><th>of which<br>no record</th></tr></thead><tbody>' + rows.join('') + '</tbody></table></div>' +
        '<div class="legend"><span><i class="sw hatch"></i>No record either way — nobody wrote anything down</span></div>'
      : (totals.owed_minutes != null
          ? '<ul class="checked"><li>' + fmtMin(totals.owed_minutes) + ' minutes owed.</li><li>' + fmtMin(totals.delivered_minutes) + ' documented as delivered.</li><li>' + fmtMin(totals.shortfall_minutes) + ' short.</li></ul><p class="basis">The per-service table could not be read from the statement’s text; totals are shown from its figures.</p>'
          : emptyHtml('The reply carried no figures.', 'Expected `statement.totals` and `statement.markdown`.'));

    var evTable = evidence.length
      ? '<div class="tablewrap"><table><thead><tr><th>Service</th><th>School record</th><th>Your notes</th><th>No record</th></tr></thead><tbody>' +
        evidence.map(function (r) {
          var total = /^total$/i.test(r[0]);
          return '<tr' + (total ? ' class="total"' : '') + '><td>' + esc(total ? 'Total' : r[0]) + '</td>' + minCell(cellInt(r[1])) + minCell(cellInt(r[2])) + minCell(cellInt(r[3]), { undoc: true }) + '</tr>';
        }).join('') + '</tbody></table></div>'
      : '';

    return '<div class="week" style="padding-top:8px"><div class="eyebrow">' + esc(fmtDate(start)) + ' to ' + esc(fmtDate(end)) + (st.today ? ' · as of ' + esc(fmtDate(st.today)) : '') + '</div>' +
      '<p class="headline-figure">' + headlineHtmlStr + '</p>' + (lede ? '<p class="lede">' + esc(lede) + '</p>' : '') + '</div>' +
      table +
      (evTable ? '<section class="tight"><h2>Where the evidence comes from</h2><p class="lede">School-record minutes come from the district’s own records or written statements. Your notes are dated but are not the school’s record. Minutes with no record either way are a gap in the evidence, not a record of non-delivery; the answer to them is a records request.</p>' + evTable + '</section>' : '') +
      (md ? '<section><h2>The statement, as it prints</h2><p class="lede">This is the document itself. Print it, or forward it as it is.</p><div class="doc printable">' + mdToHtml(md) + '</div></section>' : '');
  }

  // -------------------------------------------------------------------------
  // Letters: the outbox and the declined
  // -------------------------------------------------------------------------

  function viewLetters(id, guard) {
    var body = '<section class="tight"><div class="eyebrow">Letters</div><h1>Your outbox.</h1>' +
      '<p class="lede">Letters you released wait here for you to post. Minutes cannot send anything. Post each one by a method that proves delivery, and for a records request tell Minutes the date the district received it: the 45-day response clock runs from that date and from nothing else.</p>' +
      '<div id="outbox">' + loadingHtml('Reading the outbox.', 'Outbox') + '</div></section>' +
      '<section><h2>Declined, kept whole</h2><p class="lede">A letter you read and chose not to send stays on the record exactly as you saw it.</p><div id="declined">' + loadingHtml('Reading the record.', 'Declined') + '</div></section>';
    withCase(id, 'letters', 'Letters', body, guard, function () { setMain(body); loadLetters(id, guard); });
  }
  function loadLetters(id, guard) {
    Promise.all([
      call({ action: 'outbox', case_id: id }),
      call({ action: 'declined', case_id: id }),
      call({ action: 'requests', case_id: id }).catch(function () { return { status: 'error' }; }) // optional: the clocks
    ]).then(function (all) {
      if (!guard()) return;
      var out = all[0], dec = all[1], req = all[2];
      var requests = {};
      (req.requests || []).forEach(function (r) { requests[r.request_id] = r; });
      renderOutbox(id, out, requests);
      var dbox = document.getElementById('declined');
      if (dec.status === 'error') { dbox.innerHTML = errorHtml(errorFrom(dec, 'declined')); return; }
      var list = dec.declined || [];
      if (!list.length) { dbox.innerHTML = emptyHtml('Nothing declined.'); return; }
      dbox.innerHTML = '<div class="letters">' + list.map(function (l) {
        return '<div class="letteritem"><details class="kept"><summary>' + esc(kindLabel(l.kind)) + ' · declined ' + esc(fmtDate(l.declined_on)) + ' · ' + esc(l.subject || '') + '</summary>' +
          '<div class="paper">' + letterBodyHtml(l.body || '') + '</div></details></div>';
      }).join('') + '</div>';
    }).catch(function (err) { if (guard()) { document.getElementById('outbox').innerHTML = errorHtml(err); document.getElementById('declined').innerHTML = ''; } });
  }

  function renderOutbox(id, out, requests) {
    var box = document.getElementById('outbox');
    if (out.status === 'error') { box.innerHTML = errorHtml(errorFrom(out, 'outbox')); return; }
    var list = out.outbox || [];
    if (!list.length) { box.innerHTML = emptyHtml('Nothing released yet.', 'When you release a letter on This week, it waits here.'); return; }
    box.innerHTML = '<div class="letters">' + list.slice().reverse().map(function (l, k) {
      var ref = l.reference || '';
      var isRequest = l.kind === 'records_request' && ref;
      var known = requests[ref];
      var clock = '';
      if (isRequest) {
        if (known && known.sent_on) {
          clock = '<div class="received"><span>The district received it on <b>' + esc(fmtLongDate(known.sent_on)) + '</b>.' +
            (known.response_due ? ' A response is due by <b>' + esc(fmtLongDate(known.response_due)) + '</b>.' : '') +
            (known.state === 'unanswered_overdue' ? ' <span class="state overdue">Unanswered past the due date</span>' : '') +
            (known.answered_on ? ' Answered ' + esc(fmtDate(known.answered_on)) + '.' : '') + '</span></div>';
        } else {
          // No max on the date: the runtime refuses a date before the release, and the sample is replayed at dates past the real one.
          clock = '<form class="received" data-ref="' + attr(ref) + '"><label>The district received it on<input type="date" name="received_on" required></label>' +
            '<button type="submit" class="small">Start the response clock</button><span class="note" style="font-size:12.5px;color:var(--ink-3)">From your delivery proof, not from the day you posted it.</span></form>';
        }
      }
      return '<div class="letteritem" data-k="' + k + '"><h3>' + esc(l.subject || kindLabel(l.kind)) + '</h3>' +
        '<div class="meta"><span>' + esc(kindLabel(l.kind)) + ' · released ' + esc(fmtDate(l.released_on)) + (ref ? ' · ' + esc(ref) : '') + (l.citations ? ' · ' + l.citations + ' cited ' + plural(l.citations, 'fact') : '') + '</span>' +
        '<span class="tools"><button type="button" class="small" data-copy="' + k + '">Copy the letter</button><button type="button" class="small" data-print="' + k + '">Print</button></span></div>' +
        '<details class="letter"><summary><span>The letter</span><span class="toggle"></span></summary><div class="paper">' + letterBodyHtml(l.body || '') + '</div></details>' +
        clock + '<div class="clock-note"></div></div>';
    }).join('') + '</div>';

    var items = list.slice().reverse();
    box.querySelectorAll('[data-copy]').forEach(function (b) { b.addEventListener('click', function () { copyText(items[+b.getAttribute('data-copy')].body || ''); }); });
    box.querySelectorAll('[data-print]').forEach(function (b) { b.addEventListener('click', function () { var l = items[+b.getAttribute('data-print')]; printLetter(l.subject, l.body || ''); }); });
    box.querySelectorAll('form.received').forEach(function (f) {
      f.addEventListener('submit', function (e) {
        e.preventDefault();
        var ref = f.getAttribute('data-ref'), btn = f.querySelector('button'), note = f.parentNode.querySelector('.clock-note');
        btn.disabled = true;
        call({ action: 'mark_received', case_id: id, request_id: ref, received_on: f.received_on.value }).then(function (res) {
          btn.disabled = false;
          // A refusal (never released, already running, a date before the
          // release) comes back as status "error" with the rule it broke.
          if (res.status === 'error') { note.innerHTML = errorHtml(new ApiError(res.error || 'Not recorded.', 'Nothing changed. The date must be the one on your delivery proof, on or after the day the letter was released.')); return; }
          var r = res.request || {}; // {request: RecordsRequest} with sent_on and response_due
          note.innerHTML = '<div class="state-box"><div>Recorded. The district received it on <b>' + esc(fmtLongDate(r.sent_on || f.received_on.value)) + '</b>.' +
            (r.response_due ? ' A response is due by <b>' + esc(fmtLongDate(r.response_due)) + '</b>.' : '') + ' If nothing comes by then, the silence goes on file as a dated fact.</div></div>';
          f.hidden = true;
          toast('The response clock is running.');
        }).catch(function (err) { btn.disabled = false; note.innerHTML = errorHtml(err); });
      });
    });
  }

  // -------------------------------------------------------------------------
  // Dates the IEP sets
  // -------------------------------------------------------------------------

  function viewDates(id, guard) {
    var body = '<section class="tight"><div class="eyebrow">Dates</div><h1>Dates the IEP sets.</h1>' +
      '<p class="lede">Each one as the document states it, with how far off it is' + (asOf(id) ? ' as of ' + esc(fmtLongDate(asOf(id))) : '') + '. A date that has passed with nothing on file is where a letter starts.</p>' +
      '<div id="dates">' + loadingHtml('Reading the dates.', 'Dates') + '</div></section>';
    withCase(id, 'dates', 'Dates', body, guard, function () {
      setMain(body);
      var payload = { action: 'deadlines', case_id: id };
      if (asOf(id)) payload.today = asOf(id);
      call(payload).then(function (res) {
        if (!guard()) return;
        var box = document.getElementById('dates');
        if (res.status === 'error') { box.innerHTML = errorHtml(errorFrom(res, 'deadlines')); return; }
        // {deadlines: [{kind, due, description, state, days_remaining}]}; the
        // runtime sends no source quote or met_on here, so neither is shown.
        var list = res.deadlines;
        if (!Array.isArray(list)) { box.innerHTML = errorHtml(new ApiError('The reply carried no dates.', 'Fields received: ' + Object.keys(res).join(', ') + '. Expected `deadlines`.')); return; }
        box.innerHTML = datesHtml(list.map(function (r) {
          return { due: r.due, what: r.description, quote: r.source_quote, state: r.state, days: r.days_remaining, met_on: r.met_on };
        }));
      }).catch(function (err) { if (guard()) document.getElementById('dates').innerHTML = errorHtml(err); });
    });
  }
  function datesHtml(rows) {
    if (!rows.length) return emptyHtml('The IEP sets no dates that Minutes could read.');
    rows.sort(function (a, b) { return String(a.due).localeCompare(String(b.due)); });
    return '<ul class="rows">' + rows.map(function (r) {
      var chip, cls = '';
      var days = typeof r.days === 'number' ? r.days : null;
      if (r.state === 'met') { chip = 'Met' + (r.met_on ? ' ' + fmtDate(r.met_on) : ''); cls = ' met'; }
      else if (r.state === 'overdue' || (days != null && days < 0)) { chip = Math.abs(days) + ' ' + plural(Math.abs(days), 'day') + ' past'; cls = ' overdue'; }
      else if (r.state === 'due_soon') { chip = days + ' ' + plural(days, 'day') + ' · due soon'; cls = ' soon'; }
      else if (days != null) { chip = days + ' ' + plural(days, 'day'); }
      else chip = r.statusText || r.state || '';
      var due = /^\d{4}-\d{2}-\d{2}/.test(String(r.due)) ? fmtDate(r.due) : r.due;
      return '<li><span class="d">' + esc(due) + '</span><span>' + esc(r.what || '') + (r.quote ? '<span class="sub quote">“' + esc(r.quote) + '”</span>' : '') + '</span><span class="state' + cls + '">' + esc(chip) + '</span></li>';
    }).join('') + '</ul>';
  }

  // -------------------------------------------------------------------------
  // What Minutes did
  // -------------------------------------------------------------------------

  function viewAudit(id, guard) {
    var body = '<section class="tight"><div class="eyebrow">What Minutes did</div><h1>Every action, as it happened.</h1>' +
      '<p class="lede">The paper trail is the product. Newest first.</p><div id="audit">' + loadingHtml('Reading the trail.', 'Audit') + '</div></section>';
    withCase(id, 'audit', 'What Minutes did', body, guard, function () { setMain(body); loadAudit(id, guard); });
  }
  function loadAudit(id, guard) {
    call({ action: 'audit', case_id: id }).then(function (res) {
      if (!guard()) return;
      var box = document.getElementById('audit');
      if (res.status === 'error') { box.innerHTML = errorHtml(errorFrom(res, 'audit')); return; }
      var list = res.audit;
      if (!Array.isArray(list)) { box.innerHTML = errorHtml(new ApiError('The reply carried no trail.', 'Fields received: ' + Object.keys(res).join(', ') + '. Expected `audit`.')); return; }
      if (!list.length) { box.innerHTML = emptyHtml('Nothing recorded yet on this case.', 'The first check writes the first line.'); return; }
      box.innerHTML = '<ul class="rows audit">' + list.slice().reverse().map(function (e) {
        var ev = e.evidence;
        return '<li><span class="d">' + esc(fmtDate(e.entry_date)) + '</span><span>' + esc(capital(e.action || '')) + ' <span class="who">' + esc(e.actor || '') + '</span>' +
          (e.detail ? '<span class="sub">' + esc(e.detail) + '</span>' : '') +
          (ev ? '<span class="sub">' + provChip(ev.provenance) + ' ' + esc(ev.source || '') + (ev.detail ? ' · ' + esc(ev.detail) : '') + '</span>' : '') + '</span></li>';
      }).join('') + '</ul>';
    }).catch(function (err) { if (guard()) document.getElementById('audit').innerHTML = errorHtml(err); });
  }
  function capital(s) { s = String(s); return s.charAt(0).toUpperCase() + s.slice(1); }

  // -------------------------------------------------------------------------
  // Settings sheet
  // -------------------------------------------------------------------------

  var dialog = document.getElementById('settings');
  function openSettings() {
    document.getElementById('s-api').value = apiBase();
    document.getElementById('s-key').value = apiKey();
    document.getElementById('s-theme').value = theme();
    document.getElementById('s-note').textContent = '';
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
  }
  dialog.addEventListener('close', function () {
    if (dialog.returnValue !== 'save') return;
    var base = document.getElementById('s-api').value.trim() || '/api';
    var key = document.getElementById('s-key').value.trim();
    store.set('minutes.apiBase', base);
    if (key) store.set('minutes.key', key); else store.del('minutes.key');
    setTheme(document.getElementById('s-theme').value);
    render();
  });
  document.getElementById('s-theme').addEventListener('change', function (e) { setTheme(e.target.value); });
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-open-settings]');
    if (b) openSettings();
  });

  // -------------------------------------------------------------------------
  // Go
  // -------------------------------------------------------------------------

  window.addEventListener('hashchange', render);
  render();
})();
