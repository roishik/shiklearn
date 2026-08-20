"use strict";
// Markdown -> HTML for the agent's replies, plus the inverse for speech.
//
// Its own file for two reasons: it touches no DOM and no app state, so it
// is testable in isolation (tests/test_markdown_renderer.py runs this
// exact file under node and asserts on the output), and keeping the
// escape-first safety invariant in one small readable unit is worth more
// than saving an HTTP request. Loaded before the inline app script, so
// escapeHtml() is a global by the time anything else needs it.

function escapeHtml(s) {
  // NUL is stripped here so the claim made about CODE_SENTINEL below is
  // actually true rather than merely asserted. It is not a security
  // issue — a NUL in the source would collide with a code-span
  // placeholder and swap one already-escaped fragment for another — but
  // an invariant that is stated and not enforced is worse than no
  // invariant, because the next person relies on it.
  return String(s).replace(/\u0000/g, '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ── Markdown rendering ─────────────────────────────────────────────────
// The agent replies in markdown — headers, bold, bullet lists, the
// occasional comparison table — because that is what a chat-tuned model
// emits. This renders it, in ~90 dependency-free lines, rather than
// pulling marked.js off a CDN: the whole UI is one file you can open with
// no build step and no network, and a script tag would cost that for a
// subset of markdown covered here.
//
// SAFETY — the ordering is the security property, not a detail.
// escapeHtml() runs FIRST, once, over the entire source string. Every
// transform below therefore operates on text in which < and > are already
// entities, so nothing the model emits can become a live tag: the only
// real markup in the output is markup this function itself wrote. Link
// hrefs are the single place an attacker-controlled value lands inside an
// attribute, so schemes are allowlisted — a javascript: href would
// otherwise survive escaping intact. The model is not really the threat
// here; tool results quoted back into a reply are, and those originate in
// public data files this repo does not control.
const SAFE_URL_SCHEME = /^(?:https?:\/\/|mailto:)/i;
// NUL. escapeHtml strips it from the source, so it cannot occur in the
// text these placeholders are substituted into and a code span's
// placeholder can never collide with real content.
const CODE_SENTINEL = '\u0000';
const CODE_SENTINEL_RE = new RegExp(CODE_SENTINEL + '(\\d+)' + CODE_SENTINEL, 'g');

// Inline spans. Code spans are pulled out into placeholders first, so that
// a ** or _ inside a code span is never mistaken for emphasis.
function renderInline(text) {
  const codeSpans = [];
  let s = text.replace(/`([^`\n]+)`/g, (_m, code) => {
    codeSpans.push(code);
    return `${CODE_SENTINEL}${codeSpans.length - 1}${CODE_SENTINEL}`;
  });
  s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (whole, label, url) =>
    SAFE_URL_SCHEME.test(url)
      ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
      : whole // unsupported scheme: leave the literal markdown, render nothing clickable
  );
  s = s.replace(/\*\*([^\n]+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^\w\\])__([^\n]+?)__(?!\w)/g, '$1<strong>$2</strong>');
  s = s.replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  s = s.replace(/(^|[^_\w])_([^_\n]+?)_(?!\w)/g, '$1<em>$2</em>');
  s = s.replace(/~~([^\n]+?)~~/g, '<del>$1</del>');
  return s.replace(CODE_SENTINEL_RE, (_m, i) => `<code>${codeSpans[Number(i)]}</code>`);
}

// Bullet/numbered lists, including the one form of nesting models
// actually emit: leading indentation, two spaces per level. Ragged
// indentation rounds down to the nearest level rather than throwing the
// whole list away.
function renderList(rawLines) {
  const items = [];
  for (const raw of rawLines) {
    const m = raw.match(/^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$/);
    if (m) {
      items.push({
        depth: Math.floor(m[1].replace(/\t/g, '  ').length / 2),
        ordered: /^\s*\d/.test(raw),
        text: m[2],
      });
    } else if (items.length) {
      items[items.length - 1].text += '\n' + raw.trim(); // wrapped continuation line
    }
  }
  if (!items.length) return '';

  // Normalize against the shallowest item, not against the first one.
  // build() stops as soon as it meets an item shallower than the depth it
  // was called with, and never advances past it — so a list whose first
  // line is indented ("  - Ra'anana / - Kfar Saba") silently dropped every
  // item from the first outdent onward. In a ranked answer that is a
  // locality vanishing from the list with no error anywhere.
  const shallowest = Math.min(...items.map(i => i.depth));
  for (const item of items) item.depth -= shallowest;

  let pos = 0;
  function build(depth) {
    const tag = items[pos].ordered ? 'ol' : 'ul';
    let html = `<${tag}>`;
    while (pos < items.length && items[pos].depth >= depth) {
      // A deeper item with no sibling above it to hang off — a list whose
      // first line is indented. Wrap it in an <li> rather than emitting a
      // bare nested list, which is invalid markup, and never skip it,
      // which is what silently dropped items before.
      if (items[pos].depth > depth) { html += `<li>${build(items[pos].depth)}</li>`; continue; }
      const text = renderInline(items[pos].text).replace(/\n/g, '<br>');
      pos++;
      const nested = pos < items.length && items[pos].depth > depth ? build(items[pos].depth) : '';
      html += `<li>${text}${nested}</li>`;
    }
    return `${html}</${tag}>`;
  }
  // build(0), not build(items[0].depth): depths are normalized above so
  // the shallowest is 0, and starting at the FIRST item's depth is what
  // caused the truncation — build() exits at the first item shallower
  // than the depth it was given.
  return build(0);
}

function isTableSeparator(line) {
  return /^\s*\|?(\s*:?-{2,}:?\s*\|)+(\s*:?-{2,}:?\s*)?\|?\s*$/.test(line);
}

function splitTableRow(line) {
  return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map(c => c.trim());
}

function renderMarkdown(source) {
  // NOTE: the escape happens here, once, before anything else touches the
  // string. See the SAFETY comment above — this line is load-bearing.
  const lines = escapeHtml(String(source)).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let para = [];
  const flushParagraph = () => {
    if (!para.length) return;
    out.push(`<p>${renderInline(para.join('\n')).replace(/\n/g, '<br>')}</p>`);
    para = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/^\s*```/.test(line)) { // fenced code block
      flushParagraph();
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      out.push(`<pre><code>${body.join('\n')}</code></pre>`);
      continue;
    }
    if (!line.trim()) { flushParagraph(); continue; }
    if (/^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flushParagraph(); out.push('<hr>'); continue; }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(heading[2].trim())}</h${level}>`);
      continue;
    }

    if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushParagraph();
      const head = splitTableRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) rows.push(splitTableRow(lines[i++]));
      i--;
      out.push(
        '<table><thead><tr>' + head.map(c => `<th>${renderInline(c)}</th>`).join('') +
        '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + r.map(c => `<td>${renderInline(c)}</td>`).join('') + '</tr>').join('') +
        '</tbody></table>'
      );
      continue;
    }

    // '>' is already &gt; by the time we get here — see renderMarkdown's
    // first line. Matching the raw character would silently never fire.
    if (/^\s{0,3}&gt;\s?/.test(line)) {
      flushParagraph();
      const body = [];
      while (i < lines.length && /^\s{0,3}&gt;\s?/.test(lines[i])) body.push(lines[i++].replace(/^\s{0,3}&gt;\s?/, ''));
      i--;
      out.push(`<blockquote>${renderInline(body.join('\n')).replace(/\n/g, '<br>')}</blockquote>`);
      continue;
    }

    if (/^\s*(?:[-*+]|\d+[.)])\s+/.test(line)) {
      flushParagraph();
      const block = [];
      while (
        i < lines.length &&
        (/^\s*(?:[-*+]|\d+[.)])\s+/.test(lines[i]) || (block.length && lines[i].trim() && /^\s{2,}/.test(lines[i])))
      ) block.push(lines[i++]);
      i--;
      out.push(renderList(block));
      continue;
    }

    para.push(line);
  }
  flushParagraph();
  return out.join('\n');
}

// Markdown is for the eye. Spoken replies get the syntax stripped first —
// a synthesizer reads "**" as nothing useful, and a table read aloud
// pipe-by-pipe is worse than useless.
function stripMarkdown(text) {
  return String(text)
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/\[([^\]\n]+)\]\((?:[^()\s]|\([^()]*\))*\)/g, '$1')
    .replace(/^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$/gm, '')
    .replace(/(\*\*|__|~~)/g, '')
    .replace(/(^|\s)[*_]([^*_\n]+)[*_](?=[\s.,;:!?)]|$)/g, '$1$2')
    // Table rows: drop the outer pipes before turning the inner ones into
    // pauses, otherwise every row is read starting and ending with "comma".
    .replace(/^[ \t]*\|(.*)\|[ \t]*$/gm, '$1')
    .replace(/[ \t]*\|[ \t]*/g, ', ')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// Browser: these are plain globals on window, which is all the inline app
// script needs. Node (the test harness): CommonJS, so the same source
// file is what gets tested — not a copy that can drift away from it.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { escapeHtml, renderInline, renderList, renderMarkdown, stripMarkdown };
}
