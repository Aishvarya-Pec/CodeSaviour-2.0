document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('buggedCode');
  const outputBlock = document.getElementById('fixedCodeBlock');
  const fixBtn = document.getElementById('fixBtn');
  const lang = document.getElementById('language');
  const errorEl = document.getElementById('errorCount');
  const warnEl = document.getElementById('warningCount');
  const errorList = document.getElementById('errorList');
  const warningList = document.getElementById('warningList');
  const copyBtn = document.getElementById('copyCodeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const fixedLangEl = document.getElementById('fixedLang');

  function renderList(items, container, kind) {
    if (!container) return;
    const safeItems = Array.isArray(items) ? items : [];
    container.innerHTML = safeItems.map(it => {
      const lineLabel = (it && typeof it.line === 'number') ? `Line ${it.line}` : 'Line —';
      const msg = (it && it.message) ? it.message : '';
      return `<li class="${kind}"><span class="badge-line">${lineLabel}</span><span class="msg-text">${msg}</span></li>`;
    }).join('');
  }

  function updateAnalysis() {
    const code = input?.value || '';
    const language = lang?.value || '';
    // Always prefer backend analysis to align with OpenRouter
    fetch('http://127.0.0.1:8001/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language: language, code })
    }).then(async (res) => {
      if (!res.ok) throw new Error('analyze api failed');
      const data = await res.json();
      if (errorEl) errorEl.textContent = String(data.errors ?? 0);
      if (warnEl) warnEl.textContent = String(data.warnings ?? 0);
      renderList(data.error_items, errorList, 'error');
      renderList(data.warning_items, warningList, 'warn');
    }).catch(() => {
      const fallback = analyzeCode(language, code);
      if (errorEl) errorEl.textContent = String(fallback.errors);
      if (warnEl) warnEl.textContent = String(fallback.warnings);
      renderList(fallback.error_items, errorList, 'error');
      renderList(fallback.warning_items, warningList, 'warn');
    });
  }

  // Live analysis while typing and on language change
  if (input) input.addEventListener('input', updateAnalysis);
  if (lang) lang.addEventListener('change', updateAnalysis);
  // Initialize counts on load
  updateAnalysis();

  async function callFixAPI(language, code) {
    const res = await fetch('http://127.0.0.1:8001/api/fix', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language, code })
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Fix API error: ${res.status} ${text}`);
    }
    const data = await res.json();
    return data.fixed || '';
  }

  function basicFix(code) {
    const lines = code.split('\n').map(l => l.replace(/\s+$/g, ''));
    let fixed = lines.join('\n');
    fixed = fixed.replace(/\t/g, '    ').replace(/\r/g, '');
    return fixed;
  }

  if (fixBtn) {
    fixBtn.addEventListener('click', async () => {
      const code = input.value || '';
      const language = lang?.value || '';
      if (fixedLangEl) fixedLangEl.textContent = language ? ` ${language}` : ' —';
      // Update analysis before fixing
      updateAnalysis();

      // Notes section removed

      // Update code output state
      outputBlock.textContent = 'Contacting OpenRouter…';
      try {
        const fixed = await callFixAPI(language, code);
        outputBlock.textContent = fixed || basicFix(code);
      } catch (e) {
        outputBlock.textContent = basicFix(code);
      }
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const text = outputBlock?.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => (copyBtn.textContent = 'Copy'), 1200);
      } catch {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (input) input.value = '';
      if (outputBlock) outputBlock.textContent = '';
      if (fixedLangEl) fixedLangEl.textContent = ' —';
      if (errorEl) errorEl.textContent = '0';
      if (warnEl) warnEl.textContent = '0';
      if (errorList) errorList.innerHTML = '';
      if (warningList) warningList.innerHTML = '';
    });
  }

  // Download button removed
});

function analyzeCode(language, code) {
  const result = { errors: 0, warnings: 0, error_items: [], warning_items: [] };
  const lines = code.split('\n');

  // Balance checks for (), {}, []
  const pairs = [ ['(', ')'], ['{', '}'], ['[', ']'] ];
  for (const [open, close] of pairs) {
    let balance = 0;
    for (const ch of code) {
      if (ch === open) balance++;
      else if (ch === close) {
        if (balance === 0) result.errors++;
        else balance--;
      }
    }
    result.errors += Math.max(0, balance);
    if (balance > 0) {
      result.error_items.push({ line: null, message: `Unbalanced ${open}${close}: missing ${close}` });
    }
  }

  // Keyword occurrences
  const errMatches = code.match(/\berror\b/gi);
  const warnMatches = code.match(/\bwarn(?:ing)?\b/gi);
  if (errMatches) {
    result.errors += errMatches.length;
    // add items by line
    lines.forEach((l, i) => { if (/\berror\b/i.test(l)) result.error_items.push({ line: i + 1, message: "Contains keyword 'error'" }); });
  }
  if (warnMatches) result.warnings += warnMatches.length;
  if (warnMatches) {
    lines.forEach((l, i) => { if (/\bwarn(?:ing)?\b/i.test(l)) result.warning_items.push({ line: i + 1, message: "Contains warning keyword" }); });
  }

  // Long lines and trailing whitespace as warnings
  lines.forEach((l, i) => {
    if (l.length > 120) {
      result.warnings += 1;
      result.warning_items.push({ line: i + 1, message: 'Line exceeds 120 characters' });
    }
    if (/\s+$/.test(l)) {
      result.warnings += 1;
      result.warning_items.push({ line: i + 1, message: 'Trailing whitespace' });
    }
  });

  // Language-specific heuristics
  const langLower = (language || '').toLowerCase();
  const isPython = langLower === 'python';
  const isCStyle = ['javascript','typescript','java','csharp','cpp','c'].includes(langLower);

  if (isPython) {
    // Mixed tabs at start are a warning
    lines.forEach((l, i) => { if (/^\t+/.test(l)) { result.warnings += 1; result.warning_items.push({ line: i + 1, message: 'Tabs used for indentation' }); } });
  }

  if (isCStyle) {
    // Semicolon warnings for common statement patterns
    lines.forEach((l, i) => {
      const t = l.trim();
      if (t === '' || /^\}/.test(t)) return;
      const endsWithSemi = /;\s*$/.test(t);
      const looksLikeStmt = /([=+\-*/].+)|(^return\b)|([)\]]\s*$)/.test(t);
      if (looksLikeStmt && !endsWithSemi) {
        result.warnings += 1;
        result.warning_items.push({ line: i + 1, message: 'Missing semicolon' });
      }
    });
  }

  return result;
}