// Configuration - Auto-detect environment
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? 'http://127.0.0.1:8001' 
  : 'https://codesaviour-api.aish160502.workers.dev';

document.addEventListener('DOMContentLoaded', () => {
  const inputEl = document.getElementById('inputCode');
  const fixedEl = document.getElementById('fixedCode');
  const fixBtn = document.getElementById('fixBtn');
  const deepScanBtn = document.getElementById('deepScanBtn');
  const copyBtn = document.getElementById('copyBtn');
  const clearBtn = document.getElementById('clearBtn');
  const langSel = document.getElementById('languageSelect');
  const banner = document.getElementById('statusBanner');
  
  // Analysis report elements
  const errorCountEl = document.getElementById('errorCount');
  const warningCountEl = document.getElementById('warningCount');
  const timeTakenEl = document.getElementById('timeTaken');
  const fixTimeTakenEl = document.getElementById('fixTimeTaken');
  const analysisDetailsEl = document.getElementById('analysisDetails');
  const copyReportBtn = document.getElementById('copyReportBtn');

  // Keep last analysis for copy
  let lastAnalysisData = [];
  let lastTotals = { errors: 0, warnings: 0 };
  let lastTimeTaken = 0;

  // Start with an empty editor; remove default buggy snippet
  inputEl.value = '';

  langSel.addEventListener('change', () => {
    const v = langSel.value;
    inputEl.placeholder = `Paste your ${v} code here...`;
    // Do not auto-insert sample code on language change
  });

  function showBanner(msg) {
    banner.textContent = msg;
    banner.classList.remove('hidden');
  }
  function clearBanner() {
    banner.textContent = '';
    banner.classList.add('hidden');
  }

  function updateAnalysisReport(analysisData, timeTaken, totals) {
    const errors = analysisData.filter(item => item.type === 'error');
    const warnings = analysisData.filter(item => item.type === 'warning');
    
    // Update cards using API totals when available
    const errorCount = totals && typeof totals.errors === 'number' ? totals.errors : errors.length;
    const warningCount = totals && typeof totals.warnings === 'number' ? totals.warnings : warnings.length;
    errorCountEl.textContent = errorCount;
    warningCountEl.textContent = warningCount;
    timeTakenEl.textContent = `${timeTaken}ms`;
    
    // Persist last analysis results for copy action
    lastAnalysisData = Array.isArray(analysisData) ? analysisData : [];
    lastTotals = { errors: errorCount, warnings: warningCount };
    lastTimeTaken = typeof timeTaken === 'number' ? timeTaken : 0;
    
    // Update detailed analysis
    if (analysisData.length === 0) {
      analysisDetailsEl.innerHTML = '<div class="no-analysis"><p>No issues found in the code</p></div>';
    } else {
      const itemsHtml = analysisData.map(item => `
        <div class="analysis-item">
          <div class="analysis-item-type ${item.type}">${item.type}</div>
          <div class="analysis-item-content">
            <div class="analysis-item-line">Line ${item.line || 'N/A'}</div>
            <div class="analysis-item-message">${item.message || 'No description available'}</div>
          </div>
        </div>
      `).join('');
      analysisDetailsEl.innerHTML = itemsHtml;
    }
  }

  function clearAnalysisReport() {
    errorCountEl.textContent = '0';
    warningCountEl.textContent = '0';
    timeTakenEl.textContent = '0ms';
    analysisDetailsEl.innerHTML = '<div class="no-analysis"><p>Run code analysis to see detailed error and warning reports</p></div>';
    lastAnalysisData = [];
    lastTotals = { errors: 0, warnings: 0 };
    lastTimeTaken = 0;
  }

  async function doDeepScan() {
    clearBanner();
    const language = langSel.value || 'python';
    const code = inputEl.value || '';
    if (!code.trim()) {
      showBanner('Please paste some code to analyze.');
      return;
    }
    if (deepScanBtn) {
      deepScanBtn.disabled = true;
      deepScanBtn.textContent = 'Running deep scan…';
    }
    const startTime = Date.now();
    try {
      const analyzeRes = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language, code })
      });
      let analysisData = [];
      let totals = { errors: 0, warnings: 0 };
      if (analyzeRes.ok) {
        const analyzeData = await analyzeRes.json().catch(() => ({}));
        analysisData = [
          ...((analyzeData.error_items || []).map(it => ({ ...it, type: 'error' }))),
          ...((analyzeData.warning_items || []).map(it => ({ ...it, type: 'warning' })))
        ];
        totals = {
          errors: typeof analyzeData.errors === 'number' ? analyzeData.errors : (analyzeData.error_items?.length || 0),
          warnings: typeof analyzeData.warnings === 'number' ? analyzeData.warnings : (analyzeData.warning_items?.length || 0)
        };
      } else {
        showBanner('Deep scan failed. Check the API server logs for details.');
      }
      updateAnalysisReport(analysisData, Date.now() - startTime, totals);
    } catch (_) {
      showBanner('Network error contacting API. Check your connection and try again.');
    } finally {
      if (deepScanBtn) {
        deepScanBtn.disabled = false;
        deepScanBtn.textContent = 'Run deep scan';
      }
    }
  }
  async function doFix() {
    clearBanner();
    const language = langSel.value || 'python';
    const code = inputEl.value || '';
    if (!code.trim()) {
      showBanner('Please paste some code to fix.');
      return;
    }
    if (fixBtn) {
      fixBtn.disabled = true;
      fixBtn.textContent = 'Fixing…';
    }
    const startTime = Date.now();
    try {
      const res = await fetch(`${API_BASE_URL}/api/fix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language, code })
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        const out = data.fixed_code || data.fixed || '';
        fixedEl.value = out || code;
        if (data.notes) {
          const notes = Array.isArray(data.notes) ? data.notes.join('\n') : String(data.notes);
          console.log('Fix notes:', notes);
        }
        if (fixTimeTakenEl) fixTimeTakenEl.textContent = `${Date.now() - startTime}ms`;
        clearBanner();
      } else {
        if (res.status === 401 || res.status === 403) {
          showBanner('Fireworks unauthorized. Update FIREWORKS_API_KEY in .env and restart the API server.');
        } else if (res.status === 429) {
          showBanner('Rate limited. Please wait a moment and try again.');
        } else {
          showBanner('Fix failed. Check the API server logs for details.');
        }
        fixedEl.value = data.fixed_code || data.fixed || code;
        if (fixTimeTakenEl) fixTimeTakenEl.textContent = `${Date.now() - startTime}ms`;
      }
    } catch (err) {
      showBanner('Network error contacting API. Check your connection and try again.');
    } finally {
      if (fixBtn) {
        fixBtn.disabled = false;
        fixBtn.textContent = 'Fix Code';
      }
    }
  }

  // Copy entire analysis report
  if (copyReportBtn) {
    copyReportBtn.addEventListener('click', async () => {
      try {
        if (!lastAnalysisData || lastAnalysisData.length === 0) {
          showBanner('No analysis report to copy.');
          setTimeout(clearBanner, 2000);
          return;
        }
        const header = [
          `Total Errors: ${lastTotals.errors}`,
          `Total Warnings: ${lastTotals.warnings}`,
          `Time Taken: ${lastTimeTaken}ms`,
          ''
        ].join('\n');
        const details = lastAnalysisData.map((item, idx) => {
          const t = (item.type || '').toUpperCase();
          const line = (item.line == null ? 'N/A' : item.line);
          const msg = item.message || '';
          return `${idx + 1}. [${t}] Line ${line}: ${msg}`;
        }).join('\n');
        const report = `${header}Details:\n${details}\n`;
        await navigator.clipboard.writeText(report);
        showBanner('Analysis report copied to clipboard.');
        setTimeout(clearBanner, 2000);
      } catch (_) {
        showBanner('Unable to copy analysis report.');
        setTimeout(clearBanner, 2500);
      }
    });
  }

  fixBtn.addEventListener('click', doFix);
  if (deepScanBtn) {
    deepScanBtn.addEventListener('click', doDeepScan);
  }
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(fixedEl.value || '');
      showBanner('Copied fixed code to clipboard.');
      setTimeout(clearBanner, 2000);
    } catch (_) {
      showBanner('Unable to copy to clipboard.');
    }
  });
  clearBtn.addEventListener('click', () => {
    inputEl.value = '';
    fixedEl.value = '';
    clearBanner();
    clearAnalysisReport();
  });
});