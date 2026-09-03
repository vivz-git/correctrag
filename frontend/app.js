/**
 * CorrectRAG Browser UI — Client Logic
 *
 * All API interaction, XSS protection, error handling, and event binding
 * preserved from original implementation. Updated only for:
 * - New HTML structure class names
 * - aria-busy state management
 * - Loading skeleton injection
 * - CSS-based error icon (SVG in HTML)
 * - Focus management after results render
 */

// ── API Configuration ──────────────────────────────────────────────────────

const API_BASE_URL =
  window.__CORRECTRAG_API_URL__ ||
  new URLSearchParams(window.location.search).get('api_url') ||
  (window.location.hostname === 'localhost' && window.location.port === '8000'
    ? window.location.origin
    : (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : 'https://13.235.51.127.sslip.io'));

// ── DOM Elements ───────────────────────────────────────────────────────────

const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');
const apiStatusPill = document.getElementById('apiStatusPill');
const apiStatusText = document.getElementById('apiStatusText');
const errorBanner = document.getElementById('errorBanner');
const errorTitle = document.getElementById('errorTitle');
const errorMessage = document.getElementById('errorMessage');
const resultsContainer = document.getElementById('resultsContainer');

// Result Elements
const actionBadge = document.getElementById('actionBadge');
const sourceSummaryBadge = document.getElementById('sourceSummaryBadge');
const answerText = document.getElementById('answerText');
const internalSourcesList = document.getElementById('internalSourcesList');
const webSourcesList = document.getElementById('webSourcesList');
const internalCountBadge = document.getElementById('internalCountBadge');
const webCountBadge = document.getElementById('webCountBadge');

// Execution Trace Elements
const traceAction = document.getElementById('traceAction');
const traceMaxScore = document.getElementById('traceMaxScore');
const traceRetrievedCount = document.getElementById('traceRetrievedCount');
const traceWebUsed = document.getElementById('traceWebUsed');
const traceRewrittenQuery = document.getElementById('traceRewrittenQuery');
const traceInternalStrips = document.getElementById('traceInternalStrips');
const traceExternalStrips = document.getElementById('traceExternalStrips');
const traceContextSource = document.getElementById('traceContextSource');


// ── Health Check ───────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (res.ok) {
      apiStatusPill.className = 'status-pill status-connected';
      apiStatusText.textContent = 'API Connected';
    } else {
      setApiOffline();
    }
  } catch {
    setApiOffline();
  }
}

function setApiOffline() {
  apiStatusPill.className = 'status-pill status-error';
  apiStatusText.textContent = 'API Offline';
}


// ── XSS Protection ────────────────────────────────────────────────────────

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}


// ── Answer Formatting ──────────────────────────────────────────────────────

function formatAnswerContent(text) {
  if (!text) return '';
  const escaped = escapeHtml(text);
  // Convert **bold** to <strong>bold</strong>
  return escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}


// ── Error Display ──────────────────────────────────────────────────────────

function showError(title, message) {
  errorTitle.textContent = title;
  errorMessage.textContent = message;
  errorBanner.classList.remove('hidden');
}

function hideError() {
  errorBanner.classList.add('hidden');
}


// ── Loading State ──────────────────────────────────────────────────────────

function setLoading(loading) {
  submitBtn.disabled = loading;
  resultsContainer.setAttribute('aria-busy', loading ? 'true' : 'false');

  if (loading) {
    btnSpinner.classList.remove('hidden');
    btnText.textContent = 'Processing…';
    hideError();

    // Show loading skeleton in answer area
    answerText.innerHTML =
      '<div class="loading-skeleton" aria-label="Loading results">' +
        '<div class="skeleton-line"></div>' +
        '<div class="skeleton-line"></div>' +
        '<div class="skeleton-line"></div>' +
      '</div>';

    // Reveal results container during loading to show skeleton
    resultsContainer.classList.remove('hidden');
  } else {
    btnSpinner.classList.add('hidden');
    btnText.textContent = 'Run Query';
  }
}


// ── Source Rendering ───────────────────────────────────────────────────────

function renderInternalSources(chunks, strips) {
  internalSourcesList.innerHTML = '';
  const items = chunks && chunks.length > 0 ? chunks : [];
  internalCountBadge.textContent = items.length;

  if (items.length === 0) {
    internalSourcesList.innerHTML =
      '<p class="empty-sources">No internal document passages used.</p>';
    return;
  }

  items.forEach(function (chunk) {
    const item = document.createElement('div');
    item.className = 'source-item';

    const pageStr = chunk.page_number
      ? 'p.\u00A0' + chunk.page_number
      : 'doc';
    const scoreStr =
      typeof chunk.score === 'number'
        ? (chunk.score > 0 ? '+' : '') + chunk.score.toFixed(4)
        : '';

    const rawSource = chunk.source || "Document";
    const sourceName = rawSource.split('/').pop().split('\\').pop();

    item.innerHTML =
      '<div class="source-header">' +
        '<span class="source-title">' + escapeHtml(sourceName) + ' \u00B7 ' + pageStr + '</span>' +
        (scoreStr
          ? '<div class="source-meta"><span class="score-tag">' + scoreStr + '</span></div>'
          : '') +
      '</div>' +
      (chunk.text_snippet
        ? '<p class="source-snippet">' + escapeHtml(chunk.text_snippet) + '</p>'
        : '');

    internalSourcesList.appendChild(item);
  });
}

function renderWebSources(webResults) {
  webSourcesList.innerHTML = '';
  const items = webResults && webResults.length > 0 ? webResults : [];
  webCountBadge.textContent = items.length;

  if (items.length === 0) {
    webSourcesList.innerHTML =
      '<p class="empty-sources">No external web sources queried.</p>';
    return;
  }

  items.forEach(function (result) {
    const item = document.createElement('div');
    item.className = 'source-item';

    const scoreStr =
      typeof result.score === 'number'
        ? result.score.toFixed(2)
        : '';

    item.innerHTML =
      '<div class="source-header">' +
        '<span class="source-title">' + escapeHtml(result.title || 'Web Result') + '</span>' +
        (scoreStr
          ? '<div class="source-meta"><span class="score-tag">' + scoreStr + '</span></div>'
          : '') +
      '</div>' +
      '<a href="' + encodeURI(result.url) + '" target="_blank" rel="noopener noreferrer" class="source-link">' +
        escapeHtml(result.url) +
      '</a>' +
      (result.snippet
        ? '<p class="source-snippet">' + escapeHtml(result.snippet) + '</p>'
        : '');

    webSourcesList.appendChild(item);
  });
}


// ── Execution Trace ────────────────────────────────────────────────────────

function renderTrace(trace) {
  if (!trace) return;

  traceAction.textContent = trace.action || '–';
  traceMaxScore.textContent =
    typeof trace.max_relevance_score === 'number'
      ? (trace.max_relevance_score > 0 ? '+' : '') + trace.max_relevance_score.toFixed(4)
      : 'None (0 internal chunks)';

  traceRetrievedCount.textContent = trace.retrieved_count + ' chunk(s)';
  traceWebUsed.textContent = trace.web_search_used ? 'Yes' : 'No';
  traceRewrittenQuery.textContent = trace.rewritten_query || 'None (CORRECT branch)';
  traceInternalStrips.textContent = trace.internal_strip_count + ' strip(s)';
  traceExternalStrips.textContent = trace.external_strip_count + ' strip(s)';
  traceContextSource.textContent = trace.final_context_source || '–';
}


// ── Query Submission ───────────────────────────────────────────────────────

async function handleQuerySubmit(e) {
  if (e) e.preventDefault();

  const question = queryInput.value.trim();
  if (!question) {
    showError('Validation Error', 'Please enter a non-empty question.');
    queryInput.focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      // Hide skeleton on error
      resultsContainer.classList.add('hidden');

      if (response.status === 422) {
        const errData = await response.json().catch(function () { return {}; });
        const detailMsg = Array.isArray(errData.detail)
          ? errData.detail.map(function (d) { return d.msg; }).join(', ')
          : 'Invalid request format.';
        showError('Validation Error (HTTP 422)', detailMsg);
      } else if (response.status === 500) {
        const errData = await response.json().catch(function () { return {}; });
        showError('Server Error (HTTP 500)', errData.detail || 'An internal server error occurred.');
      } else {
        showError('HTTP Error ' + response.status, 'Request failed with status ' + response.statusText);
      }
      return;
    }

    const data = await response.json();

    // 1. Update Action Badge
    const action = data.action || 'CORRECT';
    actionBadge.textContent = action;
    actionBadge.className = 'action-badge';
    
    if (data.execution_trace && data.execution_trace.judge_reason) {
      actionBadge.title = data.execution_trace.judge_reason;
    } else {
      actionBadge.removeAttribute('title');
    }

    if (action === 'CORRECT') {
      actionBadge.classList.add('action-correct');
    } else if (action === 'AMBIGUOUS') {
      actionBadge.classList.add('action-ambiguous');
    } else {
      actionBadge.classList.add('action-incorrect');
    }

    // 1.5 Update Source Summary Badge
    let sourceSummary = "No Source";
    const hasInternal = data.retrieved_chunks && data.retrieved_chunks.length > 0;
    const hasWeb = data.web_results && data.web_results.length > 0;

    if (action === 'CORRECT' || (hasInternal && !hasWeb)) {
       const chunk = hasInternal ? data.retrieved_chunks[0] : null;
       if (chunk) {
           const rawSource = chunk.source || "CRAG.pdf";
           const sourceName = rawSource.split('/').pop().split('\\').pop();
           const pageNum = chunk.page_number ? ` · Page ${chunk.page_number}` : "";
           sourceSummary = `${sourceName}${pageNum}`;
       } else {
           sourceSummary = "CRAG.pdf";
       }
    } else if (action === 'INCORRECT' || (!hasInternal && hasWeb)) {
       sourceSummary = "Tavily Web";
    } else if (action === 'AMBIGUOUS' || (hasInternal && hasWeb)) {
       sourceSummary = "CRAG.pdf + Tavily Web";
    }
    sourceSummaryBadge.textContent = sourceSummary;

    // 2. Render Answer
    answerText.innerHTML = formatAnswerContent(data.answer);

    // 3. Render Sources
    renderInternalSources(data.retrieved_chunks, data.refined_strips);
    renderWebSources(data.web_results);

    // 4. Render Trace
    renderTrace(data.execution_trace);

    // 5. Reveal Results
    resultsContainer.classList.remove('hidden');

    // 6. Scroll answer into view and shift focus for accessibility
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    resultsContainer.classList.add('hidden');
    showError(
      'Network Error',
      'Unable to connect to the CorrectRAG API at ' +
        API_BASE_URL +
        '. Make sure the FastAPI server is running (uvicorn backend.app.main:app --reload --port 8000).'
    );
    setApiOffline();
  } finally {
    setLoading(false);
  }
}


// ── Event Listeners ────────────────────────────────────────────────────────

queryForm.addEventListener('submit', handleQuerySubmit);

// Enter key submits; Shift+Enter creates newline
queryInput.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleQuerySubmit();
  }
});

// Sample prompt chips
document.querySelectorAll('.sample-chip').forEach(function (chip) {
  chip.addEventListener('click', function () {
    var query = chip.getAttribute('data-query');
    if (query) {
      queryInput.value = query;
      queryInput.focus();
      handleQuerySubmit();
    }
  });
});

// Check health on page load
document.addEventListener('DOMContentLoaded', function () {
  checkHealth();
});
