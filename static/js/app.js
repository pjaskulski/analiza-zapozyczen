// Global variables to store current state
let currentTab = 'corpus';
let currentAnalysisData = null; // Stores currently loaded analysis with all detections
let currentCorpusDocs = []; // Stores currently loaded corpus documents

// Toast notification helper
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `flex items-center space-x-3 px-4 py-3 rounded-xl shadow-2xl pointer-events-auto transform translate-y-2 transition-all duration-300`;
    toast.style.backgroundColor = '#0f172a';
    toast.style.borderWidth = '1px';
    toast.style.borderColor = type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)';
    toast.style.color = type === 'success' ? '#a7f3d0' : '#fecdd3';
    
    const icon = type === 'success' ? 'fa-circle-check' : 'fa-circle-exclamation';
    toast.innerHTML = `
        <i class="fa-solid ${icon} text-lg"></i>
        <span class="text-sm font-medium">${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.className = toast.className.replace('translate-y-2', 'translate-y-0');
    }, 10);
    
    // Remove after 3.5 seconds
    setTimeout(() => {
        toast.className = toast.className + ' opacity-0 -translate-y-2';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3500);
}

// Tab Switching
function switchTab(tabId) {
    currentTab = tabId;
    
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    
    // Remove active styles from navigation buttons
    document.querySelectorAll('header nav button').forEach(btn => {
        btn.classList.remove('bg-violet-600/10', 'text-violet-600', 'dark:text-violet-400', 'border', 'border-violet-500/20');
        btn.classList.add('text-slate-400');
    });
    
    // Show selected tab
    const selectedTabEl = document.getElementById(`tab-${tabId}`);
    if (selectedTabEl) {
        selectedTabEl.classList.remove('hidden');
    }
    
    // Style active tab button
    const activeBtn = document.getElementById(`btn-tab-${tabId}`);
    if (activeBtn) {
        activeBtn.classList.remove('text-slate-400');
        activeBtn.classList.add('bg-violet-600/10', 'text-violet-600', 'dark:text-violet-400', 'border', 'border-violet-500/20');
    }
    
    // Trigger tab-specific loads
    if (tabId === 'corpus') {
        loadCorpusDocuments();
    } else if (tabId === 'analyze') {
        loadAnalysisHistory();
    } else if (tabId === 'settings') {
        loadSettings();
    }
}

function setProviderGroupState(groupId, enabled) {
    const group = document.getElementById(groupId);
    if (!group) return;

    group.classList.toggle('provider-disabled', !enabled);
    group.querySelectorAll('input, select, textarea, button').forEach(control => {
        control.disabled = !enabled;
    });
}

function updateLLMProviderFields() {
    const providerSelect = document.getElementById('setting-llm-provider');
    if (!providerSelect) return;

    const provider = providerSelect.value;
    const isGemini = provider === 'gemini';
    const isOllama = provider === 'ollama';

    setProviderGroupState('gemini-settings-group', isGemini);
    setProviderGroupState('gemini-thinking-group', isGemini);
    setProviderGroupState('ollama-settings-group', isOllama);
}

// File name helpers
function updateFileName(input) {
    const placeholder = document.getElementById('file-placeholder');
    if (input.files && input.files.length > 0) {
        placeholder.innerText = `Wybrany plik: ${input.files[0].name}`;
        placeholder.classList.remove('text-slate-400');
        placeholder.classList.add('text-violet-600', 'dark:text-violet-400', 'font-semibold');
    } else {
        placeholder.innerText = "Przeciągnij plik tutaj lub kliknij, aby wybrać";
        placeholder.classList.remove('text-violet-600', 'dark:text-violet-400', 'font-semibold');
        placeholder.classList.add('text-slate-400');
    }
}

function updateAnalysisFileName(input) {
    const placeholder = document.getElementById('analysis-file-placeholder');
    if (input.files && input.files.length > 0) {
        placeholder.innerText = `Wybrany plik: ${input.files[0].name}`;
        placeholder.classList.remove('text-slate-400');
        placeholder.classList.add('text-violet-600', 'dark:text-violet-400', 'font-semibold');
    } else {
        placeholder.innerText = "Kliknij, aby wybrać plik badany z dysku";
        placeholder.classList.remove('text-violet-600', 'dark:text-violet-400', 'font-semibold');
        placeholder.classList.add('text-slate-400');
    }
}

function formatAnalysisParams(params = {}) {
    const parts = [];
    const provider = params.llm_provider;
    if (params.llm_provider) {
        parts.push(`provider: ${params.llm_provider}`);
    }
    if (params.llm_model) {
        parts.push(`model LLM: ${params.llm_model}`);
    } else if (provider === 'ollama' && params.ollama_model) {
        parts.push(`model LLM: ${params.ollama_model}`);
    } else if (provider === 'gemini' && params.gemini_model) {
        parts.push(`model LLM: ${params.gemini_model}`);
    } else if (params.ollama_model) {
        parts.push(`model LLM: ${params.ollama_model}`);
    } else if (params.gemini_model) {
        parts.push(`model LLM: ${params.gemini_model}`);
    }
    if (params.ollama_base_url) {
        parts.push(`Ollama URL: ${params.ollama_base_url}`);
    }
    if (provider === 'ollama' && params.ollama_think !== undefined && params.ollama_think !== null) {
        parts.push(`Ollama thinking: ${params.ollama_think ? 'włączone' : 'wyłączone'}`);
    }
    if (params.embedding_model) {
        parts.push(`embeddingi: ${params.embedding_model}`);
    }
    if (params.gemini_temp !== undefined) {
        parts.push(`temp.: ${Number(params.gemini_temp).toFixed(2)}`);
    }
    if (provider !== 'ollama' && params.gemini_thinking_level) {
        parts.push(`myślenie: ${params.gemini_thinking_level}`);
    }
    if (params.top_k !== undefined) {
        parts.push(`top_k: ${params.top_k}`);
    }
    if (params.similarity_threshold !== undefined) {
        parts.push(`próg: ${Number(params.similarity_threshold).toFixed(2)}`);
    }
    if (params.analysis_segment_size !== undefined && params.analysis_segment_overlap !== undefined) {
        parts.push(`segment: ${params.analysis_segment_size}/${params.analysis_segment_overlap} słów`);
    }
    if (params.corpus_chunk_size !== undefined) {
        parts.push(`chunk korpusu: ${params.corpus_chunk_size} słów`);
    }
    if (params.segments_count !== undefined) {
        parts.push(`segmentów: ${params.segments_count}`);
    }
    return parts.length > 0 ? `Parametry analizy: ${parts.join(' · ')}` : 'Parametry analizy: brak danych';
}

// ================= TAB 1: KORPUS REFERENCYJNY =================

async function loadCorpusDocuments() {
    const tbody = document.getElementById('corpus-list-body');
    const docCountEl = document.getElementById('doc-count');
    
    try {
        const response = await fetch('/api/corpus');
        const docs = await response.json();
        
        currentCorpusDocs = docs;
        docCountEl.innerText = docs.length;
        
        if (docs.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="py-8 text-center text-slate-500">
                        <i class="fa-solid fa-folder-open text-2xl mb-2 block text-slate-700"></i>
                        Korpus jest pusty. Dodaj pierwszy dokument po lewej stronie.
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = docs.map(doc => `
            <tr class="hover:bg-slate-900/30 transition-colors border-b border-slate-900/50">
                <td class="py-4 px-4 font-semibold text-slate-100">${doc.title}</td>
                <td class="py-4 px-4 text-slate-400">${doc.author}</td>
                <td class="py-4 px-4 font-mono text-sm text-slate-600 dark:text-slate-400">${doc.chunk_count}</td>
                <td class="py-4 px-4 text-xs text-slate-500">${doc.added_at}</td>
                <td class="py-4 px-4 text-right flex items-center justify-end space-x-1.5">
                    <button onclick="openEditCorpusModal(${doc.id})" class="h-8 w-8 text-slate-500 hover:text-violet-400 hover:bg-violet-500/10 border border-transparent hover:border-violet-500/25 rounded-lg transition-all flex items-center justify-center inline-flex cursor-pointer" title="Edytuj metadane">
                        <i class="fa-solid fa-pen text-sm"></i>
                    </button>
                    <button onclick="deleteCorpusDocument(${doc.id})" class="h-8 w-8 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/25 rounded-lg transition-all flex items-center justify-center inline-flex cursor-pointer" title="Usuń z korpusu">
                        <i class="fa-solid fa-trash-can text-sm"></i>
                    </button>
                </td>
            </tr>
        `).join('');
        
    } catch (err) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="py-8 text-center text-red-400">
                    <i class="fa-solid fa-triangle-exclamation text-2xl mb-2 block"></i>
                    Błąd pobierania danych: ${err.message}
                </td>
            </tr>
        `;
    }
}

async function handleCorpusUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file-input');
    const titleInput = document.getElementById('corpus-title');
    const authorInput = document.getElementById('corpus-author');
    const btnSubmit = document.getElementById('btn-upload-submit');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showToast('Wybierz plik przed przesłaniem!', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('title', titleInput.value.trim());
    formData.append('author', authorInput.value.trim());
    
    // Set loading state
    btnSubmit.disabled = true;
    const origContent = btnSubmit.innerHTML;
    btnSubmit.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i> Przetwarzanie i wektoryzacja RAG...`;
    
    try {
        const response = await fetch('/api/corpus', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast('Dokument został pomyślnie dodany do korpusu i zindeksowany w ChromaDB!');
            titleInput.value = '';
            authorInput.value = '';
            fileInput.value = '';
            updateFileName(fileInput);
            loadCorpusDocuments();
        } else {
            showToast(data.error || 'Wystąpił błąd podczas dodawania dokumentu.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    } finally {
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = origContent;
    }
}

async function deleteCorpusDocument(id) {
    if (!confirm('Czy na pewno chcesz usunąć ten dokument z korpusu? Spowoduje to nieodwracalne usunięcie powiązanych wektorów w ChromaDB.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/corpus/${id}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (response.ok) {
            showToast('Dokument usunięty pomyślnie.');
            loadCorpusDocuments();
        } else {
            showToast(data.error || 'Błąd podczas usuwania dokumentu.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    }
}


// ================= TAB 2: ANALIZY TEKSTÓW =================

async function loadAnalysisHistory() {
    const listContainer = document.getElementById('analysis-history-list');
    
    try {
        const response = await fetch('/api/analyze');
        const history = await response.json();
        
        if (history.length === 0) {
            listContainer.innerHTML = `
                <div class="text-center py-8 text-slate-500">
                    <i class="fa-solid fa-chart-line text-2xl text-slate-700 mb-2 block"></i>
                    Brak wcześniejszych analiz.
                </div>
            `;
            return;
        }
        
        listContainer.innerHTML = history.map(item => `
            <div class="group bg-slate-900/40 hover:bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between transition-all duration-200" id="analysis-row-${item.id}">
                <div class="cursor-pointer flex-grow" onclick="loadAnalysisResults(${item.id})">
                    <h4 class="font-semibold text-slate-100 text-sm group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors" id="analysis-title-text-${item.id}">${item.title}</h4>
                    <span class="text-[10px] text-slate-500 block mt-1"><i class="fa-regular fa-clock mr-1"></i> ${item.analyzed_at}</span>
                </div>
                <div class="flex items-center space-x-1">
                    <button onclick="openEditAnalysisModal(${item.id}, '${item.title.replace(/'/g, "\\'")}')" class="h-8 w-8 text-slate-600 hover:text-violet-400 rounded-lg hover:bg-violet-500/10 flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 cursor-pointer" title="Edytuj tytuł">
                        <i class="fa-solid fa-pen text-xs"></i>
                    </button>
                    <button onclick="deleteAnalysis(${item.id})" class="h-8 w-8 text-slate-600 hover:text-rose-400 rounded-lg hover:bg-rose-500/10 flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 cursor-pointer" title="Usuń analizę">
                        <i class="fa-solid fa-trash-can text-xs"></i>
                    </button>
                </div>
            </div>
        `).join('');
        
    } catch (err) {
        listContainer.innerHTML = `<p class="text-xs text-red-400 text-center py-4">Błąd historii: ${err.message}</p>`;
    }
}

async function handleRunAnalysis(event) {
    event.preventDefault();
    
    const titleInput = document.getElementById('analysis-title');
    const textInput = document.getElementById('analysis-text');
    const fileInput = document.getElementById('analysis-file-input');
    const btnSubmit = document.getElementById('btn-analyze-submit');
    
    const text = textInput.value.trim();
    const hasFile = fileInput.files && fileInput.files.length > 0;
    
    if (!text && !hasFile) {
        showToast('Wprowadź tekst łaciński lub wgraj plik do analizy!', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('title', titleInput.value.trim());
    if (text) {
        formData.append('text', text);
    }
    if (hasFile) {
        formData.append('file', fileInput.files[0]);
    }
    
    btnSubmit.disabled = true;
    const origContent = btnSubmit.innerHTML;
    btnSubmit.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i> Skanowanie okna przesuwnego, wektoryzacja i weryfikacja LLM...`;
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast(`Analiza zakończona! Wykryto ${data.detections_count} powiązań.`);
            titleInput.value = '';
            textInput.value = '';
            fileInput.value = '';
            updateAnalysisFileName(fileInput);
            
            // Open the analysis view and load the result
            loadAnalysisResults(data.analysis_id, false);
        } else {
            showToast(data.error || 'Błąd analizy.', 'error');
            if (data.analysis_id) {
                loadAnalysisResults(data.analysis_id, true);
            }
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    } finally {
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = origContent;
    }
}

async function deleteAnalysis(id) {
    if (!confirm('Czy chcesz usunąć tę analizę z historii?')) return;
    
    try {
        const response = await fetch(`/api/analyze/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast('Analiza usunięta.');
            loadAnalysisHistory();
        } else {
            showToast('Błąd usuwania analizy.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    }
}

async function loadAnalysisResults(analysisId, forceExpandLogs = false) {
    try {
        const response = await fetch(`/api/analyze/${analysisId}`);
        const data = await response.json();
        
        if (!response.ok) {
            showToast(data.error || 'Nie udało się wczytać analizy.', 'error');
            return;
        }
        
        currentAnalysisData = data;
        
        // Show results screen
        document.getElementById('analysis-setup-view').classList.add('hidden');
        document.getElementById('analysis-results-view').classList.remove('hidden');
        
        // Update Title and Date
        document.getElementById('result-analysis-title').innerText = data.title;
        document.getElementById('result-analysis-date').innerText = `Wykonano: ${data.analyzed_at}`;
        
        // Update Tokens and Cost
        document.getElementById('result-prompt-tokens').innerText = data.prompt_tokens || 0;
        document.getElementById('result-candidates-tokens').innerText = data.candidates_tokens || 0;
        document.getElementById('result-thoughts-tokens').innerText = data.thoughts_tokens || 0;
        document.getElementById('result-llm-calls').innerText = data.llm_calls || 0;
        document.getElementById('result-total-cost').innerText = '$' + (data.estimated_cost || 0.0).toFixed(6);
        document.getElementById('result-analysis-params').innerText = formatAnalysisParams(data.analysis_params || {});
        
        const summaryElement = document.getElementById('result-analysis-summary');
        if (summaryElement) {
            summaryElement.innerHTML = `
                <i class="fa-solid fa-link text-violet-500 mr-1"></i> 
                Razem zapożyczeń: ${data.detections.length}
            `;
        }
        
        // Reset details panel to empty state
        resetDetailsPanel();
        
        // Render statistics
        const stats = { Full: 0, Partial: 0, Paraphrase: 0, Allusion: 0 };
        data.detections.forEach(d => {
            if (stats[d.match_type] !== undefined) {
                stats[d.match_type]++;
            }
        });
        
        document.getElementById('stats-count-full').innerText = stats.Full;
        document.getElementById('stats-count-partial').innerText = stats.Partial;
        document.getElementById('stats-count-para').innerText = stats.Paraphrase;
        document.getElementById('stats-count-allusion').innerText = stats.Allusion;
        
        // Highlight logic
        renderHighlightedSourceText(data.content, data.detections);
        
        // --- Render Logs ---
        const logsContainer = document.getElementById('log-console-container');
        const statusBadge = document.getElementById('log-status-badge');
        
        if (logsContainer && data.logs) {
            // Escape HTML
            const escapedLogs = escapeHtml(data.logs);
            // Color code each line
            const formattedLines = escapedLogs.split('\n').map(line => {
                if (line.includes('[FATAL]') || line.includes('[ERROR]')) {
                    return `<span class="text-red-600 dark:text-red-400 font-semibold">${line}</span>`;
                } else if (line.includes('[WARNING]')) {
                    return `<span class="text-amber-600 dark:text-amber-400 font-semibold">${line}</span>`;
                } else if (line.includes('[SUCCESS]')) {
                    return `<span class="text-emerald-600 dark:text-emerald-400 font-semibold">${line}</span>`;
                } else if (line.includes('[LLM]') || line.includes('[Gemini LLM]')) {
                    return `<span class="text-violet-600 dark:text-violet-400 font-medium">${line}</span>`;
                } else if (line.includes('[ChromaDB]')) {
                    return `<span class="text-sky-600 dark:text-sky-400 font-medium">${line}</span>`;
                } else if (line.includes('[CONFIG]') || line.includes('[SEGMENTACJA]')) {
                    return `<span class="text-indigo-600 dark:text-indigo-400 font-medium">${line}</span>`;
                } else if (line.includes('[INFO]')) {
                    return `<span class="text-slate-600 dark:text-slate-400 font-light">${line}</span>`;
                }
                return `<span>${line}</span>`;
            });
            logsContainer.innerHTML = formattedLines.join('\n');
            
            // Set status badge
            if (statusBadge) {
                statusBadge.classList.remove('hidden');
                if (data.logs.includes('[FATAL]') || data.logs.includes('[ERROR]') || forceExpandLogs) {
                    statusBadge.innerText = 'BŁĄD';
                    statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-red-500/20 text-red-300 border border-red-500/30';
                } else if (data.detections.length === 0) {
                    statusBadge.innerText = 'BRAK POWIĄZAŃ';
                    statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-yellow-500/20 text-yellow-300 border border-yellow-500/30';
                } else {
                    statusBadge.innerText = 'SUKCES';
                    statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
                }
            }
            
            // Scroll to bottom
            setTimeout(() => {
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }, 50);
        }
        
        // Auto expand or collapse logs
        if (forceExpandLogs || data.detections.length === 0) {
            expandLogConsole();
        } else {
            collapseLogConsole();
        }
        
    } catch (err) {
        showToast(`Błąd wczytywania: ${err.message}`, 'error');
    }
}

function backToAnalysisSetup() {
    document.getElementById('analysis-results-view').classList.add('hidden');
    document.getElementById('analysis-setup-view').classList.remove('hidden');
    loadAnalysisHistory();
}

function resetDetailsPanel() {
    document.getElementById('details-content-state').classList.add('hidden');
    document.getElementById('details-empty-state').classList.remove('hidden');
}

// Algorithm to render highlights by avoiding overlapping indices
function renderHighlightedSourceText(content, detections) {
    const container = document.getElementById('result-source-text-area');
    
    if (detections.length === 0) {
        container.innerText = content;
        return;
    }
    
    // Sort detections: primary by start_char_idx ascending, secondary by similarity score descending
    const sortedDetections = [...detections].sort((a, b) => {
        if (a.start_char_idx !== b.start_char_idx) {
            return a.start_char_idx - b.start_char_idx;
        }
        return b.similarity_score - a.similarity_score;
    });
    
    // Filter out overlapping detections (keeping the first one, which has the highest similarity due to sorting)
    const filteredDetections = [];
    let lastEnd = 0;
    
    for (const det of sortedDetections) {
        if (det.start_char_idx >= lastEnd) {
            filteredDetections.push(det);
            lastEnd = det.end_char_idx;
        }
    }
    
    // Build HTML block step-by-step
    let htmlContent = '';
    let currentIdx = 0;
    
    for (const det of filteredDetections) {
        // Plain text before highlight
        if (det.start_char_idx > currentIdx) {
            htmlContent += escapeHtml(content.substring(currentIdx, det.start_char_idx));
        }
        
        // Highlight slice
        const textSlice = content.substring(det.start_char_idx, det.end_char_idx);
        const matchClass = `highlight-${det.match_type.toLowerCase()}`;
        
        htmlContent += `<span class="${matchClass}" onclick="showDetectionDetails(${det.id})">${escapeHtml(textSlice)}</span>`;
        
        currentIdx = det.end_char_idx;
    }
    
    // Plain text after last highlight
    if (currentIdx < content.length) {
        htmlContent += escapeHtml(content.substring(currentIdx));
    }
    
    container.innerHTML = htmlContent;
}

function renderCorpusSourceFragments(det) {
    const container = document.getElementById('detail-corpus-snippet');
    const fragments = Array.isArray(det.corpus_source_fragments) && det.corpus_source_fragments.length > 0
        ? det.corpus_source_fragments
        : [{ chunk_index: null, text: det.corpus_snippet_text, similarity: det.similarity_score }];

    container.innerHTML = fragments.map((fragment, index) => {
        const chunkLabel = fragment.chunk_index === null || fragment.chunk_index === undefined
            ? `fragment ${index + 1}`
            : `chunk ${fragment.chunk_index}`;
        const similarity = typeof fragment.similarity === 'number'
            ? ` · ${(fragment.similarity * 100).toFixed(1)}%`
            : '';
        return `
            <div class="bg-violet-500/5 dark:bg-violet-950/10 p-3 rounded-xl border border-violet-500/20">
                <div class="flex items-center justify-between gap-2 mb-2">
                    <span class="text-[10px] font-bold uppercase tracking-wider text-violet-700 dark:text-violet-300">${escapeHtml(chunkLabel)}</span>
                    <span class="text-[10px] text-slate-500">${escapeHtml(similarity)}</span>
                </div>
                <div class="text-xs font-mono leading-relaxed text-slate-300 whitespace-pre-wrap">${escapeHtml(fragment.text || '')}</div>
            </div>
        `;
    }).join('');
}

// Escape HTML utility to prevent HTML breaking or XSS
function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function showDetectionDetails(detectionId) {
    if (!currentAnalysisData) return;
    
    const det = currentAnalysisData.detections.find(d => d.id === detectionId);
    if (!det) return;
    
    // Show details content state
    document.getElementById('details-empty-state').classList.add('hidden');
    document.getElementById('details-content-state').classList.remove('hidden');
    
    // Update badge details
    const typeBadge = document.getElementById('detail-type-badge');
    typeBadge.innerText = det.match_type;
    
    // Styling classes for Type badge
    typeBadge.className = "px-3 py-1 rounded-lg text-xs font-bold uppercase tracking-wider ";
    if (det.match_type === 'Full') {
        typeBadge.classList.add('bg-red-500/20', 'text-red-800', 'dark:text-red-300', 'border', 'border-red-500/30');
    } else if (det.match_type === 'Partial') {
        typeBadge.classList.add('bg-amber-500/20', 'text-amber-800', 'dark:text-amber-300', 'border', 'border-amber-500/30');
    } else if (det.match_type === 'Paraphrase') {
        typeBadge.classList.add('bg-yellow-500/20', 'text-yellow-800', 'dark:text-yellow-300', 'border', 'border-yellow-500/30');
    } else {
        typeBadge.classList.add('bg-violet-500/20', 'text-violet-800', 'dark:text-violet-300', 'border', 'border-violet-500/30');
    }
    
    // Similarity badge
    const simPercent = (det.similarity_score * 100).toFixed(1);
    document.getElementById('detail-similarity-badge').innerText = `Podobieństwo: ${simPercent}%`;
    
    // Source metadata
    document.getElementById('detail-corpus-title').innerText = det.corpus_document_title;
    document.getElementById('detail-corpus-author').innerText = `Autor: ${det.corpus_document_author}`;
    
    // Snippets
    document.getElementById('detail-source-snippet').innerText = det.source_snippet;
    renderCorpusSourceFragments(det);
    
    // Explanation
    document.getElementById('detail-explanation').innerText = det.explanation || "Brak szczegółowego uzasadnienia.";
}


// ================= TAB 3: PARAMETRY =================

async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        const settings = await response.json();
        
        // Populate RAG form
        if (settings.similarity_threshold !== undefined) {
            document.getElementById('setting-threshold').value = settings.similarity_threshold;
            document.getElementById('threshold-val').innerText = parseFloat(settings.similarity_threshold).toFixed(2);
        }
        if (settings.top_k !== undefined) {
            document.getElementById('setting-top-k').value = settings.top_k;
        }
        if (settings.chunk_size !== undefined) {
            document.getElementById('setting-chunk-size').value = settings.chunk_size;
        }
        if (settings.chunk_overlap !== undefined) {
            document.getElementById('setting-chunk-overlap').value = settings.chunk_overlap;
        }
        
        // Populate LLM form
        if (settings.llm_provider !== undefined) {
            document.getElementById('setting-llm-provider').value = settings.llm_provider;
        }
        if (settings.gemini_model !== undefined) {
            document.getElementById('setting-model-name').value = settings.gemini_model;
        }
        if (settings.gemini_temp !== undefined) {
            document.getElementById('setting-temp').value = settings.gemini_temp;
            document.getElementById('temp-val').innerText = parseFloat(settings.gemini_temp).toFixed(2);
        }
        if (settings.gemini_thinking_level !== undefined) {
            document.getElementById('setting-thinking-level').value = settings.gemini_thinking_level;
        }
        if (settings.ollama_base_url !== undefined) {
            document.getElementById('setting-ollama-url').value = settings.ollama_base_url;
        }
        if (settings.ollama_model !== undefined) {
            document.getElementById('setting-ollama-model').value = settings.ollama_model;
        }
        if (settings.ollama_think !== undefined) {
            document.getElementById('setting-ollama-think').checked = ['1', 'true', 'yes', 'on', 'tak'].includes(String(settings.ollama_think).toLowerCase());
        }
        updateLLMProviderFields();
        
    } catch (err) {
        showToast('Nie udało się załadować konfiguracji.', 'error');
    }
}

async function handleSaveRAGSettings(event) {
    event.preventDefault();
    
    const threshold = document.getElementById('setting-threshold').value;
    const top_k = document.getElementById('setting-top-k').value;
    const chunk_size = document.getElementById('setting-chunk-size').value;
    const chunk_overlap = document.getElementById('setting-chunk-overlap').value;
    
    const payload = {
        similarity_threshold: threshold,
        top_k: top_k,
        chunk_size: chunk_size,
        chunk_overlap: chunk_overlap
    };
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            showToast('Parametry algorytmu RAG zostały zapisane.');
        } else {
            showToast('Wystąpił błąd podczas zapisywania parametrów.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    }
}

async function handleSaveLLMSettings(event) {
    event.preventDefault();
    
    const apiKey = document.getElementById('setting-api-key').value.trim();
    const llmProvider = document.getElementById('setting-llm-provider').value;
    const modelName = document.getElementById('setting-model-name').value;
    const temp = document.getElementById('setting-temp').value;
    const thinkingLevel = document.getElementById('setting-thinking-level').value;
    const ollamaBaseUrl = document.getElementById('setting-ollama-url').value.trim();
    const ollamaModel = document.getElementById('setting-ollama-model').value.trim();
    const ollamaThink = document.getElementById('setting-ollama-think').checked;
    
    const payload = {
        llm_provider: llmProvider,
        gemini_model: modelName,
        gemini_temp: temp,
        gemini_thinking_level: thinkingLevel,
        ollama_base_url: ollamaBaseUrl,
        ollama_model: ollamaModel,
        ollama_think: ollamaThink
    };
    
    // Only include API key if the user typed something (to prevent overwriting with empty text)
    if (apiKey) {
        payload.gemini_api_key = apiKey;
    }
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            showToast('Ustawienia LLM i klucza API zostały zaktualizowane.');
            document.getElementById('setting-api-key').value = ''; // clear input for security
        } else {
            showToast('Błąd podczas zapisywania parametrów LLM.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const providerSelect = document.getElementById('setting-llm-provider');
    if (providerSelect) {
        providerSelect.addEventListener('change', updateLLMProviderFields);
        updateLLMProviderFields();
    }
});


// ================= LOG CONSOLE FUNCTIONS =================

function toggleLogConsole() {
    const wrapper = document.getElementById('log-console-wrapper');
    if (wrapper.classList.contains('hidden')) {
        expandLogConsole();
    } else {
        collapseLogConsole();
    }
}

function expandLogConsole() {
    const wrapper = document.getElementById('log-console-wrapper');
    const chevron = document.getElementById('log-console-chevron');
    const container = document.getElementById('log-console-container');
    
    if (wrapper) wrapper.classList.remove('hidden');
    if (chevron) {
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-up');
    }
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function collapseLogConsole() {
    const wrapper = document.getElementById('log-console-wrapper');
    const chevron = document.getElementById('log-console-chevron');
    
    if (wrapper) wrapper.classList.add('hidden');
    if (chevron) {
        chevron.classList.remove('fa-chevron-up');
        chevron.classList.add('fa-chevron-down');
    }
}

function copyLogsToClipboard() {
    const container = document.getElementById('log-console-container');
    if (!container) return;
    
    const textToCopy = container.innerText;
    navigator.clipboard.writeText(textToCopy).then(() => {
        showToast('Logi analizy zostały skopiowane do schowka!');
    }).catch(err => {
        showToast('Nie udało się skopiować logów.', 'error');
    });
}


function exportToDocx() {
    if (!currentAnalysisData) return;
    const url = `/api/analyze/${currentAnalysisData.id}/export`;
    window.location.href = url;
}

function exportListToDocx() {
    if (!currentAnalysisData) return;
    const url = `/api/analyze/${currentAnalysisData.id}/export-list`;
    window.location.href = url;
}


// ================= EDIT CORPUS MODAL =================

function openEditCorpusModal(docId) {
    const doc = currentCorpusDocs.find(d => d.id === docId);
    if (!doc) return;
    
    document.getElementById('edit-doc-id').value = doc.id;
    document.getElementById('edit-doc-title').value = doc.title;
    document.getElementById('edit-doc-author').value = doc.author === 'Unknown' ? '' : doc.author;
    
    const modal = document.getElementById('edit-corpus-modal');
    modal.classList.remove('hidden');
    // Force browser reflow to enable animation transition
    modal.offsetHeight;
    modal.classList.remove('opacity-0', 'pointer-events-none');
    modal.querySelector('.modal-content').classList.remove('scale-95', 'opacity-0');
}

function closeEditCorpusModal() {
    const modal = document.getElementById('edit-corpus-modal');
    if (!modal) return;
    
    modal.classList.add('opacity-0', 'pointer-events-none');
    modal.querySelector('.modal-content').classList.add('scale-95', 'opacity-0');
    setTimeout(() => {
        modal.classList.add('hidden');
        document.getElementById('edit-corpus-form').reset();
    }, 300);
}

async function saveCorpusDocument(event) {
    event.preventDefault();
    
    const docId = document.getElementById('edit-doc-id').value;
    const title = document.getElementById('edit-doc-title').value.trim();
    const author = document.getElementById('edit-doc-author').value.trim();
    
    if (!title) {
        showToast('Tytuł dokumentu jest wymagany!', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/corpus/${docId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title, author })
        });
        
        const data = await response.json();
        if (response.ok) {
            showToast('Metadane dokumentu zostały zaktualizowane.');
            closeEditCorpusModal();
            loadCorpusDocuments();
        } else {
            showToast(data.error || 'Błąd zapisu zmian.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    }
}


// ================= EDIT ANALYSIS MODAL =================

function openEditAnalysisModal(analysisId, currentTitle) {
    const modal = document.getElementById('edit-analysis-modal');
    const modalContent = modal.querySelector('.modal-content');
    
    document.getElementById('edit-analysis-id').value = analysisId;
    document.getElementById('edit-analysis-title').value = currentTitle || '';
    
    modal.classList.remove('hidden');
    // Force reflow
    modal.offsetHeight;
    modal.classList.remove('opacity-0', 'pointer-events-none');
    modalContent.classList.remove('scale-95', 'opacity-0');
}

function closeEditAnalysisModal() {
    const modal = document.getElementById('edit-analysis-modal');
    const modalContent = modal.querySelector('.modal-content');
    
    modal.classList.add('opacity-0', 'pointer-events-none');
    modalContent.classList.add('scale-95', 'opacity-0');
    
    setTimeout(() => {
        modal.classList.add('hidden');
        document.getElementById('edit-analysis-id').value = '';
        document.getElementById('edit-analysis-title').value = '';
    }, 300);
}

function triggerEditAnalysisFromResults() {
    if (currentAnalysisData && currentAnalysisData.id) {
        openEditAnalysisModal(currentAnalysisData.id, currentAnalysisData.title);
    }
}

async function saveAnalysisTitle(event) {
    event.preventDefault();
    
    const analysisId = document.getElementById('edit-analysis-id').value;
    const title = document.getElementById('edit-analysis-title').value.trim();
    
    if (!title) {
        showToast('Tytuł analizy jest wymagany!', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/analyze/${analysisId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title })
        });
        
        const data = await response.json();
        if (response.ok) {
            showToast('Tytuł analizy został zaktualizowany.');
            closeEditAnalysisModal();
            
            // Update title in results view if currently active
            if (currentAnalysisData && currentAnalysisData.id == analysisId) {
                currentAnalysisData.title = title;
                document.getElementById('result-analysis-title').innerText = title;
            }
            
            // Reload history to ensure variables are fully sync'd
            loadAnalysisHistory();
        } else {
            showToast(data.error || 'Błąd zapisu zmian.', 'error');
        }
    } catch (err) {
        showToast(`Błąd sieci: ${err.message}`, 'error');
    }
}


// ================= INITIALIZATION =================

document.addEventListener('DOMContentLoaded', () => {
    // Default open tab
    switchTab('corpus');
});
