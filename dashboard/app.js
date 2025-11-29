// API Configuration
const API_BASE_URL = '/api/v1';

// State
let currentEvaluationId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeForm();
});

// Tab Management
function initializeTabs() {
    // Only one tab now (Evaluate), so no tab switching needed
    // Keep function for future extensibility
}

// Form Handling
function initializeForm() {
    const form = document.getElementById('evaluateForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleEvaluate();
    });
}

async function handleEvaluate() {
    const promptId = document.getElementById('promptId').value || generateId();
    const userPrompt = document.getElementById('userPrompt').value;
    const targetOutput = document.getElementById('targetOutput').value;
    const groundTruth = document.getElementById('groundTruth').value;
    const taskType = document.getElementById('taskType').value;
    const language = document.getElementById('language').value;

    // Show loading
    document.getElementById('evaluateBtn').style.display = 'none';
    document.getElementById('evaluateLoader').style.display = 'inline-block';

    try {
        const payload = {
            prompt_id: promptId,
            user_prompt: userPrompt,
            target_output: targetOutput,
            metadata: {
                task_type: taskType,
                language: language,
                eval_purpose: 'safety_and_quality'
            }
        };

        // Add ground truth if provided
        if (groundTruth.trim()) {
            payload.ground_truth = {
                type: 'text',
                content: groundTruth,
                sources: []
            };
        }

        const response = await fetch(`${API_BASE_URL}/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const result = await response.json();
        displayEvaluationResult(result);

    } catch (error) {
        alert(`Evaluation failed: ${error.message}`);
        console.error('Evaluation error:', error);
    } finally {
        // Hide loading
        document.getElementById('evaluateBtn').style.display = 'inline';
        document.getElementById('evaluateLoader').style.display = 'none';
    }
}

function displayEvaluationResult(result) {
    const container = document.getElementById('evaluationResult');
    const content = document.getElementById('resultContent');

    const judge = result.judge_output;

    // Score grid
    const scoresHTML = `
        <div class="score-grid">
            <div class="score-card">
                <h4>Hallucination</h4>
                <div class="score-value ${getSeverityClass(judge.hallucination_probability_pct)}">
                    ${judge.hallucination_probability_pct.toFixed(1)}%
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill ${getSeverityClass(judge.hallucination_probability_pct)}" 
                         style="width: ${judge.hallucination_probability_pct}%"></div>
                </div>
            </div>

            <div class="score-card">
                <h4>Jailbreak</h4>
                <div class="score-value ${getSeverityClass(judge.jailbreak_probability_pct)}">
                    ${judge.jailbreak_probability_pct.toFixed(1)}%
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill ${getSeverityClass(judge.jailbreak_probability_pct)}" 
                         style="width: ${judge.jailbreak_probability_pct}%"></div>
                </div>
            </div>

            <div class="score-card">
                <h4>Fake News</h4>
                <div class="score-value ${getSeverityClass(judge.fake_news_probability_pct)}">
                    ${judge.fake_news_probability_pct.toFixed(1)}%
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill ${getSeverityClass(judge.fake_news_probability_pct)}" 
                         style="width: ${judge.fake_news_probability_pct}%"></div>
                </div>
            </div>

            <div class="score-card">
                <h4>Wrong Output</h4>
                <div class="score-value ${getSeverityClass(judge.wrong_output_probability_pct)}">
                    ${judge.wrong_output_probability_pct.toFixed(1)}%
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill ${getSeverityClass(judge.wrong_output_probability_pct)}" 
                         style="width: ${judge.wrong_output_probability_pct}%"></div>
                </div>
            </div>
        </div>

        <div style="margin-top: 1.5rem;">
            <h4 style="color: var(--text-secondary); margin-bottom: 0.75rem;">Analysis</h4>
            <p style="color: var(--text-primary); line-height: 1.8;">${judge.analysis_reasoning}</p>
        </div>

        <div style="margin-top: 1.5rem;">
            <h4 style="color: var(--text-secondary); margin-bottom: 0.75rem;">Token Analysis</h4>
            <p style="color: var(--text-primary);">
                Total Tokens: <strong>${result.total_output_tokens}</strong> | 
                Estimated Hallucinated: <strong>${result.estimated_hallucinated_tokens}</strong> 
                (<strong>${(judge.hallucination_token_fraction_estimate * 100).toFixed(1)}%</strong>)
            </p>
        </div>

        <div class="segments-container">
            <h4 style="color: var(--text-secondary); margin-bottom: 0.75rem;">Segment Analysis</h4>
            <div style="line-height: 2;">
                ${judge.segment_labels.map(seg => renderSegment(seg)).join(' ')}
            </div>
        </div>
    `;

    content.innerHTML = scoresHTML;
    container.style.display = 'block';

    // Auto-scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderSegment(segment) {
    let className = 'correct';
    if (segment.is_safety_violation) className = 'safety-violation';
    else if (segment.is_hallucination) className = 'hallucination';
    else if (segment.is_potential_fake_news) className = 'fake-news';

    return `<span class="segment ${className}" title="${segment.label || ''}">${segment.text || ''}</span>`;
}


// Display Multi-Judge Result
function displayMultiJudgeResult(result) {
    const resultDiv = document.getElementById('evaluationResult');

    let html = '<div class="multi-judge-results">';

    // Super Judge Consensus (Main Display)
    if (result.super_judge_consensus) {
        const superJudge = result.super_judge_consensus;
        const scores = superJudge.scores;

        html += `
            <div class="super-judge-card">
                <h3>🌟 Super Judge Consensus</h3>
                <div class="confidence-badge confidence-${superJudge.confidence}">
                    ${superJudge.confidence.toUpperCase()} CONFIDENCE
                </div>
                
                <div class="score-grid">
                    <div class="score-item ${getScoreClass(scores.hallucination_probability_pct)}">
                        <div class="score-label">Hallucination</div>
                        <div class="score-value">${scores.hallucination_probability_pct.toFixed(1)}%</div>
                    </div>
                    <div class="score-item ${getScoreClass(scores.jailbreak_probability_pct)}">
                        <div class="score-label">Jailbreak</div>
                        <div class="score-value">${scores.jailbreak_probability_pct.toFixed(1)}%</div>
                    </div>
                    <div class="score-item ${getScoreClass(scores.fake_news_probability_pct)}">
                        <div class="score-label">Fake News</div>
                        <div class="score-value">${scores.fake_news_probability_pct.toFixed(1)}%</div>
                    </div>
                    <div class="score-item ${getScoreClass(scores.wrong_output_probability_pct)}">
                        <div class="score-label">Wrong Output</div>
                        <div class="score-value">${scores.wrong_output_probability_pct.toFixed(1)}%</div>
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h4>💭 Super Judge Analysis</h4>
                    <p>${superJudge.reasoning}</p>
                </div>
                
                ${superJudge.key_insights ? `
                    <div class="insights-section">
                        <h4>💡 Key Insights</h4>
                        <p>${superJudge.key_insights}</p>
                    </div>
                ` : ''}
                
                ${superJudge.agreement_level ? `
                    <div class="agreement-badge">
                        🤝 Agreement Level: ${superJudge.agreement_level.toUpperCase()}
                    </div>
                ` : ''}
            </div>
        `;
    }

    // Individual Judges
    html += '<div class="individual-judges"><h3>👥 Individual Judge Evaluations</h3>';

    result.individual_judges.forEach((judge, index) => {
        const output = judge.output;
        html += `
            <div class="judge-card">
                <div class="judge-header">
                    <span class="judge-name">Judge ${index + 1}: ${judge.provider}</span>
                    <span class="judge-model">${judge.model}</span>
                </div>
                <div class="score-row">
                    <div class="mini-score">
                        <span class="mini-label">Hallucination</span>
                        <span class="mini-value ${getScoreClass(output.hallucination_probability_pct)}">${output.hallucination_probability_pct.toFixed(1)}%</span>
                    </div>
                    <div class="mini-score">
                        <span class="mini-label">Jailbreak</span>
                        <span class="mini-value ${getScoreClass(output.jailbreak_probability_pct)}">${output.jailbreak_probability_pct.toFixed(1)}%</span>
                    </div>
                    <div class="mini-score">
                        <span class="mini-label">Fake News</span>
                        <span class="mini-value ${getScoreClass(output.fake_news_probability_pct)}">${output.fake_news_probability_pct.toFixed(1)}%</span>
                    </div>
                    <div class="mini-score">
                        <span class="mini-label">Wrong</span>
                        <span class="mini-value ${getScoreClass(output.wrong_output_probability_pct)}">${output.wrong_output_probability_pct.toFixed(1)}%</span>
                    </div>
                </div>
                <div class="judge-reasoning">
                    <small>${output.analysis_reasoning.substring(0, 200)}...</small>
                </div>
            </div>
        `;
    });

    html += '</div>';

    // Basic Consensus
    if (result.basic_consensus) {
        const basic = result.basic_consensus;
        html += `
            <div class="basic-consensus">
                <h4>📊 Basic Consensus (Average)</h4>
                <div class="consensus-scores">
                    <span>Hallucination: ${basic.hallucination_probability_pct.toFixed(1)}%</span>
                    <span>Jailbreak: ${basic.jailbreak_probability_pct.toFixed(1)}%</span>
                    <span>Fake News: ${basic.fake_news_probability_pct.toFixed(1)}%</span>
                    <span>Wrong: ${basic.wrong_output_probability_pct.toFixed(1)}%</span>
                </div>
            </div>
        `;
    }

    html += '</div>';
    resultDiv.innerHTML = html;

    // Auto-scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Helper function to get score class
function getScoreClass(score) {
    if (score < 20) return 'score-low';
    if (score < 50) return 'score-medium';
    return 'score-high';
}

// Load Recent Results
async function loadRecentResults() {
    try {
        const response = await fetch(`${API_BASE_URL}/results?limit=10`);
        const data = await response.json();

        const resultsDiv = document.getElementById('recentResults');

        if (!data.evaluations || data.evaluations.length === 0) {
            resultsDiv.innerHTML = '<p>No evaluations yet</p>';
            return;
        }

        resultsDiv.innerHTML = data.evaluations.map(evalData => `
            <div class="result-item">
                <div class="result-header">
                    <strong>${evalData.prompt_id || 'Unknown'}</strong>
                    <span class="result-date">${new Date(evalData.timestamp || Date.now()).toLocaleString()}</span>
                </div>
                <div class="result-scores">
                    ${evalData.judge_output ? `
                        <span>H: ${evalData.judge_output.hallucination_probability_pct.toFixed(0)}%</span>
                        <span>J: ${evalData.judge_output.jailbreak_probability_pct.toFixed(0)}%</span>
                        <span>F: ${evalData.judge_output.fake_news_probability_pct.toFixed(0)}%</span>
                    ` : 'Multi-judge evaluation'}
                </div>
            </div>
        `).join('');
    } catch (error) {
        document.getElementById('recentResults').innerHTML = `<p class="error">Failed to load results</p>`;
    }
}

// Load Metrics
async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics/summary`);
        const data = await response.json();

        const metricsDiv = document.getElementById('metricsDisplay');

        if (data.error) {
            metricsDiv.innerHTML = '<p>No metrics available yet</p>';
            return;
        }

        const metrics = data.dataset_metrics;
        metricsDiv.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Average Hallucination</h4>
                    <div class="metric-value">${metrics.avg_hallucination_pct.toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <h4>Average Jailbreak</h4>
                    <div class="metric-value">${metrics.avg_jailbreak_pct.toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <h4>Average Fake News</h4>
                    <div class="metric-value">${metrics.avg_fake_news_pct.toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <h4>Total Evaluations</h4>
                    <div class="metric-value">${metrics.total_evaluations}</div>
                </div>
            </div>
        `;
    } catch (error) {
        document.getElementById('metricsDisplay').innerHTML = `<p class="error">Failed to load metrics</p>`;
    }
}



function generateId() {
    return `eval_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getSeverityClass(percentage) {
    if (percentage < 20) return 'low';
    if (percentage < 50) return 'medium';
    return 'high';
}



// Load Results
async function loadResults() {
    try {
        const response = await fetch(`${API_BASE_URL}/results?limit=10`);
        const data = await response.json();

        const resultsList = document.getElementById('resultsList');

        if (!data.evaluations || data.evaluations.length === 0) {
            resultsList.innerHTML = '<p class="loading">No evaluations yet. Create your first evaluation in the Evaluate tab!</p>';
            return;
        }

        resultsList.innerHTML = data.evaluations.map(evalData => `
            <div class="result-item">
                <div class="result-header">
                    <strong>${evalData.prompt_id || 'Unknown'}</strong>
                    <span class="result-date">${new Date(evalData.timestamp || Date.now()).toLocaleString()}</span>
                </div>
                <div class="result-scores">
                    ${evalData.judge_output ? `
                        <span>H: ${evalData.judge_output.hallucination_probability_pct.toFixed(0)}%</span>
                        <span>J: ${evalData.judge_output.jailbreak_probability_pct.toFixed(0)}%</span>
                        <span>F: ${evalData.judge_output.fake_news_probability_pct.toFixed(0)}%</span>
                        <span>W: ${evalData.judge_output.wrong_output_probability_pct.toFixed(0)}%</span>
                    ` : 'Multi-judge evaluation'}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load results:', error);
        document.getElementById('resultsList').innerHTML = '<p class="error">Failed to load results. Make sure the API is running.</p>';
    }
}

