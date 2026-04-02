const apiBase = "/api";

const scoreForm = document.getElementById("scoreForm");
const scoreResult = document.getElementById("scoreResult");
const trainForm = document.getElementById("trainForm");
const trainMlBtn = document.getElementById("trainMlBtn");
const predictForm = document.getElementById("predictForm");
const mlMetricsEl = document.getElementById("mlMetrics");
const mlPredictEl = document.getElementById("mlPredict");
const mlSimulationEl = document.getElementById("mlSimulation");
const mlDiagnosticsEl = document.getElementById("mlDiagnostics");
const dataProfileEl = document.getElementById("dataProfile");
const researchBriefEl = document.getElementById("researchBrief");
const alertsEl = document.getElementById("alerts");
const graphEl = document.getElementById("graph");
const overviewCardsEl = document.getElementById("overviewCards");
const modelCatalogEl = document.getElementById("modelCatalog");
const configPickerEl = document.getElementById("configPicker");
const runConfigBtn = document.getElementById("runConfigBtn");
const configStatusEl = document.getElementById("configStatus");
const visualSummaryEl = document.getElementById("visualSummary");
const caseSummaryEl = document.getElementById("caseSummary");
const caseTableBodyEl = document.getElementById("caseTableBody");
const caseSearchEl = document.getElementById("caseSearch");
const caseDecisionFilterEl = document.getElementById("caseDecisionFilter");
const refreshAllBtn = document.getElementById("refreshAllBtn");
const loadExamplesBtn = document.getElementById("loadExamplesBtn");
const simulateMlBtn = document.getElementById("simulateMlBtn");
const healthStatusEl = document.getElementById("healthStatus");
const mlStatusHeroEl = document.getElementById("mlStatusHero");
const caseCountHeroEl = document.getElementById("caseCountHero");

let allCases = [];
let availableConfigs = [];

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}

function badgeClass(decision) {
  return decision === "block" ? "block" : decision === "review" ? "review" : "allow";
}

function showBox(element, html, variant = "allow") {
  element.className = `result ${variant}`;
  element.classList.remove("hidden");
  element.innerHTML = html;
}

function renderOverview(metrics, graph, mlStatus, caseSummary) {
  const cards = [
    ["Transactions", metrics.total_transactions, "Seeded and live-scored transaction volume."],
    ["Average Risk", metrics.avg_risk_score, "Average risk score across scored transactions."],
    ["Suspicious Components", graph.suspicious_components, "Connected user/device/IP clusters under watch."],
    ["Saved Models", mlStatus.model_count, "Available serialized model versions on disk."],
    ["ML Trained", mlStatus.trained ? "Yes" : "No", "Model pipeline readiness for account inference."],
    ["Example Case Blocks", caseSummary.blocked_count, "High-risk examples in the 100-case library."]
  ];

  overviewCardsEl.innerHTML = cards
    .map(
      ([title, value, desc]) => `
        <article class="overview-card">
          <h3>${title}</h3>
          <strong>${value}</strong>
          <p>${desc}</p>
        </article>
      `
    )
    .join("");
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    alertsEl.innerHTML = "<div class='alert'>No high-risk alerts yet. Score a transaction to populate the queue.</div>";
    return;
  }

  alertsEl.innerHTML = alerts
    .slice(0, 8)
    .map(
      (alert) => `
        <div class="alert">
          <strong>${alert.transaction_id}</strong>
          <span class="badge ${badgeClass(alert.decision)}">${alert.decision}</span>
          <div class="muted">Risk score ${alert.risk_score}</div>
          <div>${alert.reasons.join(" ")}</div>
        </div>
      `
    )
    .join("");
}

function renderGraph(graph) {
  graphEl.innerHTML = "";
  const width = 960;
  const height = 420;
  const nodes = graph.nodes.slice(0, 42);
  const nodeMap = new Map(nodes.map((node) => [node.id, { ...node }]));
  const edges = graph.edges.filter((edge) => nodeMap.has(edge.source) && nodeMap.has(edge.target)).slice(0, 140);

  nodes.forEach((node, index) => {
    const ring = index % 3;
    const angle = (index / Math.max(nodes.length, 1)) * Math.PI * 2;
    const radius = ring === 0 ? 98 : ring === 1 ? 148 : 190;
    const x = width / 2 + Math.cos(angle) * radius;
    const y = height / 2 + Math.sin(angle) * (radius * 0.7);
    const entry = nodeMap.get(node.id);
    entry.x = x;
    entry.y = y;
  });

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("stroke", "rgba(24, 33, 40, 0.18)");
    line.setAttribute("stroke-width", "1.2");
    graphEl.appendChild(line);
  });

  nodes.forEach((node) => {
    const entry = nodeMap.get(node.id);
    const color = node.type === "user" ? "#0f6c68" : node.type === "device" ? "#a85c1f" : "#3a5575";
    const radius = node.type === "user" ? 7.5 : 5.8;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", entry.x);
    circle.setAttribute("cy", entry.y);
    circle.setAttribute("r", radius);
    circle.setAttribute("fill", color);
    graphEl.appendChild(circle);
  });
}

function renderMlDiagnostics(result) {
  const metrics = result.metrics || {};
  const confusion = result.confusion_matrix || {};
  const topFeatures = (result.top_features || [])
    .map((feature) => `<tr><td>${feature.feature}</td><td>${feature.importance}</td></tr>`)
    .join("");

  mlDiagnosticsEl.innerHTML = `
    <div class="diag-card">
      <h3>Model Quality</h3>
      <div class="diag-row">Accuracy: ${metrics.accuracy ?? "-"}</div>
      <div class="diag-row">Precision: ${metrics.precision ?? "-"}</div>
      <div class="diag-row">Recall: ${metrics.recall ?? "-"}</div>
      <div class="diag-row">F1: ${metrics.f1 ?? "-"}</div>
      <div class="diag-row">ROC-AUC: ${metrics.roc_auc ?? "-"}</div>
    </div>
    <div class="diag-card">
      <h3>Graph Snapshot</h3>
      <div class="diag-row">Trained at: ${result.trained_at ?? "-"}</div>
      <div class="diag-row">Accounts: ${result.accounts ?? "-"}</div>
      <div class="diag-row">Graph: ${result.graph_nodes ?? "-"} nodes / ${result.graph_edges ?? "-"} edges</div>
      <div class="diag-row">Confusion Matrix: TN=${confusion.tn ?? "-"} FP=${confusion.fp ?? "-"} FN=${confusion.fn ?? "-"} TP=${confusion.tp ?? "-"}</div>
    </div>
    <div class="diag-card">
      <h3>Top Feature Importances</h3>
      <table class="diag-table">
        <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
        <tbody>${topFeatures || "<tr><td colspan='2'>No diagnostics available.</td></tr>"}</tbody>
      </table>
    </div>
  `;
}

function renderDataProfile(profile) {
  const channels = Object.entries(profile.channel_distribution || {})
    .map(([key, value]) => `${key}: ${value}%`)
    .join(" | ");
  const countries = Object.entries(profile.country_distribution || {})
    .map(([key, value]) => `${key}: ${value}%`)
    .join(" | ");
  const riskyRows = (profile.top_risky_devices || [])
    .map(
      (row) => `
        <tr>
          <td>${row.device_id}</td>
          <td>${row.total_txn}</td>
          <td>${row.fraud_txn}</td>
          <td>${(Number(row.fraud_ratio) * 100).toFixed(1)}%</td>
        </tr>
      `
    )
    .join("");

  dataProfileEl.innerHTML = `
    <div class="diag-card">
      <h3>Dataset Profile</h3>
      <div class="diag-row">Transactions: ${profile.transactions}</div>
      <div class="diag-row">Accounts: ${profile.accounts}</div>
      <div class="diag-row">Fraud rate: ${(profile.fraud_rate * 100).toFixed(2)}%</div>
      <div class="diag-row">Average amount: ${profile.avg_amount}</div>
      <div class="diag-row">P95 amount: ${profile.p95_amount}</div>
      <div class="diag-row">Channels: ${channels}</div>
      <div class="diag-row">Countries: ${countries}</div>
    </div>
    <div class="diag-card">
      <h3>Risky Devices</h3>
      <table class="diag-table">
        <thead><tr><th>Device</th><th>Total</th><th>Fraud</th><th>Ratio</th></tr></thead>
        <tbody>${riskyRows || "<tr><td colspan='4'>No device signals available.</td></tr>"}</tbody>
      </table>
    </div>
  `;
}

function renderResearchBrief(report) {
  const findings = (report.findings || []).map((item) => `<li>${item}</li>`).join("");
  const recommendations = (report.recommendations || []).map((item) => `<li>${item}</li>`).join("");

  researchBriefEl.innerHTML = `
    <div class="diag-card">
      <h3>${report.title}</h3>
      <div class="diag-row">${report.summary}</div>
      <h4>Findings</h4>
      <ul class="plain-list">${findings}</ul>
      <h4>Recommendations</h4>
      <ul class="plain-list">${recommendations}</ul>
    </div>
  `;
}

function renderModelCatalog(catalog) {
  modelCatalogEl.innerHTML = catalog
    .map(
      (entry) => `
        <div class="diag-card">
          <h3>${entry.model_name}</h3>
          <div class="diag-row">Family: ${entry.family}</div>
          <div class="diag-row">Status: ${entry.status}</div>
          <div class="diag-row">${entry.description}</div>
        </div>
      `
    )
    .join("");
}

function renderConfigPicker(configs) {
  availableConfigs = configs;
  configPickerEl.innerHTML = configs
    .map((entry) => `<option value="${entry.name}">${entry.name} (${entry.model_name})</option>`)
    .join("");
}

function renderBarRows(rows, valueKey, maxValue) {
  return rows
    .map((row) => {
      const label = row.label || row.feature;
      const value = Number(row[valueKey]);
      const width = maxValue > 0 ? Math.max(8, (value / maxValue) * 100) : 0;
      return `
        <div class="bar-row">
          <span>${label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${value}</strong>
        </div>
      `;
    })
    .join("");
}

function renderVisualSummary(summary) {
  const metricsMax = Math.max(...summary.metric_series.map((item) => Number(item.value)), 1);
  const featureMax = Math.max(...summary.feature_importance_series.map((item) => Number(item.importance)), 1);
  const riskMax = Math.max(...summary.risk_distribution.map((item) => Number(item.count)), 1);

  visualSummaryEl.innerHTML = `
    <div class="chart-card">
      <h4>Metrics</h4>
      <div class="bar-list">${renderBarRows(summary.metric_series, "value", metricsMax)}</div>
    </div>
    <div class="chart-card">
      <h4>Top Features</h4>
      <div class="bar-list">${renderBarRows(summary.feature_importance_series, "importance", featureMax)}</div>
    </div>
    <div class="chart-card">
      <h4>Risk Distribution</h4>
      <div class="bar-list">${renderBarRows(summary.risk_distribution, "count", riskMax)}</div>
    </div>
  `;
}

function renderCaseSummary(summary) {
  caseCountHeroEl.textContent = `${summary.total_cases} seeded scenarios`;
  caseSummaryEl.innerHTML = [
    ["Total Cases", summary.total_cases],
    ["Blocked", summary.blocked_count],
    ["Review", summary.review_count],
    ["Allow", summary.allow_count],
    ["Avg Score", summary.average_risk_score]
  ]
    .map(
      ([label, value]) => `
        <div class="mini-stat">
          <span class="label">${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
}

function renderCaseTable() {
  const search = caseSearchEl.value.trim().toLowerCase();
  const decision = caseDecisionFilterEl.value;

  const filtered = allCases.filter((item) => {
    const matchesDecision = decision === "all" || item.decision === decision;
    const haystack = [
      item.case_id,
      item.scenario,
      item.transaction.user_id,
      item.transaction.transaction_id,
      item.expected_pattern
    ]
      .join(" ")
      .toLowerCase();
    return matchesDecision && (!search || haystack.includes(search));
  });

  caseTableBodyEl.innerHTML = filtered.length
    ? filtered
        .map(
          (item) => `
            <tr>
              <td>${item.case_id}</td>
              <td>${item.scenario}</td>
              <td>${item.transaction.user_id}</td>
              <td>${item.transaction.amount}</td>
              <td><span class="badge ${badgeClass(item.decision)}">${item.decision}</span></td>
              <td>${item.risk_score}</td>
              <td>${item.reasons[0] || item.expected_pattern}</td>
            </tr>
          `
        )
        .join("")
    : "<tr><td colspan='7'>No example cases match the current filter.</td></tr>";
}

async function refreshPlatform() {
  const [health, metrics, alerts, graph, mlStatus, caseSummary, profile, research, visual, catalog, configs] = await Promise.all([
    fetchJson(`${apiBase}/health`),
    fetchJson(`${apiBase}/dashboard/metrics`),
    fetchJson(`${apiBase}/alerts?min_score=70`),
    fetchJson(`${apiBase}/graph/summary`),
    fetchJson(`${apiBase}/ml/status`),
    fetchJson(`${apiBase}/example-cases/summary`),
    fetchJson(`${apiBase}/ml/data-profile`),
    fetchJson(`${apiBase}/ml/research`),
    fetchJson(`${apiBase}/ml/visual-summary`),
    fetchJson(`${apiBase}/ml/model-catalog`),
    fetchJson(`${apiBase}/ml/configs`)
  ]);

  healthStatusEl.textContent = health.status === "ok" ? "Healthy" : "Check service";
  mlStatusHeroEl.textContent = mlStatus.trained ? "Model ready" : "Not trained";
  renderOverview(metrics, graph, mlStatus, caseSummary);
  renderAlerts(alerts);
  renderGraph(graph);
  renderCaseSummary(caseSummary);
  renderDataProfile(profile);
  renderResearchBrief(research);
  renderVisualSummary(visual);
  renderModelCatalog(catalog);
  renderConfigPicker(configs);
}

async function loadExampleCases() {
  allCases = await fetchJson(`${apiBase}/example-cases`);
  renderCaseTable();
}

scoreForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(scoreForm).entries());
  payload.amount = Number(payload.amount);
  payload.timestamp = new Date().toISOString();

  try {
    const result = await fetchJson(`${apiBase}/transactions/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    showBox(
      scoreResult,
      `<strong>Risk Score ${result.risk_score}</strong>
       <span class="badge ${badgeClass(result.decision)}">${result.decision}</span>
       <div>${result.reasons.join(" ")}</div>`,
      badgeClass(result.decision)
    );
    await refreshPlatform();
  } catch (error) {
    showBox(scoreResult, `Unable to score transaction: ${error.message}`, "block");
  }
});

trainForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(trainForm).entries());
  payload.n_transactions = Number(payload.n_transactions);
  payload.n_accounts = Number(payload.n_accounts);
  payload.fraud_rate = Number(payload.fraud_rate);
  payload.random_seed = Number(payload.random_seed);

  trainMlBtn.disabled = true;
  trainMlBtn.textContent = "Training...";
  try {
    const result = await fetchJson(`${apiBase}/ml/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    showBox(
      mlMetricsEl,
      `Pipeline trained with ${result.transactions} transactions and ${result.accounts} accounts. ROC-AUC ${result.metrics.roc_auc}.`,
      "allow"
    );
    renderMlDiagnostics(result);
    await refreshPlatform();
  } catch (error) {
    showBox(mlMetricsEl, `Training failed: ${error.message}`, "block");
  } finally {
    trainMlBtn.disabled = false;
    trainMlBtn.textContent = "Train Pipeline";
  }
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(predictForm).entries());
  payload.threshold = Number(payload.threshold);

  try {
    const result = await fetchJson(`${apiBase}/ml/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const variant = result.risk_tier === "HIGH" ? "block" : result.risk_tier === "MEDIUM" ? "review" : "allow";
    showBox(
      mlPredictEl,
      `${result.account_id} is ${result.risk_tier} risk with fraud probability ${result.fraud_probability}. ${result.explanation.join(" ")}`,
      variant
    );
  } catch (error) {
    showBox(mlPredictEl, `Prediction failed: ${error.message}`, "block");
  }
});

simulateMlBtn.addEventListener("click", async () => {
  try {
    const rows = await fetchJson(`${apiBase}/ml/simulate?n=5`);
    mlSimulationEl.innerHTML = rows
      .map(
        (row) => `
          <div class="alert">
            <strong>${row.account_id}</strong>
            <span class="badge ${row.risk_tier === "HIGH" ? "block" : row.risk_tier === "MEDIUM" ? "review" : "allow"}">${row.risk_tier}</span>
            <div class="muted">Fraud probability ${row.fraud_probability}</div>
            <div>${row.explanation.join(" ")}</div>
          </div>
        `
      )
      .join("");
  } catch (error) {
    mlSimulationEl.innerHTML = `<div class="alert">Simulation failed: ${error.message}</div>`;
  }
});

runConfigBtn.addEventListener("click", async () => {
  const selected = configPickerEl.value;
  if (!selected) {
    showBox(configStatusEl, "No config available to run.", "review");
    return;
  }

  runConfigBtn.disabled = true;
  runConfigBtn.textContent = "Running...";
  try {
    const result = await fetchJson(`${apiBase}/ml/run-config?config_name=${encodeURIComponent(selected)}`, {
      method: "POST"
    });
    showBox(
      configStatusEl,
      `Config ${selected} ran with ${result.model_name}. Accuracy ${result.metrics.accuracy}, ROC-AUC ${result.metrics.roc_auc}, saved tag ${result.model_tag}.`,
      "allow"
    );
    await Promise.all([refreshPlatform(), refreshMlOverview()]);
  } catch (error) {
    showBox(configStatusEl, `Config run failed: ${error.message}`, "block");
  } finally {
    runConfigBtn.disabled = false;
    runConfigBtn.textContent = "Run Config";
  }
});

refreshAllBtn.addEventListener("click", async () => {
  refreshAllBtn.disabled = true;
  refreshAllBtn.textContent = "Refreshing...";
  try {
    await Promise.all([refreshPlatform(), loadExampleCases(), refreshMlOverview()]);
  } finally {
    refreshAllBtn.disabled = false;
    refreshAllBtn.textContent = "Refresh Platform";
  }
});

loadExamplesBtn.addEventListener("click", loadExampleCases);
caseSearchEl.addEventListener("input", renderCaseTable);
caseDecisionFilterEl.addEventListener("change", renderCaseTable);

async function refreshMlOverview() {
  try {
    const result = await fetchJson(`${apiBase}/ml/metrics`);
    renderMlDiagnostics(result);
    showBox(
      mlMetricsEl,
      `Current model loaded. Accuracy ${result.metrics.accuracy}, F1 ${result.metrics.f1}, ROC-AUC ${result.metrics.roc_auc}.`,
      "allow"
    );
  } catch {
    mlDiagnosticsEl.innerHTML = "<div class='diag-card'>Train the pipeline to populate model diagnostics.</div>";
  }
}

Promise.all([refreshPlatform(), loadExampleCases(), refreshMlOverview()]).catch((error) => {
  alertsEl.innerHTML = `<div class="alert">Initial load failed: ${error.message}</div>`;
});
