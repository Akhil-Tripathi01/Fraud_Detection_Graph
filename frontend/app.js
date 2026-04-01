const apiBase = "/api";

const form = document.getElementById("scoreForm");
const scoreResult = document.getElementById("scoreResult");
const metricsEl = document.getElementById("metrics");
const alertsEl = document.getElementById("alerts");
const graphEl = document.getElementById("graph");
const trainMlBtn = document.getElementById("trainMlBtn");
const simulateMlBtn = document.getElementById("simulateMlBtn");
const predictForm = document.getElementById("predictForm");
const mlMetricsEl = document.getElementById("mlMetrics");
const mlPredictEl = document.getElementById("mlPredict");
const mlSimulationEl = document.getElementById("mlSimulation");
const mlDiagnosticsEl = document.getElementById("mlDiagnostics");
const refreshProfileBtn = document.getElementById("refreshProfileBtn");
const refreshResearchBtn = document.getElementById("refreshResearchBtn");
const dataProfileEl = document.getElementById("dataProfile");
const researchBriefEl = document.getElementById("researchBrief");

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return await res.json();
}

function metricCard(label, value) {
  return `<div class="metric"><div class="k">${label}</div><div class="v">${value}</div></div>`;
}

function renderMetrics(data) {
  metricsEl.innerHTML = [
    metricCard("Total Transactions", data.total_transactions),
    metricCard("Average Risk", data.avg_risk_score),
    metricCard("Blocked", data.blocked_count),
    metricCard("Review", data.review_count),
    metricCard("Allow", data.allow_count),
    metricCard("High Risk %", `${data.high_risk_percentage}%`)
  ].join("");
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    alertsEl.innerHTML = "<div class='alert'>No alerts yet. Score a transaction to generate signals.</div>";
    return;
  }
  alertsEl.innerHTML = alerts
    .slice(0, 8)
    .map(
      (a) =>
        `<div class="alert"><strong>${a.transaction_id}</strong> score=${a.risk_score}<span class="badge ${a.decision}">${a.decision}</span><br/>${a.reasons.join(
          " "
        )}</div>`
    )
    .join("");
}

function renderGraph(graph) {
  graphEl.innerHTML = "";
  const width = 900;
  const height = 380;
  const nodes = graph.nodes.slice(0, 40);
  const nodeMap = new Map(nodes.map((n, i) => [n.id, { ...n, i }]));
  const edges = graph.edges.filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target)).slice(0, 120);

  nodes.forEach((node, idx) => {
    const angle = (idx / nodes.length) * Math.PI * 2;
    const radius = 140 + (idx % 5) * 8;
    const x = width / 2 + Math.cos(angle) * radius;
    const y = height / 2 + Math.sin(angle) * radius;
    nodeMap.get(node.id).x = x;
    nodeMap.get(node.id).y = y;
  });

  for (const e of edges) {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", s.x);
    line.setAttribute("y1", s.y);
    line.setAttribute("x2", t.x);
    line.setAttribute("y2", t.y);
    line.setAttribute("stroke", "#b8cbc1");
    line.setAttribute("stroke-width", "1");
    graphEl.appendChild(line);
  }

  for (const node of nodes) {
    const n = nodeMap.get(node.id);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", n.x);
    circle.setAttribute("cy", n.y);
    circle.setAttribute("r", node.type === "user" ? 8 : 6);
    circle.setAttribute("fill", node.type === "user" ? "#0b6b4a" : node.type === "device" ? "#0e4f87" : "#93450a");
    graphEl.appendChild(circle);
  }
}

function showBox(el, html, variant = "allow") {
  el.className = `result ${variant}`;
  el.classList.remove("hidden");
  el.innerHTML = html;
}

function renderMlDiagnostics(result) {
  const cm = result.confusion_matrix || {};
  const featureRows = (result.top_features || [])
    .map((f) => `<tr><td>${f.feature}</td><td>${f.importance}</td></tr>`)
    .join("");

  mlDiagnosticsEl.innerHTML = `
    <div class="diag-card">
      <h3>Training Snapshot</h3>
      <div class="diag-row">Trained At: ${result.trained_at}</div>
      <div class="diag-row">Accounts: ${result.accounts} | Graph: ${result.graph_nodes} nodes / ${result.graph_edges} edges</div>
      <div class="diag-row">Confusion Matrix: TN=${cm.tn ?? "-"} FP=${cm.fp ?? "-"} FN=${cm.fn ?? "-"} TP=${cm.tp ?? "-"}</div>
    </div>
    <div class="diag-card">
      <h3>Top Feature Importances</h3>
      <table class="diag-table">
        <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
        <tbody>${featureRows || "<tr><td colspan='2'>No data</td></tr>"}</tbody>
      </table>
    </div>
  `;
}

function renderDataProfile(profile) {
  const channels = Object.entries(profile.channel_distribution || {})
    .map(([k, v]) => `${k}: ${v}%`)
    .join(" | ");
  const countries = Object.entries(profile.country_distribution || {})
    .map(([k, v]) => `${k}: ${v}%`)
    .join(" | ");
  const riskyRows = (profile.top_risky_devices || [])
    .map(
      (r) =>
        `<tr><td>${r.device_id}</td><td>${r.total_txn}</td><td>${r.fraud_txn}</td><td>${(Number(r.fraud_ratio) * 100).toFixed(
          1
        )}%</td></tr>`
    )
    .join("");

  dataProfileEl.innerHTML = `
    <div class="diag-card">
      <h3>Dataset Profile</h3>
      <div class="diag-row">Transactions: ${profile.transactions} | Accounts: ${profile.accounts}</div>
      <div class="diag-row">Fraud Rate: ${(profile.fraud_rate * 100).toFixed(2)}%</div>
      <div class="diag-row">Average Amount: ${profile.avg_amount} | P95 Amount: ${profile.p95_amount}</div>
      <div class="diag-row">Channel Mix: ${channels}</div>
      <div class="diag-row">Country Mix: ${countries}</div>
    </div>
    <div class="diag-card">
      <h3>Top Risky Devices</h3>
      <table class="diag-table">
        <thead><tr><th>Device</th><th>Total Txn</th><th>Fraud Txn</th><th>Fraud Ratio</th></tr></thead>
        <tbody>${riskyRows || "<tr><td colspan='4'>No data</td></tr>"}</tbody>
      </table>
    </div>
  `;
}

function renderResearchBrief(report) {
  const findings = (report.findings || []).map((x) => `<li>${x}</li>`).join("");
  const recs = (report.recommendations || []).map((x) => `<li>${x}</li>`).join("");
  researchBriefEl.innerHTML = `
    <div class="diag-card">
      <h3>${report.title}</h3>
      <div class="diag-row">${report.summary}</div>
      <h4>Findings</h4>
      <ul class="flat-list">${findings}</ul>
      <h4>Recommendations</h4>
      <ul class="flat-list">${recs}</ul>
    </div>
  `;
}

async function refreshDashboard() {
  const [metrics, alerts, graph] = await Promise.all([
    fetchJson(`${apiBase}/dashboard/metrics`),
    fetchJson(`${apiBase}/alerts?min_score=70`),
    fetchJson(`${apiBase}/graph/summary`)
  ]);
  renderMetrics(metrics);
  renderAlerts(alerts);
  renderGraph(graph);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(form);
  const payload = Object.fromEntries(data.entries());
  payload.amount = Number(payload.amount);
  payload.timestamp = new Date().toISOString();

  try {
    const result = await fetchJson(`${apiBase}/transactions/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    scoreResult.className = `result ${result.decision}`;
    scoreResult.classList.remove("hidden");
    scoreResult.innerHTML = `
      <strong>Risk Score: ${result.risk_score}</strong>
      <span class="badge ${result.decision}">${result.decision.toUpperCase()}</span>
      <div>${result.reasons.join(" ")}</div>
    `;
    await refreshDashboard();
  } catch (err) {
    scoreResult.className = "result block";
    scoreResult.classList.remove("hidden");
    scoreResult.textContent = `Error: ${err.message}`;
  }
});

refreshDashboard().catch((err) => {
  alertsEl.innerHTML = `<div class='alert'>Failed to load dashboard: ${err.message}</div>`;
});

trainMlBtn.addEventListener("click", async () => {
  trainMlBtn.disabled = true;
  trainMlBtn.textContent = "Training...";
  try {
    const result = await fetchJson(`${apiBase}/ml/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        n_transactions: 3000,
        n_accounts: 500,
        fraud_rate: 0.08,
        random_seed: 42
      })
    });

    showBox(
      mlMetricsEl,
      `<strong>Model Trained</strong><br/>
       Accounts=${result.accounts}, Graph=${result.graph_nodes} nodes / ${result.graph_edges} edges<br/>
       Accuracy=${result.metrics.accuracy}, Precision=${result.metrics.precision}, Recall=${result.metrics.recall}, F1=${result.metrics.f1}, ROC-AUC=${result.metrics.roc_auc}`,
      "allow"
    );
    renderMlDiagnostics(result);
  } catch (err) {
    showBox(mlMetricsEl, `Training failed: ${err.message}`, "block");
  } finally {
    trainMlBtn.disabled = false;
    trainMlBtn.textContent = "Train Graph ML Pipeline";
  }
});

predictForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = Object.fromEntries(new FormData(predictForm).entries());
  try {
    const result = await fetchJson(`${apiBase}/ml/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const variant = result.risk_tier === "HIGH" ? "block" : result.risk_tier === "MEDIUM" ? "review" : "allow";
    const foundTag = result.account_found ? "" : "<br/><em>Requested account not found in graph. Fallback profile used.</em>";
    showBox(
      mlPredictEl,
      `<strong>${result.account_id}</strong> -> Risk ${result.risk_tier}<br/>
       Fraud Probability=${result.fraud_probability}<br/>
       ${result.explanation.join(" ")}${foundTag}`,
      variant
    );
  } catch (err) {
    showBox(mlPredictEl, `Prediction failed: ${err.message}`, "block");
  }
});

async function refreshMlOverview() {
  try {
    const result = await fetchJson(`${apiBase}/ml/metrics`);
    showBox(
      mlMetricsEl,
      `<strong>Model Ready</strong><br/>
       Accounts=${result.accounts}, Graph=${result.graph_nodes} nodes / ${result.graph_edges} edges<br/>
       Accuracy=${result.metrics.accuracy}, Precision=${result.metrics.precision}, Recall=${result.metrics.recall}, F1=${result.metrics.f1}, ROC-AUC=${result.metrics.roc_auc}`,
      "allow"
    );
    renderMlDiagnostics(result);
  } catch {
    // Keep quiet if model isn't trained yet.
  }
}

refreshMlOverview();

async function refreshDataAndResearch() {
  try {
    const [profile, research] = await Promise.all([
      fetchJson(`${apiBase}/ml/data-profile`),
      fetchJson(`${apiBase}/ml/research`)
    ]);
    renderDataProfile(profile);
    renderResearchBrief(research);
  } catch (err) {
    dataProfileEl.innerHTML = `<div class="alert">Unable to load data profile: ${err.message}</div>`;
    researchBriefEl.innerHTML = `<div class="alert">Unable to load research brief: ${err.message}</div>`;
  }
}

refreshProfileBtn.addEventListener("click", refreshDataAndResearch);
refreshResearchBtn.addEventListener("click", refreshDataAndResearch);
refreshDataAndResearch();

simulateMlBtn.addEventListener("click", async () => {
  try {
    const rows = await fetchJson(`${apiBase}/ml/simulate?n=5`);
    mlSimulationEl.innerHTML = rows
      .map(
        (r) =>
          `<div class="alert"><strong>${r.account_id}</strong> prob=${r.fraud_probability} tier=<span class="badge ${
            r.risk_tier === "HIGH" ? "block" : r.risk_tier === "MEDIUM" ? "review" : "allow"
          }">${r.risk_tier}</span></div>`
      )
      .join("");
  } catch (err) {
    mlSimulationEl.innerHTML = `<div class="alert">Simulation failed: ${err.message}</div>`;
  }
});
