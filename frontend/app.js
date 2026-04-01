const apiBase = "/api";

const form = document.getElementById("scoreForm");
const scoreResult = document.getElementById("scoreResult");
const metricsEl = document.getElementById("metrics");
const alertsEl = document.getElementById("alerts");
const graphEl = document.getElementById("graph");

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
