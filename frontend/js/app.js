/* ConveyorGuard dashboard - plain JS, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
async function api(path, method = 'GET', body) {
  const r = await fetch(path, { method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) { let m = r.statusText; try { const j = await r.json(); m = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) { } throw new Error(m); }
  return r.json();
}
const fmtTime = ts => new Date(ts * 1000).toLocaleTimeString();
const fmtDT = ts => new Date(ts * 1000).toLocaleString();

let SETTINGS = {};
const SENSORS = [
  { key: 'vibration', label: 'Vibration', unit: 'm/s²', warn: 'vib_warn', crit: 'vib_crit' },
  { key: 'load', label: 'Load', unit: 'kg', warn: 'load_warn', crit: 'load_crit' },
  { key: 'speed', label: 'Belt speed', unit: 'm/s' },
];
const level = (v, w, c) => (w == null || v == null) ? 'ok' : v >= c ? 'crit' : v >= w ? 'warn' : 'ok';

/* ---------------- tabs ---------------- */
let activeTab = 'live';
$('#tabs').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  activeTab = b.dataset.tab;
  $$('#tabs button').forEach(x => x.classList.toggle('active', x === b));
  $$('.tab').forEach(x => x.classList.toggle('active', x.id === 'tab-' + activeTab));
  onTabShow(activeTab);
});
function onTabShow(t) {
  if (t === 'sensors') loadSensorCharts();
  if (t === 'alerts') loadAlerts();
  if (t === 'predict') loadPredictTrend();
  if (t === 'data') { loadGallery(); loadModels(); }
  if (t === 'settings') renderSettings();
}
setInterval(() => {
  if (activeTab === 'sensors') loadSensorCharts();
  if (activeTab === 'predict') loadPredictTrend();
  if (activeTab === 'data') { pollTraining(); }
}, 5000);

/* ---------------- websocket ---------------- */
let ws, lastData = null;
function connectWS() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`);
  ws.onmessage = e => { lastData = JSON.parse(e.data); handleLive(lastData); };
  ws.onclose = () => setTimeout(connectWS, 2000);
  ws.onopen = () => setInterval(() => ws.readyState === 1 && ws.send('ping'), 20000);
}
const recentAlerts = [];
function beep() {
  try {
    const a = new (window.AudioContext || window.webkitAudioContext)(), o = a.createOscillator(), g = a.createGain();
    o.connect(g); g.connect(a.destination); o.frequency.value = 880; g.gain.value = 0.08; o.start(); o.stop(a.currentTime + 0.25);
  } catch (_) { }
}
function setPill(id, text, cls) { const p = $(id); p.textContent = text; p.className = 'pill ' + cls; }

function handleLive(d) {
  const st = d.stream;
  setPill('#pill-stream', st.connected && !st.stalled ? `Stream: live ${st.fps} fps` : st.running ? 'Stream: connecting...' : 'Stream: offline',
    st.connected && !st.stalled ? 'ok' : st.running ? 'warn' : 'off');
  const feed = $('#feed');
  if (feed && st.connected && !st.stalled && (!feed.dataset.connected || feed.dataset.connected !== 'true')) {
    feed.src = '/video_feed?t=' + Date.now();
    feed.dataset.connected = 'true';
  } else if (feed && !st.connected) {
    feed.dataset.connected = 'false';
  }

  if (d.sensors) setPill('#pill-sensors', `Sensors: ${d.sensors.source}`, 'ok'); else setPill('#pill-sensors', 'Sensors: none', 'off');

  const h = d.health, col = { green: 'var(--green)', yellow: 'var(--yellow)', red: 'var(--red)' }[h.status];
  setPill('#pill-health', `Health: ${Math.round(h.score)}`, h.status === 'green' ? 'ok' : h.status === 'yellow' ? 'warn' : 'bad');
  $('#ring-text').textContent = Math.round(h.score);
  $('#ring-fg').style.strokeDashoffset = 326.7 * (1 - h.score / 100);
  $('#ring-fg').style.stroke = col;
  const hs = $('#health-status'); hs.textContent = h.status; hs.className = 'status-badge ' + h.status;
  const src = []; if (h.sources.vision) src.push('vision'); if (h.sources.sensors) src.push('sensors');
  $('#health-sources').textContent = src.length ? 'Based on: ' + src.join(' + ') : 'No data sources yet';

  if (d.lifespan) {
    const ls = d.lifespan;
    const mEl = $('#lifespan-main'); if (mEl) mEl.textContent = ls.text || '--';
    const dEl = $('#lifespan-days'); if (dEl) dEl.textContent = ls.days != null ? ls.days : '--';
    const moEl = $('#lifespan-months'); if (moEl) moEl.textContent = ls.months != null ? ls.months : '--';
    const yEl = $('#lifespan-years'); if (yEl) yEl.textContent = ls.years != null ? ls.years : '--';
    const cEl = $('#lifespan-condition');
    if (cEl) {
      cEl.textContent = ls.condition || 'Optimal Condition';
      cEl.className = 'sev ' + (h.score >= 75 ? 'info' : h.score >= 50 ? 'warning' : 'critical');
    }
    const rLs = $('#rul-lifespan'); if (rLs) rLs.textContent = ls.text || '--';
    const rDays = $('#rul-days'); if (rDays) rDays.textContent = `${ls.days} days`;
    const rMo = $('#rul-months'); if (rMo) rMo.textContent = `${ls.months} months`;
    const rYr = $('#rul-years'); if (rYr) rYr.textContent = `${ls.years} years`;
  }
  const manLoadInput = $('#input-manual-load');
  if (manLoadInput && document.activeElement !== manLoadInput && d.sensors && d.sensors.load != null) {
    manLoadInput.placeholder = Math.round(d.sensors.load);
  }

  $('#sensor-tiles').innerHTML = SENSORS.map(s => {
    const v = d.sensors ? d.sensors[s.key] : null;
    const lv = v == null ? '' : level(v, SETTINGS[s.warn], SETTINGS[s.crit]);
    return `<div class="tile ${lv}"><div class="l">${s.label}</div><div class="v">${v == null ? '--' : v.toFixed(1)} <span class="u">${s.unit}</span></div></div>`;
  }).join('');
  const v = d.vision;
  const tile = (l, val, u, lv) => `<div class="tile ${lv || ''}"><div class="l">${l}</div><div class="v">${val} <span class="u">${u}</span></div></div>`;
  $('#vision-tiles').innerHTML = v ? [
    tile('Defects', v.n_defects, '', v.n_defects ? 'crit' : 'ok'),
    tile('Belt offset', v.belt_found ? v.offset_pct.toFixed(1) : '--', '%', !v.belt_found ? '' : level(Math.abs(v.offset_pct), SETTINGS.misalign_warn_pct, SETTINGS.misalign_crit_pct)),
    tile('Dust', v.dust_pct.toFixed(0), '%', level(v.dust_pct, SETTINGS.dust_warn_pct, SETTINGS.dust_crit_pct)),
    tile('Sharpness', v.blur.toFixed(0), '', v.blur_ok ? 'ok' : 'warn'),
  ].join('') : '<span class="muted small">Connect a camera stream to see vision metrics.</span>';

  // SCADA Derived Telemetry & Throughput
  const curLoad = (d.sensors && d.sensors.load != null) ? d.sensors.load : 450;
  const curSpeed = (d.sensors && d.sensors.speed != null) ? d.sensors.speed : 2.4;
  const throughput = Math.round((curLoad * curSpeed * 3.6) / 10);
  const pTp = $('#pill-throughput'); if (pTp) pTp.textContent = `Throughput: ${throughput} T/h`;
  const flB = $('#flow-badge'); if (flB) flB.textContent = `Flow: ${throughput} T/h`;

  const pIl = $('#pill-interlock');
  const scadaLed = $('#scada-led');
  if (pIl) {
    if (h.status === 'red') {
      pIl.className = 'pill bad';
      pIl.innerHTML = '<span class="led-dot led-red"></span>Interlocks: TRIP WARNING';
      if (scadaLed) scadaLed.className = 'led-dot led-red';
    } else if (h.status === 'yellow') {
      pIl.className = 'pill warn';
      pIl.innerHTML = '<span class="led-dot led-amber"></span>Interlocks: Advisory';
      if (scadaLed) scadaLed.className = 'led-dot led-amber';
    } else {
      pIl.className = 'pill ok';
      pIl.innerHTML = '<span class="led-dot led-green"></span>Interlocks: Clear';
      if (scadaLed) scadaLed.className = 'led-dot led-green';
    }
  }

  // Structural Diagnostics & Integrity
  const nDef = v ? v.n_defects : 0;
  const dSplice = $('#diag-splice');
  if (dSplice) {
    if (nDef > 0) {
      dSplice.textContent = `${Math.max(55, 98 - nDef * 15)}% Review Needed`;
      dSplice.style.color = 'var(--yellow)';
    } else {
      dSplice.textContent = '98% Normal';
      dSplice.style.color = 'var(--green)';
    }
  }
  const dTrack = $('#diag-tracking');
  if (dTrack) {
    const off = v && v.belt_found ? v.offset_pct : 0;
    dTrack.textContent = `${off >= 0 ? '+' : ''}${off.toFixed(1)}% ${Math.abs(off) > 8 ? 'Critical Drift' : Math.abs(off) > 4 ? 'Mild Drift' : 'True Alignment'}`;
    dTrack.style.color = Math.abs(off) > 8 ? 'var(--red)' : Math.abs(off) > 4 ? 'var(--yellow)' : 'var(--green)';
  }
  const dCover = $('#diag-cover');
  if (dCover) {
    const dust = v ? v.dust_pct : 0;
    dCover.textContent = dust > 30 ? 'Grade 3 (Heavy Spillage)' : dust > 15 ? 'Grade 2 (Moderate Dust)' : 'Grade 1 (Nominal)';
    dCover.style.color = dust > 30 ? 'var(--red)' : dust > 15 ? 'var(--yellow)' : 'var(--green)';
  }
  const dPower = $('#diag-power');
  if (dPower) {
    const pwr = Math.round((curLoad / 900) * 100);
    dPower.textContent = `${pwr}% Nominal`;
    dPower.style.color = pwr > 85 ? 'var(--red)' : pwr > 70 ? 'var(--yellow)' : '#3b9eff';
  }

  // Maintenance Advisory
  const advText = $('#advisory-text');
  if (advText) {
    if (h.score < 50) {
      advText.textContent = 'Critical Action Required: High defect density or severe spillage detected. Schedule immediate maintenance window for splice audit & belt cleanout.';
      advText.style.color = 'var(--red)';
    } else if (h.score < 75) {
      advText.textContent = 'Preventative Advisory: Moderate wear / tracking drift observed. Schedule visual idler inspection and skirtboard rubber adjustment during next maintenance cycle.';
      advText.style.color = 'var(--yellow)';
    } else {
      advText.textContent = 'Operational condition optimal. Maintain scheduled idler lubrication intervals and standard belt tension inspection.';
      advText.style.color = 'var(--text)';
    }
  }

  const sim = d.simulator;
  $('#btn-sim').textContent = sim.enabled ? 'Disable simulator' : 'Enable simulator';
  $('#btn-fault').textContent = sim.inject ? 'Stop fault' : 'Inject fault';
  $('#sim-state').textContent = sim.enabled ? `Simulator running - fault level ${(sim.fault * 100).toFixed(0)}%` : 'Simulator off';

  if (d.new_alerts.length) {
    d.new_alerts.forEach(a => recentAlerts.unshift(a));
    if (d.new_alerts.some(a => a.severity === 'critical')) beep();
    renderLiveAlerts();
  }
  updateRisk(d.risk);
}
function alertRow(a, withAck) {
  return `<div class="al"><span class="sev ${a.severity}">${a.severity}</span><span class="muted small">${fmtTime(a.ts)}</span><span style="flex:1">${a.message}</span>${a.snapshot ? `<a href="${a.snapshot}" target="_blank"><img src="${a.snapshot}" style="height:34px;border-radius:4px"></a>` : ''}</div>`;
}
function renderLiveAlerts() {
  $('#live-alerts').innerHTML = recentAlerts.slice(0, 8).map(a => alertRow(a)).join('') || '<span class="muted small">No alerts yet.</span>';
}

/* ---------------- connect ---------------- */
$('#btn-connect').onclick = async () => {
  let url = $('#stream-url').value.trim();
  if (!url) {
    url = SETTINGS.stream_url || 'http://10.221.206.93:8080';
    $('#stream-url').value = url;
  }
  try {
    await api('/api/stream/connect', 'POST', { url });
    const feed = $('#feed');
    if (feed) {
      feed.dataset.connected = 'true';
      feed.src = '/video_feed?t=' + Date.now();
    }
  } catch (e) { alert(e.message); }
};
$('#btn-disconnect').onclick = () => {
  const feed = $('#feed');
  if (feed) feed.dataset.connected = 'false';
  api('/api/stream/disconnect', 'POST');
};

const feedImg = $('#feed');
if (feedImg) {
  feedImg.onerror = () => {
    setTimeout(() => {
      if (lastData && lastData.stream && lastData.stream.connected) {
        feedImg.src = '/video_feed?t=' + Date.now();
      }
    }, 1500);
  };
}

const btnSetLoad = $('#btn-set-load');
if (btnSetLoad) {
  btnSetLoad.onclick = async () => {
    const inp = $('#input-manual-load');
    const val = parseFloat(inp.value);
    const msg = $('#load-msg');
    if (isNaN(val) || val < 0) {
      if (msg) msg.textContent = 'Enter valid kg';
      return;
    }
    try {
      if (msg) msg.textContent = 'Updating...';
      await api('/api/sensors', 'POST', { load: val });
      if (msg) {
        msg.textContent = `Applied ${val} kg`;
        setTimeout(() => { msg.textContent = ''; }, 2500);
      }
    } catch (e) {
      if (msg) msg.textContent = 'Error: ' + e.message;
    }
  };
}

/* ---------------- Shift Report Modal ---------------- */
function populateReport() {
  const d = lastData || {};
  const h = d.health || { score: 100, status: 'green' };
  const ls = d.lifespan || { text: '--', condition: 'Optimal' };
  const s = d.sensors || {};
  const v = d.vision || {};
  const curLoad = s.load != null ? s.load : 450;
  const curSpeed = s.speed != null ? s.speed : 2.4;
  const curVib = s.vibration != null ? s.vibration : 2.0;
  const throughput = Math.round((curLoad * curSpeed * 3.6) / 10);

  const selAsset = $('#select-asset');
  const repLine = $('#rep-line');
  if (repLine && selAsset) repLine.textContent = selAsset.options[selAsset.selectedIndex].text;
  const repTime = $('#rep-time');
  if (repTime) repTime.textContent = new Date().toLocaleString();
  const repH = $('#rep-health');
  if (repH) {
    repH.textContent = `${Math.round(h.score)} / 100 (${h.status.toUpperCase()})`;
    repH.style.color = h.status === 'green' ? 'var(--green)' : h.status === 'yellow' ? 'var(--yellow)' : 'var(--red)';
  }
  const repLs = $('#rep-lifespan'); if (repLs) repLs.textContent = ls.text || '--';
  const repCond = $('#rep-condition'); if (repCond) repCond.textContent = ls.condition || 'Optimal';
  const repTp = $('#rep-throughput'); if (repTp) repTp.textContent = `${throughput} T/h`;
  const repL = $('#rep-load'); if (repL) repL.textContent = `${curLoad.toFixed(1)} kg`;
  const repSp = $('#rep-speed'); if (repSp) repSp.textContent = `${curSpeed.toFixed(1)} m/s`;
  const repVib = $('#rep-vib'); if (repVib) repVib.textContent = `${curVib.toFixed(1)} m/s²`;
  const repOff = $('#rep-offset'); if (repOff) repOff.textContent = v.belt_found ? `${v.offset_pct >= 0 ? '+' : ''}${v.offset_pct.toFixed(1)} %` : '--';
  const repDust = $('#rep-dust'); if (repDust) repDust.textContent = v.dust_pct != null ? `${v.dust_pct.toFixed(0)} %` : '--';
  const repDef = $('#rep-defects'); if (repDef) repDef.textContent = v.n_defects != null ? v.n_defects : 0;
  const repAdv = $('#rep-advisory');
  const advEl = $('#advisory-text');
  if (repAdv && advEl) repAdv.textContent = advEl.textContent;
}

const modRep = $('#modal-report');
const openRep = () => { populateReport(); if (modRep) modRep.classList.remove('hidden'); };
const closeRep = () => { if (modRep) modRep.classList.add('hidden'); };

const btnOpenRep = $('#btn-open-report'); if (btnOpenRep) btnOpenRep.onclick = openRep;
const btnWorkOrder = $('#btn-create-workorder'); if (btnWorkOrder) btnWorkOrder.onclick = openRep;
const btnCloseRep = $('#btn-close-report'); if (btnCloseRep) btnCloseRep.onclick = closeRep;
const btnDismissRep = $('#btn-dismiss-report'); if (btnDismissRep) btnDismissRep.onclick = closeRep;
const btnPrintRep = $('#btn-print-report'); if (btnPrintRep) btnPrintRep.onclick = () => window.print();

if (modRep) {
  modRep.addEventListener('click', e => {
    if (e.target === modRep) closeRep();
  });
}
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeRep();
});

const selAssetEl = $('#select-asset');
if (selAssetEl) {
  selAssetEl.onchange = () => {
    const repLine = $('#rep-line');
    if (repLine) repLine.textContent = selAssetEl.options[selAssetEl.selectedIndex].text;
  };
}

/* ---------------- sensor charts ---------------- */
const sensorCharts = {};
function mkChart(ctx, datasets, opts = {}) {
  return new Chart(ctx, {
    type: 'line', data: { labels: [], datasets }, options: {
      animation: false, responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8794a6' } } },
      scales: { x: { ticks: { color: '#8794a6', maxTicksLimit: 8 }, grid: { color: '#2a3441' } }, y: { ticks: { color: '#8794a6' }, grid: { color: '#2a3441' }, ...(opts.y || {}) } }
    }
  });
}
function initSensorCharts() {
  const box = $('#sensor-charts');
  SENSORS.forEach(s => {
    const wrap = document.createElement('div'); wrap.className = 'card'; wrap.style.height = '260px';
    wrap.innerHTML = `<h3>${s.label} (${s.unit})</h3><div style="height:200px"><canvas></canvas></div>`;
    box.appendChild(wrap);
    const ds = [{ label: s.label, data: [], borderColor: '#3b9eff', pointRadius: 0, borderWidth: 2, tension: .25 }];
    if (s.warn) { ds.push({ label: 'warn', data: [], borderColor: '#f1c40f', borderDash: [6, 6], pointRadius: 0, borderWidth: 1 }); ds.push({ label: 'crit', data: [], borderColor: '#ff4d4f', borderDash: [6, 6], pointRadius: 0, borderWidth: 1 }); }
    sensorCharts[s.key] = mkChart($('canvas', wrap), ds);
  });
}
async function loadSensorCharts() {
  try {
    const h = await api('/api/sensors/history?minutes=' + $('#range').value);
    const labels = h.ts.map(fmtTime);
    SENSORS.forEach(s => {
      const c = sensorCharts[s.key]; c.data.labels = labels; c.data.datasets[0].data = h[s.key];
      if (s.warn) { c.data.datasets[1].data = labels.map(() => SETTINGS[s.warn]); c.data.datasets[2].data = labels.map(() => SETTINGS[s.crit]); }
      c.update();
    });
  } catch (_) { }
}
$('#range').onchange = loadSensorCharts;

/* ---------------- alerts table ---------------- */
async function loadAlerts() {
  const q = new URLSearchParams({ limit: 200, severity: $('#f-sev').value, acknowledged: $('#f-ack').value });
  const rows = await api('/api/alerts?' + q);
  $('#alert-table tbody').innerHTML = rows.map(a => `<tr>
    <td>${fmtDT(a.ts)}</td><td><span class="sev ${a.severity}">${a.severity}</span></td><td>${a.kind}</td><td>${a.message}</td>
    <td>${a.snapshot ? `<a href="${a.snapshot}" target="_blank"><img src="${a.snapshot}"></a>` : ''}</td>
    <td>${a.acknowledged ? '<span class="muted small">acknowledged</span>' : `<button data-ack="${a.id}">Acknowledge</button>`}</td></tr>`).join('')
    || '<tr><td colspan="6" class="muted">No alerts.</td></tr>';
}
$('#alert-table').addEventListener('click', async e => {
  const id = e.target.dataset.ack; if (!id) return;
  await api(`/api/alerts/${id}/ack`, 'POST'); loadAlerts();
});
$('#btn-refresh-alerts').onclick = loadAlerts; $('#f-sev').onchange = loadAlerts; $('#f-ack').onchange = loadAlerts;

/* ---------------- prediction ---------------- */
let predChart;
function updateRisk(r) {
  const p = r && r.risk_probability;
  const fg = $('#g-fg');
  if (p == null) { $('#g-text').textContent = r && r.training ? '...' : '--'; fg.style.strokeDashoffset = 251.3; }
  else {
    $('#g-text').textContent = Math.round(p * 100) + '%';
    fg.style.strokeDashoffset = 251.3 * (1 - p); fg.style.stroke = p > .7 ? 'var(--red)' : p > .35 ? 'var(--yellow)' : 'var(--green)';
  }
  $('#risk-sub').textContent = r && r.horizon_min ? `Probability of failure within ${r.horizon_min} min` : '';
  $('#rul').textContent = r && r.rul_min != null ? `${r.rul_min.toFixed(1)} min` : '--';
  $('#anomaly').textContent = !r ? '--' : r.anomaly == null ? `Learning baseline ${r.baseline_have}/${r.baseline_n}` : r.anomaly ? 'ANOMALY' : 'Normal';
  const m = r && r.model || {};
  $('#model-info').innerHTML = m.trained_at ? `Trained: ${m.trained_at}<br>Source: <b>${m.source}</b> (${m.n_rows} rows)<br>Hold-out accuracy ${m.accuracy}, F1 ${m.f1}, RUL error ~${m.rul_mae_min} min` : (r && r.training ? 'Training...' : 'No model yet');
}
async function loadPredictTrend() {
  try {
    const h = await api('/api/predict/history?minutes=60');
    if (!predChart) predChart = mkChart($('#chart-predict'), [
      { label: 'Health score', data: [], borderColor: '#2ecc71', pointRadius: 0, yAxisID: 'y' },
      { label: 'Failure risk %', data: [], borderColor: '#ff4d4f', pointRadius: 0, yAxisID: 'y' }], { y: { min: 0, max: 100 } });
    predChart.data.labels = h.ts.map(fmtTime);
    predChart.data.datasets[0].data = h.health; predChart.data.datasets[1].data = h.risk.map(x => x == null ? null : x * 100);
    predChart.update();
  } catch (_) { }
}
$('#btn-retrain').onclick = async () => { try { await api('/api/predict/retrain', 'POST'); } catch (e) { alert(e.message); } };

/* ---------------- dataset & training ---------------- */
async function loadGallery() {
  const d = await api('/api/dataset/images');
  $('#gallery').innerHTML = d.images.map(i => `<div class="g"><img src="${i.url}" loading="lazy">${i.labeled ? '<span class="lab">labelled</span>' : ''}<button class="x" data-del="${i.name}">✕</button></div>`).join('') || '<span class="muted small">No images yet. Connect a stream and capture frames.</span>';
  $('#data-count').textContent = `${d.images.length} images captured, ${d.labeled} labelled (need ≥ 10 labelled to train from captures).`;
  const sel = $('#t-source'), cur = sel.value;
  sel.innerHTML = '<option value="labeled">Labelled captures</option>' + d.uploads.map(u => `<option value="${u}">Upload: ${u}</option>`).join('');
  sel.value = [...sel.options].some(o => o.value === cur) ? cur : 'labeled';
}
$('#gallery').addEventListener('click', async e => { const n = e.target.dataset.del; if (n) { await api('/api/dataset/images/' + n, 'DELETE'); loadGallery(); } });
$('#btn-capture').onclick = async () => { try { await api('/api/dataset/capture', 'POST'); loadGallery(); } catch (e) { alert(e.message); } };
$('#btn-auto').onclick = async () => { await api('/api/settings', 'POST', { auto_capture_sec: +$('#auto-sec').value || 0 }); SETTINGS.auto_capture_sec = +$('#auto-sec').value || 0; };
$('#zip-file').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append('file', f); $('#upload-msg').textContent = 'Uploading...';
  const r = await fetch('/api/dataset/upload', { method: 'POST', body: fd });
  const j = await r.json(); $('#upload-msg').textContent = r.ok ? `Imported "${j.name}": ${j.images} images` : 'Error: ' + j.detail; loadGallery();
};
let trainChart;
async function pollTraining() {
  const s = await api('/api/train/status');
  $('#t-bar').style.width = s.epochs ? (100 * s.epoch / s.epochs) + '%' : '0';
  $('#t-msg').textContent = `${s.state}${s.message ? ' - ' + s.message : ''}`;
  if (!trainChart) trainChart = mkChart($('#chart-train'), [
    { label: 'val box loss', data: [], borderColor: '#ff9f1c', pointRadius: 2 }, { label: 'mAP50', data: [], borderColor: '#2ecc71', pointRadius: 2 }]);
  trainChart.data.labels = s.history.map(h => h.epoch);
  trainChart.data.datasets[0].data = s.history.map(h => h.box_loss); trainChart.data.datasets[1].data = s.history.map(h => h.map50);
  trainChart.update();
  if (s.state === 'done') loadModels();
}
$('#btn-train').onclick = async () => {
  try { await api('/api/train/start', 'POST', { base_model: $('#t-base').value, epochs: +$('#t-epochs').value, imgsz: +$('#t-imgsz').value, batch: +$('#t-batch').value, data_source: $('#t-source').value }); pollTraining(); }
  catch (e) { alert(e.message); }
};
async function loadModels() {
  const d = await api('/api/models');
  $('#model-list').innerHTML = (d.models.map(m => `<div class="kv"><span><b>${m.version}</b> ${m.active ? '<span class="sev info">active</span>' : ''}<br>mAP50 ${m.metrics && m.metrics.map50 != null ? (+m.metrics.map50).toFixed(3) : '--'} · ${m.trained_at || ''}</span>${m.active ? '' : `<button data-deploy="${m.version}">Deploy to live feed</button>`}</div>`).join('')
    || 'No trained models yet.') + (d.error ? `<div class="sev warning">${d.error}</div>` : '') + (d.models.some(m => m.active) ? '<div style="margin-top:8px"><button id="btn-unload">Unload active model</button></div>' : '');
}
$('#model-list').addEventListener('click', async e => {
  if (e.target.dataset.deploy) { try { await api('/api/model/deploy', 'POST', { version: e.target.dataset.deploy }); } catch (err) { alert(err.message); } loadModels(); }
  if (e.target.id === 'btn-unload') { await api('/api/model/unload', 'POST'); loadModels(); }
});

/* ---------------- settings ---------------- */
const FIELDS = [
  ['belt_orientation', 'Belt orientation', 'select:vertical,horizontal'],
  ['conf_threshold', 'Detection confidence', 'number:0.05'], ['resize_width', 'Processing width (px)', 'number:10'],
  ['detect_every_n', 'Run YOLO every N frames', 'number:1'], ['blur_min', 'Min sharpness', 'number:5'],
  ['misalign_warn_pct', 'Misalignment warn %', 'number:1'], ['misalign_crit_pct', 'Misalignment crit %', 'number:1'],
  ['dust_warn_pct', 'Dust warn %', 'number:1'], ['dust_crit_pct', 'Dust crit %', 'number:1'],
  ['vib_warn', 'Vibration warn', 'number:0.1'], ['vib_crit', 'Vibration crit', 'number:0.1'],
  ['load_warn', 'Load warn', 'number:10'], ['load_crit', 'Load crit', 'number:10'],
  ['sim_ramp_s', 'Simulated fault ramp (s)', 'number:10'],
];
function renderSettings() {
  const f = $('#settings-form');
  f.innerHTML = FIELDS.map(([k, l, t]) => {
    if (t.startsWith('select:')) return `<label>${l}<select data-k="${k}">${t.slice(7).split(',').map(o => `<option ${SETTINGS[k] === o ? 'selected' : ''}>${o}</option>`).join('')}</select></label>`;
    return `<label>${l}<input data-k="${k}" type="number" step="${t.slice(7)}" value="${SETTINGS[k] ?? ''}"></label>`;
  }).join('') + `<label class="wide">Belt region of interest (normalised x, y, w, h - leave empty for full frame)
    <span style="display:flex;gap:6px">${[0, 1, 2, 3].map(i => `<input data-roi="${i}" type="number" step="0.05" min="0" max="1" style="width:25%" value="${SETTINGS.roi ? SETTINGS.roi[i] : ''}">`).join('')}</span></label>`;
}
$('#btn-save').onclick = async () => {
  const patch = {};
  $$('#settings-form [data-k]').forEach(el => { patch[el.dataset.k] = el.tagName === 'SELECT' ? el.value : parseFloat(el.value); });
  ['resize_width', 'detect_every_n', 'sim_ramp_s'].forEach(k => patch[k] = Math.round(patch[k]));
  const roi = $$('#settings-form [data-roi]').map(e => e.value);
  patch.roi = roi.every(v => v !== '') ? roi.map(Number) : null;
  try { SETTINGS = await api('/api/settings', 'POST', patch); $('#save-msg').textContent = 'Saved ✓'; }
  catch (e) { $('#save-msg').textContent = 'Error: ' + e.message; }
  setTimeout(() => $('#save-msg').textContent = '', 3000);
};
$('#btn-sim').onclick = () => api('/api/simulator/toggle', 'POST', {});
$('#btn-fault').onclick = () => api('/api/simulator/fault', 'POST', {});

/* ---------------- init ---------------- */
(async function init() {
  try { SETTINGS = await api('/api/settings'); } catch (_) { SETTINGS = {}; }
  $('#stream-url').value = SETTINGS.stream_url || 'http://10.221.206.93:8080';
  $('#auto-sec').value = SETTINGS.auto_capture_sec || 0;
  initSensorCharts();
  try {
    const recent = await api('/api/alerts?limit=8'); recent.forEach(a => recentAlerts.push(a)); renderLiveAlerts();
  } catch (_) { }
  connectWS();
})();
