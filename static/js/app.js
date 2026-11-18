/**
 * Aegis — Minimalist AI Voice Sentinel Client
 * - Liquid 60 FPS Fluid AI Orb + Smooth Mirror Voice Equalizer
 * - Direct Browser WebAudio Streaming (Earphones / Bluetooth / USB) + System Mic Option
 * - Parallel Multilingual Speech Captions & Keyword Detection
 * - Clean 3-Tab Views (Live | Test Audio | Activity) + Settings Drawer
 * - Strict CSRF headers & XSS-safe DOM manipulation
 */

(() => {
  "use strict";

  const CSRF = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const TARGET_SR = 16000;
  const WINDOW_SEC = 2.0;

  const state = {
    armed: false,
    calibrating: false,
    sourceMode: "browser", // "browser" | "server"
    uiState: "idle", // "idle" | "listening" | "elevated" | "emergency"
    threatScore: 0,
    sensitivity: "balanced",
    language: "en-US",
    dryRun: true,
    saveEvidence: false,
    cancelWindow: 5,
    cooldown: 25,
    keywords: [],
    serverDevices: [],
    browserDevices: [],
    incidents: [],
    pendingAlert: null,
    countdownTimer: null,
    captionTimer: null,
    inFlight: false,

    // WebAudio
    audioCtx: null,
    stream: null,
    analyser: null,
    processor: null,
    ring: new Float32Array(TARGET_SR * WINDOW_SEC),
    ringFilled: 0,
    chunkTimer: null,
    freq: new Uint8Array(256),
    liveRms: 0,
    smoothRms: 0,

    // Speech API
    speechRec: null,

    // Animation
    t: 0,
    motes: Array.from({ length: 28 }, () => ({
      a: Math.random() * Math.PI * 2,
      r: 0.95 + Math.random() * 0.42,
      s: (0.004 + Math.random() * 0.008) * (Math.random() < 0.5 ? 1 : -1),
      sz: 1.2 + Math.random() * 2.0,
      op: 0.25 + Math.random() * 0.55,
    })),
  };

  // DOM References
  const $ = (id) => document.getElementById(id);
  const el = {
    body: document.body,
    alertBanner: $("alertBanner"),
    alertTitle: $("alertTitle"),
    alertDetail: $("alertDetail"),
    alertTimer: $("alertTimer"),
    alertBarFill: $("alertBarFill"),
    btnCancelAlert: $("btnCancelAlert"),

    mainNavTabs: $("mainNavTabs"),
    btnSafeMode: $("btnSafeMode"),
    safeDot: $("safeDot"),
    safeText: $("safeText"),
    btnSettings: $("btnSettings"),
    btnSos: $("btnSos"),

    orbCanvas: $("orbCanvas"),
    eqCanvas: $("eqCanvas"),
    statusHeading: $("statusHeading"),
    statusSubtitle: $("statusSubtitle"),
    btnListen: $("btnListen"),
    listenLabel: $("listenLabel"),
    btnCalibrate: $("btnCalibrate"),
    calibrateLabel: $("calibrateLabel"),
    micSelect: $("micSelect"),
    btnRefreshMics: $("btnRefreshMics"),
    liveCaption: $("liveCaption"),

    dropZone: $("dropZone"),
    fileInput: $("fileInput"),
    dropTitle: $("dropTitle"),
    labResult: $("labResult"),
    labFileName: $("labFileName"),
    labMeta: $("labMeta"),
    labBadge: $("labBadge"),
    labScoreText: $("labScoreText"),
    labScoreFill: $("labScoreFill"),
    labTimeline: $("labTimeline"),

    modelSubtitle: $("modelSubtitle"),
    overallAucPill: $("overallAucPill"),
    actCombined: $("actCombined"),
    actCombinedBar: $("actCombinedBar"),
    actCnn: $("actCnn"),
    actAc: $("actAc"),
    actAgc: $("actAgc"),
    benchmarkBars: $("benchmarkBars"),
    logTabs: $("logTabs"),
    incidentList: $("incidentList"),
    eventList: $("eventList"),

    drawerOverlay: $("drawerOverlay"),
    settingsDrawer: $("settingsDrawer"),
    btnCloseSettings: $("btnCloseSettings"),
    sensitivitySeg: $("sensitivitySeg"),
    langSelect: $("langSelect"),
    sourceModeSeg: $("sourceModeSeg"),
    setDryRun: $("setDryRun"),
    setSaveEvidence: $("setSaveEvidence"),
    setCancelWin: $("setCancelWin"),
    cancelWinText: $("cancelWinText"),
    setCooldown: $("setCooldown"),
    cooldownText: $("cooldownText"),
    emailBadge: $("emailBadge"),
    smsBadge: $("smsBadge"),
    btnTestAlert: $("btnTestAlert"),
    setKeywords: $("setKeywords"),
    btnSaveSettings: $("btnSaveSettings"),

    toasts: $("toasts"),
  };

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  async function apiPost(url, body = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": CSRF },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function toast(msg, kind = "info") {
    const div = document.createElement("div");
    div.className = `toast ${kind}`;
    div.textContent = msg;
    el.toasts.appendChild(div);
    setTimeout(() => {
      div.style.opacity = "0";
      div.style.transition = "opacity 0.25s ease";
      setTimeout(() => div.remove(), 260);
    }, 3000);
  }

  function float32ToBase64(f32) {
    const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
  }

  function downsampleLinear(buf, inSr, outSr) {
    if (inSr === outSr) return buf;
    const ratio = inSr / outSr;
    const len = Math.max(1, Math.floor(buf.length / ratio));
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = buf[idx] || 0;
      const b = buf[Math.min(idx + 1, buf.length - 1)] || a;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Bootstrap & SSE
  // ------------------------------------------------------------------
  async function bootstrap() {
    try {
      const res = await fetch("/api/bootstrap");
      const data = await res.json();
      const det = data.detector || {};
      const al = data.alerts || {};

      state.sensitivity = det.sensitivity || "balanced";
      state.language = det.language || "en-US";
      state.keywords = det.keywords || [];
      state.serverDevices = data.server_devices || [];
      state.dryRun = al.dry_run !== false;
      state.saveEvidence = Boolean(al.save_evidence);
      state.cancelWindow = al.cancel_window_seconds ?? 5;
      state.cooldown = al.cooldown_seconds ?? 25;
      state.incidents = al.incidents || [];

      if (det.model_version) {
        el.modelSubtitle.textContent = `Custom Local AI (${det.model_version}) • No Pretrained Models`;
      }

      syncSettingsUI(al);
      renderBenchmarks(det.overall_oof_metrics, det.mic_benchmarks);
      renderIncidents();
      el.eventList.replaceChildren();
      (data.logs || []).forEach((item) => addEventRow(item, false));

      if (al.pending_alert) showPendingBanner(al.pending_alert);

      await refreshMics();
      connectSSE();
    } catch (err) {
      toast(`Connection issue: ${err.message}`, "danger");
    }
  }

  function connectSSE() {
    const es = new EventSource("/api/stream");
    es.addEventListener("telemetry", (ev) => applyTelemetry(JSON.parse(ev.data)));
    es.addEventListener("log", (ev) => addEventRow(JSON.parse(ev.data), true));
    es.addEventListener("alert_pending", (ev) => {
      const inc = JSON.parse(ev.data);
      showPendingBanner(inc);
      setUIState("emergency", "Distress detected", inc.detail || "Tap Cancel if this was a false alarm.");
    });
    es.addEventListener("alert_dispatched", (ev) => {
      const inc = JSON.parse(ev.data);
      hidePendingBanner();
      state.incidents.unshift(inc);
      state.incidents = state.incidents.slice(0, 40);
      renderIncidents();
      toast(
        inc.dry_run ? `Safe Mode: Alert ${inc.id} simulated` : `Emergency Alert ${inc.id} sent!`,
        inc.dry_run ? "info" : "danger"
      );
    });
    es.addEventListener("alert_cancelled", (ev) => {
      const inc = JSON.parse(ev.data);
      hidePendingBanner();
      state.incidents.unshift(inc);
      renderIncidents();
      setUIState(
        state.armed ? "listening" : "idle",
        state.armed ? "Listening — All clear" : "Ready to listen",
        state.armed ? "Monitoring voice and emergency keywords." : "Choose your earphones or mic below and tap Start."
      );
      toast("Alert cancelled.", "ok");
    });
  }

  // ------------------------------------------------------------------
  // Microphone Discovery & Audio Capture
  // ------------------------------------------------------------------
  async function refreshMics() {
    if (state.sourceMode === "browser") {
      try {
        const devs = (await navigator.mediaDevices?.enumerateDevices?.()) || [];
        state.browserDevices = devs.filter((d) => d.kind === "audioinput");
      } catch {
        state.browserDevices = [];
      }
      populateMics(
        state.browserDevices.map((d, i) => ({
          value: d.deviceId,
          label: d.label || `Microphone ${i + 1}`,
        })),
        "Default Microphone / Earphones"
      );
    } else {
      populateMics(
        state.serverDevices.map((d) => ({
          value: String(d.id),
          label: `${d.name}${d.is_default ? " (Default)" : ""}`,
        })),
        "Default System Microphone"
      );
    }
  }

  function populateMics(items, fallback) {
    const prev = el.micSelect.value;
    el.micSelect.replaceChildren();
    const def = document.createElement("option");
    def.value = "";
    def.textContent = fallback;
    el.micSelect.appendChild(def);
    items.forEach((it) => {
      if (!it.value && it.value !== "0") return;
      const o = document.createElement("option");
      o.value = it.value;
      o.textContent = it.label;
      el.micSelect.appendChild(o);
    });
    if ([...el.micSelect.options].some((o) => o.value === prev)) el.micSelect.value = prev;
  }

  async function toggleListening() {
    if (state.armed) await stopListening();
    else await startListening();
  }

  async function startListening() {
    if (state.sourceMode === "server") {
      try {
        const dev = el.micSelect.value;
        await apiPost("/api/server-mic/start", { device_id: dev !== "" ? Number(dev) : null });
        state.armed = true;
        updateListenButton();
        setUIState("listening", "Listening — All clear", "Monitoring your system microphone.");
      } catch (err) {
        toast(`Could not start mic: ${err.message}`, "danger");
      }
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      toast("Browser mic unavailable. Switch to System OS Mic in Settings.", "danger");
      return;
    }

    try {
      const devId = el.micSelect.value;
      const constraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };
      if (devId) constraints.deviceId = { ideal: devId };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }

      state.stream = stream;
      const AC = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new AC();
      if (state.audioCtx.state === "suspended") await state.audioCtx.resume();

      const src = state.audioCtx.createMediaStreamSource(stream);
      state.analyser = state.audioCtx.createAnalyser();
      state.analyser.fftSize = 512;
      state.analyser.smoothingTimeConstant = 0.8;
      state.freq = new Uint8Array(state.analyser.frequencyBinCount);

      state.processor = state.audioCtx.createScriptProcessor(4096, 1, 1);
      state.ring = new Float32Array(TARGET_SR * WINDOW_SEC);
      state.ringFilled = 0;

      const inSr = state.audioCtx.sampleRate;
      state.processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        let sq = 0;
        for (let i = 0; i < input.length; i++) sq += input[i] * input[i];
        state.liveRms = Math.sqrt(sq / input.length);

        const down = downsampleLinear(input, inSr, TARGET_SR);
        const n = down.length;
        if (n >= state.ring.length) {
          state.ring.set(down.subarray(n - state.ring.length));
        } else {
          state.ring.copyWithin(0, n);
          state.ring.set(down, state.ring.length - n);
        }
        state.ringFilled = Math.min(state.ring.length, state.ringFilled + n);
      };

      const mute = state.audioCtx.createGain();
      mute.gain.value = 0;
      src.connect(state.analyser);
      state.analyser.connect(state.processor);
      state.processor.connect(mute);
      mute.connect(state.audioCtx.destination);

      await refreshMics();

      state.armed = true;
      updateListenButton();
      setUIState("listening", "Listening — All clear", "Monitoring acoustic distress and emergency keywords.");
      startSpeech();
      state.chunkTimer = setInterval(pushAudioChunk, 900);
    } catch (err) {
      toast(`Microphone access failed: ${err.message}`, "danger");
    }
  }

  async function stopListening() {
    state.armed = false;
    stopSpeech();
    if (state.chunkTimer) { clearInterval(state.chunkTimer); state.chunkTimer = null; }
    try { state.processor?.disconnect(); } catch {}
    try { state.analyser?.disconnect(); } catch {}
    try { await state.audioCtx?.close(); } catch {}
    state.stream?.getTracks().forEach((t) => t.stop());

    state.processor = null;
    state.analyser = null;
    state.audioCtx = null;
    state.stream = null;
    state.liveRms = 0;
    state.threatScore = 0;
    el.liveCaption.classList.add("hidden");

    try { await apiPost("/api/session/reset"); } catch {}

    updateListenButton();
    setUIState("idle", "Ready to listen", "Choose your earphones or mic below and tap Start.");
  }

  async function pushAudioChunk() {
    if (!state.armed || state.inFlight || state.ringFilled < TARGET_SR * 1.1) return;
    state.inFlight = true;
    try {
      const snap = new Float32Array(state.ring);
      const data = await apiPost("/api/analyze-chunk", {
        sample_rate: TARGET_SR,
        format: "f32",
        pcm_b64: float32ToBase64(snap),
      });
      applyTelemetry(data);
    } catch {} finally {
      state.inFlight = false;
    }
  }

  async function calibrateMic() {
    if (state.calibrating) return;
    state.calibrating = true;
    el.btnCalibrate.disabled = true;
    el.calibrateLabel.textContent = "Stay quiet (2s)...";

    try {
      let pcm;
      if (state.armed && state.sourceMode === "browser" && state.ringFilled >= TARGET_SR) {
        await new Promise((r) => setTimeout(r, 1800));
        pcm = new Float32Array(state.ring);
      } else {
        const devId = el.micSelect.value;
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: devId ? { deviceId: { ideal: devId }, autoGainControl: false } : { autoGainControl: false },
        });
        const AC = window.AudioContext || window.webkitAudioContext;
        const ctx = new AC();
        const src = ctx.createMediaStreamSource(stream);
        const proc = ctx.createScriptProcessor(4096, 1, 1);
        const chunks = [];
        const mute = ctx.createGain();
        mute.gain.value = 0;
        proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        src.connect(proc);
        proc.connect(mute);
        mute.connect(ctx.destination);

        await new Promise((r) => setTimeout(r, 1800));

        proc.disconnect();
        src.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        const sr = ctx.sampleRate;
        await ctx.close();

        const len = chunks.reduce((a, c) => a + c.length, 0);
        const merged = new Float32Array(len);
        let off = 0;
        for (const c of chunks) { merged.set(c, off); off += c.length; }
        pcm = downsampleLinear(merged, sr, TARGET_SR);
      }

      await apiPost("/api/calibrate", {
        sample_rate: TARGET_SR,
        format: "f32",
        pcm_b64: float32ToBase64(pcm),
      });
      el.calibrateLabel.textContent = "Calibrated ✓";
      toast("Microphone adapted to your room & earphones.", "ok");
      setTimeout(() => { el.calibrateLabel.textContent = "Calibrate Mic"; }, 2500);
    } catch (err) {
      el.calibrateLabel.textContent = "Calibrate Mic";
      toast(`Calibration failed: ${err.message}`, "danger");
    } finally {
      state.calibrating = false;
      el.btnCalibrate.disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // Parallel Speech Captions & Keyword Matching
  // ------------------------------------------------------------------
  function startSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    stopSpeech();

    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = state.language || "en-US";

    rec.onresult = (ev) => {
      let interim = "";
      let fin = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const txt = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) fin += txt + " ";
        else interim += txt;
      }
      const text = (fin || interim).trim();
      if (!text) return;

      showCaption(text, null);
      apiPost("/api/transcript", { text, is_final: Boolean(fin) })
        .then((res) => {
          if (res.matched && res.match) showCaption(text, res.match.keyword);
        })
        .catch(() => {});
    };

    rec.onend = () => {
      if (state.armed && state.sourceMode === "browser") {
        setTimeout(() => {
          if (state.armed && state.sourceMode === "browser") {
            try { rec.start(); } catch {}
          }
        }, 450);
      }
    };

    state.speechRec = rec;
    try { rec.start(); } catch {}
  }

  function stopSpeech() {
    if (state.speechRec) {
      try { state.speechRec.onend = null; state.speechRec.stop(); } catch {}
      state.speechRec = null;
    }
  }

  function showCaption(text, kw) {
    el.liveCaption.classList.remove("hidden");
    el.liveCaption.replaceChildren();
    if (!kw) {
      el.liveCaption.textContent = `"${text}"`;
    } else {
      const idx = text.toLowerCase().indexOf(kw.toLowerCase());
      if (idx === -1) {
        const b = document.createElement("span");
        b.className = "kw-hit";
        b.textContent = kw.toUpperCase();
        el.liveCaption.append(b, document.createTextNode(` — "${text}"`));
      } else {
        el.liveCaption.append(
          document.createTextNode(`"${text.slice(0, idx)}`),
          Object.assign(document.createElement("span"), { className: "kw-hit", textContent: text.slice(idx, idx + kw.length) }),
          document.createTextNode(`${text.slice(idx + kw.length)}"`)
        );
      }
    }
    clearTimeout(state.captionTimer);
    state.captionTimer = setTimeout(() => {
      if (!state.pendingAlert) el.liveCaption.classList.add("hidden");
    }, 5500);
  }

  // ------------------------------------------------------------------
  // Telemetry & State Updates
  // ------------------------------------------------------------------
  function applyTelemetry(d) {
    state.threatScore = d.smoothed_score ?? 0;
    if (state.sourceMode === "server") state.liveRms = d.raw_rms ?? 0;

    // Update Activity view insights silently in background
    const pct = Math.round(state.threatScore * 100);
    el.actCombined.textContent = `${pct}%`;
    el.actCombinedBar.style.width = `${pct}%`;
    el.actCombinedBar.classList.toggle("danger", Boolean(d.distress_detected));
    el.actCnn.textContent = `${Math.round((d.cnn_score ?? 0) * 100)}%`;
    el.actAc.textContent = `${Math.round((d.acoustic_score ?? 0) * 100)}%`;
    el.actAgc.textContent = `+${(d.agc_gain_db ?? 0).toFixed(1)} dB`;

    if (!state.armed || state.pendingAlert) return;

    if (d.threat_level === "critical") {
      setUIState("emergency", "Distress detected", "Emergency vocal pattern recognized.");
    } else if (d.threat_level === "elevated") {
      setUIState("elevated", "Elevated vocal tension", "Listening closely to confirm...");
    } else {
      setUIState("listening", "Listening — All clear", "Monitoring acoustic distress and emergency keywords.");
    }
  }

  function setUIState(mode, heading, subtitle) {
    state.uiState = mode;
    el.body.setAttribute("data-state", mode);
    if (heading) el.statusHeading.textContent = heading;
    if (subtitle) el.statusSubtitle.textContent = subtitle;
  }

  function updateListenButton() {
    el.btnListen.classList.toggle("listening", state.armed);
    el.listenLabel.textContent = state.armed ? "Stop Listening" : "Start Listening";
  }

  function syncSettingsUI(al) {
    el.safeDot.className = `dot ${state.dryRun ? "warn" : "danger"}`;
    el.safeText.textContent = state.dryRun ? "Safe Mode" : "Live Alerts";

    el.setDryRun.checked = state.dryRun;
    el.setSaveEvidence.checked = state.saveEvidence;
    el.setCancelWin.value = state.cancelWindow;
    el.cancelWinText.textContent = `${state.cancelWindow}s`;
    el.setCooldown.value = state.cooldown;
    el.cooldownText.textContent = `${state.cooldown}s`;
    el.langSelect.value = state.language;
    el.setKeywords.value = state.keywords.join(", ");

    el.sensitivitySeg.querySelectorAll(".seg-opt").forEach((b) => {
      b.classList.toggle("active", b.dataset.sens === state.sensitivity);
    });
    el.sourceModeSeg.querySelectorAll(".seg-opt").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === state.sourceMode);
    });

    if (al) {
      el.emailBadge.textContent = al.email_configured ? "Email: Ready" : "Email: Not Set";
      el.smsBadge.textContent = al.sms_configured ? "SMS: Ready" : "SMS: Not Set";
    }
  }

  function renderBenchmarks(overall, mics) {
    if (!mics || !Object.keys(mics).length) return;
    if (overall?.roc_auc) el.overallAucPill.textContent = `AUC ${(overall.roc_auc * 100).toFixed(1)}%`;

    const names = {
      clean: "Clean / Studio Mic",
      wired_earbud: "Wired Earbud Mic",
      bluetooth_hfp: "Bluetooth Earphone (HFP)",
      low_gain_distant: "Quiet / Distant Room",
      noisy_street: "Noisy Environment",
    };

    el.benchmarkBars.replaceChildren();
    Object.entries(mics).forEach(([k, m]) => {
      const row = document.createElement("div");
      row.className = "bench-row";

      const top = document.createElement("div");
      top.className = "bench-top";
      top.append(
        Object.assign(document.createElement("span"), { textContent: names[k] || k }),
        Object.assign(document.createElement("span"), { className: "mono muted", textContent: `${(m.roc_auc * 100).toFixed(1)}% AUC` })
      );

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${Math.round(m.roc_auc * 100)}%`;
      track.appendChild(fill);

      row.append(top, track);
      el.benchmarkBars.appendChild(row);
    });
  }

  function renderIncidents() {
    el.incidentList.replaceChildren();
    if (!state.incidents.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No alerts triggered yet.";
      el.incidentList.appendChild(empty);
      return;
    }
    state.incidents.forEach((inc) => {
      const item = document.createElement("div");
      item.className = `feed-item ${inc.status === "cancelled" ? "success" : "alert"}`;
      item.append(
        Object.assign(document.createElement("span"), {
          className: "feed-meta",
          textContent: `${inc.id} • ${inc.dry_run ? "Safe Mode" : "Live"} • ${inc.status.toUpperCase()}`,
        }),
        Object.assign(document.createElement("span"), { textContent: inc.detail })
      );
      el.incidentList.appendChild(item);
    });
  }

  function addEventRow(entry, prepend = true) {
    const item = document.createElement("div");
    item.className = `feed-item ${entry.level || "info"}`;
    item.append(
      Object.assign(document.createElement("span"), { className: "feed-meta", textContent: entry.time || "" }),
      Object.assign(document.createElement("span"), { textContent: entry.message || "" })
    );
    if (prepend && el.eventList.firstChild) el.eventList.insertBefore(item, el.eventList.firstChild);
    else el.eventList.appendChild(item);
    while (el.eventList.children.length > 60) el.eventList.lastChild.remove();
  }

  // ------------------------------------------------------------------
  // Emergency Countdown Banner
  // ------------------------------------------------------------------
  function showPendingBanner(inc) {
    state.pendingAlert = inc;
    const total = Math.max(1, inc.seconds_remaining || state.cancelWindow || 5);
    const deadline = Date.now() + total * 1000;

    el.alertTitle.textContent = inc.dry_run ? "Distress Detected (Safe Mode)" : "Emergency Detected — Sending Alert";
    el.alertDetail.textContent = inc.detail || "";
    el.alertBanner.classList.remove("hidden");

    clearInterval(state.countdownTimer);
    state.countdownTimer = setInterval(() => {
      const rem = Math.max(0, (deadline - Date.now()) / 1000);
      el.alertTimer.textContent = `${rem.toFixed(1)}s`;
      el.alertBarFill.style.width = `${(rem / total) * 100}%`;
      if (rem <= 0) clearInterval(state.countdownTimer);
    }, 80);
  }

  function hidePendingBanner() {
    state.pendingAlert = null;
    clearInterval(state.countdownTimer);
    el.alertBanner.classList.add("hidden");
  }

  // ------------------------------------------------------------------
  // Audio File Test Lab
  // ------------------------------------------------------------------
  async function analyzeFile(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      toast("File must be 10 MB or smaller.", "danger");
      return;
    }

    const fd = new FormData();
    fd.append("file", file);
    el.dropTitle.textContent = `Analyzing ${file.name}...`;

    try {
      const res = await fetch("/api/analyze-file", {
        method: "POST",
        headers: { "X-CSRF-Token": CSRF },
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

      el.labResult.classList.remove("hidden");
      el.labFileName.textContent = data.filename;
      el.labMeta.textContent = `${data.duration_sec}s • Checked in ${data.latency_ms} ms`;

      const pct = Math.round(data.peak_fused_score * 100);
      el.labScoreText.textContent = `${pct}%`;
      el.labScoreFill.style.width = `${pct}%`;
      el.labScoreFill.classList.toggle("danger", data.is_emergency);

      el.labBadge.textContent = data.is_emergency ? "DISTRESS" : "SAFE";
      el.labBadge.className = `verdict-pill ${data.is_emergency ? "danger" : "safe"}`;

      el.labTimeline.replaceChildren();
      (data.timeline || []).forEach((w) => {
        const bar = document.createElement("div");
        bar.className = `lab-bar ${w.flagged ? "hit" : ""}`;
        bar.style.height = `${Math.max(10, Math.round(w.fused_score * 100))}%`;
        bar.title = `${w.start_sec}s–${w.end_sec}s: ${Math.round(w.fused_score * 100)}%`;
        el.labTimeline.appendChild(bar);
      });
    } catch (err) {
      toast(`Could not analyze file: ${err.message}`, "danger");
    } finally {
      el.dropTitle.textContent = "Drop audio file here or browse";
    }
  }

  // ------------------------------------------------------------------
  // 60 FPS Fluid AI Orb + Minimal Mirror Equalizer
  // ------------------------------------------------------------------
  function fitCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    const w = Math.max(60, Math.floor(r.width * dpr));
    const h = Math.max(24, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { w, h, dpr };
  }

  function palette() {
    if (state.uiState === "emergency") return { c1: "#f43f5e", c2: "#fb7185", halo: "rgba(244,63,94,0.42)" };
    if (state.uiState === "elevated") return { c1: "#f59e0b", c2: "#fb923c", halo: "rgba(245,158,11,0.34)" };
    if (state.armed) return { c1: "#22d3ee", c2: "#818cf8", halo: "rgba(34,211,238,0.34)" };
    return { c1: "#38bdf8", c2: "#6366f1", halo: "rgba(56,189,248,0.22)" };
  }

  function drawOrb() {
    const { w, h, dpr } = fitCanvas(el.orbCanvas);
    const ctx = el.orbCanvas.getContext("2d");
    const cx = w / 2;
    const cy = h / 2;
    const baseR = Math.min(w, h) * 0.30;
    const pal = palette();

    ctx.clearRect(0, 0, w, h);

    state.smoothRms += (state.liveRms - state.smoothRms) * 0.18;
    const energy = Math.min(1, state.smoothRms * 11 + state.threatScore * 0.35);
    state.t += 0.014 + energy * 0.032;

    // Soft outer volumetric bloom
    const halo = ctx.createRadialGradient(cx, cy, baseR * 0.1, cx, cy, baseR * (1.75 + energy * 0.28));
    halo.addColorStop(0, pal.halo);
    halo.addColorStop(0.55, "rgba(15,23,42,0.12)");
    halo.addColorStop(1, "rgba(5,7,14,0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR * 2.0, 0, Math.PI * 2);
    ctx.fill();

    // Inner glowing liquid glass core
    const coreGrad = ctx.createRadialGradient(
      cx - baseR * 0.25,
      cy - baseR * 0.25,
      baseR * 0.1,
      cx,
      cy,
      baseR * (0.92 + energy * 0.14)
    );
    coreGrad.addColorStop(0, "rgba(255,255,255,0.22)");
    coreGrad.addColorStop(0.45, pal.halo);
    coreGrad.addColorStop(1, "rgba(15,23,42,0.08)");
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR * (0.86 + energy * 0.14), 0, Math.PI * 2);
    ctx.fill();

    // 3 Smooth organic audio-reactive plasma waves
    const pts = 96;
    for (let layer = 0; layer < 3; layer++) {
      ctx.beginPath();
      for (let i = 0; i <= pts; i++) {
        const theta = (i / pts) * Math.PI * 2;
        let fft = 0;
        if (state.analyser && state.armed) {
          fft = (state.freq[Math.floor(((i % pts) / pts) * 64)] || 0) / 255;
        }
        const w1 = Math.sin(theta * (2 + layer) + state.t * (1.1 + layer * 0.35)) * (5 + energy * 15);
        const w2 = Math.cos(theta * (5 - layer) - state.t * 1.4) * (3 + fft * 18);
        const r = baseR * (0.88 + layer * 0.08) + (w1 + w2) * dpr * 0.55;
        const x = cx + Math.cos(theta) * r;
        const y = cy + Math.sin(theta) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = layer % 2 === 0 ? pal.c1 : pal.c2;
      ctx.lineWidth = (2.2 - layer * 0.45) * dpr;
      ctx.globalAlpha = 0.88 - layer * 0.24;
      ctx.shadowColor = pal.c1;
      ctx.shadowBlur = 16 * dpr;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;

    // Gentle floating light motes
    for (const m of state.motes) {
      m.a += m.s * (1 + energy * 2.2);
      const mr = baseR * m.r * (1.06 + energy * 0.15);
      ctx.fillStyle = pal.c1;
      ctx.globalAlpha = m.op * (0.45 + energy * 0.55);
      ctx.beginPath();
      ctx.arc(cx + Math.cos(m.a) * mr, cy + Math.sin(m.a) * mr, m.sz * dpr * 0.65, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawEqualizer() {
    const { w, h, dpr } = fitCanvas(el.eqCanvas);
    const ctx = el.eqCanvas.getContext("2d");
    const pal = palette();
    ctx.clearRect(0, 0, w, h);

    const bars = 24;
    const gap = w / bars;
    const barW = Math.max(2 * dpr, gap * 0.52);

    for (let i = 0; i < bars; i++) {
      let amp = 0.12 + 0.08 * Math.sin(state.t * 1.6 + i * 0.45);
      if (state.analyser && state.armed) {
        const idx = Math.floor((Math.abs(i - bars / 2) / (bars / 2)) * 48) + 2;
        amp = Math.max(0.12, (state.freq[idx] || 0) / 255);
      }
      const bh = Math.max(3 * dpr, amp * h * 0.92);
      const x = i * gap + (gap - barW) / 2;
      const y = (h - bh) / 2;

      ctx.fillStyle = i % 2 === 0 ? pal.c1 : pal.c2;
      ctx.globalAlpha = state.armed ? 0.9 : 0.35;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, bh, barW / 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function loop() {
    if (state.analyser && state.armed) state.analyser.getByteFrequencyData(state.freq);
    if (!document.getElementById("view-live").classList.contains("hidden")) {
      drawOrb();
      drawEqualizer();
    }
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------
  // Event Listeners
  // ------------------------------------------------------------------
  function bind() {
    // 3-Tab Navigation
    el.mainNavTabs.addEventListener("click", (ev) => {
      const btn = ev.target.closest(".tab-btn");
      if (!btn) return;
      const view = btn.dataset.view;
      el.mainNavTabs.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
      ["live", "lab", "activity"].forEach((v) => {
        document.getElementById(`view-${v}`).classList.toggle("hidden", v !== view);
      });
    });

    el.btnListen.addEventListener("click", toggleListening);
    el.btnCalibrate.addEventListener("click", calibrateMic);
    el.btnRefreshMics.addEventListener("click", refreshMics);

    el.btnSafeMode.addEventListener("click", async () => {
      state.dryRun = !state.dryRun;
      syncSettingsUI();
      try {
        await apiPost("/api/settings", { dry_run: state.dryRun });
        toast(state.dryRun ? "Safe Mode ON (simulated alerts)" : "Live Alerts ARMED", state.dryRun ? "ok" : "danger");
      } catch (err) {
        toast(err.message, "danger");
      }
    });

    el.btnSos.addEventListener("click", async () => {
      try {
        await apiPost("/api/alert/sos");
        setUIState("emergency", "SOS Alert Triggered", "Manual emergency dispatch activated.");
      } catch (err) {
        toast(err.message, "danger");
      }
    });

    el.btnCancelAlert.addEventListener("click", async () => {
      try { await apiPost("/api/alert/cancel"); } catch (err) { toast(err.message, "danger"); }
    });

    // Test Audio Drag & Drop
    el.dropZone.addEventListener("click", () => el.fileInput.click());
    el.dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") el.fileInput.click(); });
    el.fileInput.addEventListener("change", () => {
      if (el.fileInput.files?.[0]) analyzeFile(el.fileInput.files[0]);
      el.fileInput.value = "";
    });
    el.dropZone.addEventListener("dragover", (e) => { e.preventDefault(); el.dropZone.classList.add("dragover"); });
    el.dropZone.addEventListener("dragleave", () => el.dropZone.classList.remove("dragover"));
    el.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      el.dropZone.classList.remove("dragover");
      if (e.dataTransfer?.files?.[0]) analyzeFile(e.dataTransfer.files[0]);
    });

    // Activity sub-tabs
    el.logTabs.addEventListener("click", (ev) => {
      const btn = ev.target.closest(".mini-tab");
      if (!btn) return;
      const tab = btn.dataset.logtab;
      el.logTabs.querySelectorAll(".mini-tab").forEach((b) => b.classList.toggle("active", b === btn));
      el.incidentList.classList.toggle("hidden", tab !== "incidents");
      el.eventList.classList.toggle("hidden", tab !== "logs");
    });

    // Settings Drawer
    const openDrawer = () => { el.drawerOverlay.classList.remove("hidden"); el.settingsDrawer.classList.remove("hidden"); };
    const closeDrawer = () => { el.drawerOverlay.classList.add("hidden"); el.settingsDrawer.classList.add("hidden"); };
    el.btnSettings.addEventListener("click", openDrawer);
    el.btnCloseSettings.addEventListener("click", closeDrawer);
    el.drawerOverlay.addEventListener("click", closeDrawer);

    el.sensitivitySeg.addEventListener("click", (ev) => {
      const b = ev.target.closest(".seg-opt");
      if (!b) return;
      state.sensitivity = b.dataset.sens;
      syncSettingsUI();
    });

    el.sourceModeSeg.addEventListener("click", async (ev) => {
      const b = ev.target.closest(".seg-opt");
      if (!b || b.dataset.mode === state.sourceMode) return;
      if (state.armed) await stopListening();
      state.sourceMode = b.dataset.mode;
      syncSettingsUI();
      await refreshMics();
    });

    el.setCancelWin.addEventListener("input", () => { el.cancelWinText.textContent = `${el.setCancelWin.value}s`; });
    el.setCooldown.addEventListener("input", () => { el.cooldownText.textContent = `${el.setCooldown.value}s`; });

    el.btnSaveSettings.addEventListener("click", async () => {
      const kws = el.setKeywords.value.split(",").map((s) => s.trim()).filter(Boolean);
      state.language = el.langSelect.value;
      try {
        const res = await apiPost("/api/settings", {
          sensitivity: state.sensitivity,
          language: state.language,
          dry_run: el.setDryRun.checked,
          save_evidence: el.setSaveEvidence.checked,
          cancel_window_seconds: Number(el.setCancelWin.value),
          cooldown_seconds: Number(el.setCooldown.value),
          keywords: kws,
        });
        state.dryRun = res.alerts.dry_run;
        state.saveEvidence = res.alerts.save_evidence;
        state.cancelWindow = res.alerts.cancel_window_seconds;
        state.cooldown = res.alerts.cooldown_seconds;
        state.keywords = res.detector.keywords;
        syncSettingsUI(res.alerts);
        if (state.armed && state.sourceMode === "browser") startSpeech();
        closeDrawer();
        toast("Settings saved.", "ok");
      } catch (err) {
        toast(err.message, "danger");
      }
    });

    el.btnTestAlert.addEventListener("click", async () => {
      try {
        await apiPost("/api/alert/test");
        toast("Test alert triggered.", "info");
      } catch (err) {
        toast(err.message, "danger");
      }
    });
  }

  bind();
  bootstrap();
  requestAnimationFrame(loop);
})();
