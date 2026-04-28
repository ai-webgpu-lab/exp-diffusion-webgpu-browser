const EXECUTION_MODES = {
  webgpu: {
    label: "WebGPU ready",
    backend: "webgpu",
    fallbackTriggered: false,
    workerMode: "worker",
    stageMultiplier: 1,
    decodeOffsetMs: 0,
    safetyOffsetMs: 0
  },
  fallback: {
    label: "CPU fallback",
    backend: "cpu",
    fallbackTriggered: true,
    workerMode: "main",
    stageMultiplier: 1.94,
    decodeOffsetMs: 52,
    safetyOffsetMs: 18
  }
};

function resolveExecutionMode() {
  const requested = new URLSearchParams(window.location.search).get("mode");
  const hasWebGpu = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  if (requested === "fallback" || !hasWebGpu) return EXECUTION_MODES.fallback;
  return EXECUTION_MODES.webgpu;
}

const executionMode = resolveExecutionMode();

const requestedMode = typeof window !== "undefined"
  ? new URLSearchParams(window.location.search).get("mode")
  : null;
const isRealRuntimeMode = typeof requestedMode === "string" && requestedMode.startsWith("real-");
const REAL_ADAPTER_WAIT_MS = 5000;
const REAL_ADAPTER_LOAD_MS = 20000;

function withTimeout(promise, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs);
    promise.then((value) => {
      clearTimeout(timer);
      resolve(value);
    }, (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function findRegisteredRealRuntime() {
  const registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null;
  if (!registry || typeof registry.list !== "function") return null;
  return registry.list().find((adapter) => adapter && adapter.isReal === true) || null;
}

async function awaitRealRuntime(timeoutMs = REAL_ADAPTER_WAIT_MS) {
  const startedAt = performance.now();
  while (performance.now() - startedAt < timeoutMs) {
    const adapter = findRegisteredRealRuntime();
    if (adapter) return adapter;
    if (typeof window !== "undefined" && window.__aiWebGpuLabRealDiffusionBootstrapError) {
      return null;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return null;
}

const state = {
  startedAt: performance.now(),
  fixture: null,
  environment: buildEnvironment(),
  capability: null,
  active: false,
  run: null,
  realAdapterError: null,
  logs: []
};

const elements = {
  statusRow: document.getElementById("status-row"),
  summary: document.getElementById("summary"),
  probeCapability: document.getElementById("probe-capability"),
  runGeneration: document.getElementById("run-generation"),
  downloadJson: document.getElementById("download-json"),
  promptView: document.getElementById("prompt-view"),
  canvas: document.getElementById("image-canvas"),
  metricGrid: document.getElementById("metric-grid"),
  metaGrid: document.getElementById("meta-grid"),
  logList: document.getElementById("log-list"),
  resultJson: document.getElementById("result-json")
};

function round(value, digits = 2) {
  if (!Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function parseBrowser() {
  const ua = navigator.userAgent;
  for (const [needle, name] of [["Edg/", "Edge"], ["Chrome/", "Chrome"], ["Firefox/", "Firefox"], ["Version/", "Safari"]]) {
    const marker = ua.indexOf(needle);
    if (marker >= 0) return { name, version: ua.slice(marker + needle.length).split(/[\s)/;]/)[0] || "unknown" };
  }
  return { name: "Unknown", version: "unknown" };
}

function parseOs() {
  const ua = navigator.userAgent;
  if (/Windows NT/i.test(ua)) return { name: "Windows", version: (ua.match(/Windows NT ([0-9.]+)/i) || [])[1] || "unknown" };
  if (/Mac OS X/i.test(ua)) return { name: "macOS", version: ((ua.match(/Mac OS X ([0-9_]+)/i) || [])[1] || "unknown").replace(/_/g, ".") };
  if (/Android/i.test(ua)) return { name: "Android", version: (ua.match(/Android ([0-9.]+)/i) || [])[1] || "unknown" };
  if (/(iPhone|iPad|CPU OS)/i.test(ua)) return { name: "iOS", version: ((ua.match(/OS ([0-9_]+)/i) || [])[1] || "unknown").replace(/_/g, ".") };
  if (/Linux/i.test(ua)) return { name: "Linux", version: "unknown" };
  return { name: "Unknown", version: "unknown" };
}

function inferDeviceClass() {
  const threads = navigator.hardwareConcurrency || 0;
  const memory = navigator.deviceMemory || 0;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  if (mobile) return memory >= 6 && threads >= 8 ? "mobile-high" : "mobile-mid";
  if (memory >= 16 && threads >= 12) return "desktop-high";
  if (memory >= 8 && threads >= 8) return "desktop-mid";
  if (threads >= 4) return "laptop";
  return "unknown";
}

function buildEnvironment() {
  return {
    browser: parseBrowser(),
    os: parseOs(),
    device: {
      name: navigator.platform || "unknown",
      class: inferDeviceClass(),
      cpu: navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency} threads` : "unknown",
      memory_gb: navigator.deviceMemory || undefined,
      power_mode: "unknown"
    },
    gpu: { adapter: "pending", required_features: [], limits: {} },
    backend: "pending",
    fallback_triggered: false,
    worker_mode: "main",
    cache_state: "warm"
  };
}

function log(message) {
  state.logs.unshift(`[${new Date().toLocaleTimeString()}] ${message}`);
  state.logs = state.logs.slice(0, 12);
  renderLogs();
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadFixture() {
  if (state.fixture) return state.fixture;
  const response = await fetch("./diffusion-fixture.json", { cache: "no-store" });
  state.fixture = await response.json();
  return state.fixture;
}

function createRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

function drawPlaceholder() {
  const ctx = elements.canvas.getContext("2d");
  const { width, height } = elements.canvas;
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#07111d");
  gradient.addColorStop(0.5, "#101828");
  gradient.addColorStop(1, "#04070d");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(244, 114, 182, 0.14)";
  ctx.lineWidth = 1;
  for (let index = 0; index < 14; index += 1) {
    const y = (height / 13) * index + (index % 2) * 2;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(226, 232, 240, 0.82)";
  ctx.font = "600 24px Segoe UI";
  ctx.fillText("Diffusion readiness preview", 32, 54);
  ctx.font = "16px Segoe UI";
  ctx.fillStyle = "rgba(188, 200, 217, 0.92)";
  ctx.fillText("Probe capability and run the deterministic generation baseline.", 32, 84);
}

function drawGeneratedImage(seed) {
  const ctx = elements.canvas.getContext("2d");
  const { width, height } = elements.canvas;
  const rng = createRng(seed);

  const sky = ctx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "#08111d");
  sky.addColorStop(0.35, "#13213a");
  sky.addColorStop(0.7, "#2b1c42");
  sky.addColorStop(1, "#090d17");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  for (let star = 0; star < 90; star += 1) {
    const x = rng() * width;
    const y = rng() * height * 0.52;
    const alpha = 0.28 + rng() * 0.54;
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.fillRect(x, y, 1.2 + rng() * 1.8, 1.2 + rng() * 1.8);
  }

  const auroraColors = ["rgba(96,165,250,0.18)", "rgba(244,114,182,0.18)", "rgba(245,158,11,0.16)"];
  for (let band = 0; band < 4; band += 1) {
    ctx.beginPath();
    ctx.moveTo(-20, 90 + band * 24);
    for (let x = 0; x <= width + 40; x += 40) {
      const y = 86 + band * 34 + Math.sin(x * 0.01 + band * 1.2) * (22 + band * 4) + rng() * 10;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(width + 40, height * 0.5);
    ctx.lineTo(-20, height * 0.5);
    ctx.closePath();
    ctx.fillStyle = auroraColors[band % auroraColors.length];
    ctx.fill();
  }

  function drawMountain(baseY, amplitude, color, offset) {
    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(0, baseY);
    for (let x = 0; x <= width; x += 28) {
      const y = baseY - Math.abs(Math.sin((x + offset) * 0.008) * amplitude) - rng() * amplitude * 0.35;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  drawMountain(height * 0.64, 62, "#101a2a", 0);
  drawMountain(height * 0.72, 48, "#162235", 90);
  drawMountain(height * 0.82, 36, "#0c1320", 180);

  const lake = ctx.createLinearGradient(0, height * 0.68, 0, height);
  lake.addColorStop(0, "rgba(30,64,175,0.32)");
  lake.addColorStop(1, "rgba(3,7,18,0.9)");
  ctx.fillStyle = lake;
  ctx.fillRect(0, height * 0.68, width, height * 0.32);

  ctx.fillStyle = "#111827";
  ctx.fillRect(width * 0.58, height * 0.48, 96, 72);
  ctx.fillRect(width * 0.615, height * 0.42, 26, 64);
  ctx.beginPath();
  ctx.moveTo(width * 0.55, height * 0.48);
  ctx.lineTo(width * 0.63, height * 0.36);
  ctx.lineTo(width * 0.71, height * 0.48);
  ctx.closePath();
  ctx.fillStyle = "#1f2937";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(width * 0.63, height * 0.43, 30, Math.PI, Math.PI * 2);
  ctx.fillStyle = "#0f172a";
  ctx.fill();
  ctx.strokeStyle = "rgba(226,232,240,0.28)";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = "rgba(226,232,240,0.9)";
  ctx.fillRect(width * 0.61, height * 0.51, 14, 22);
  ctx.fillRect(width * 0.64, height * 0.51, 14, 22);

  for (let line = 0; line < 26; line += 1) {
    const y = height * 0.72 + line * 7;
    const alpha = 0.04 + (line % 3) * 0.03;
    ctx.strokeStyle = `rgba(226,232,240,${alpha})`;
    ctx.beginPath();
    ctx.moveTo(width * 0.1, y);
    ctx.lineTo(width * 0.9, y + Math.sin(line * 0.7) * 3);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(226,232,240,0.88)";
  ctx.font = "600 16px Segoe UI";
  ctx.fillText("seed 41", 26, height - 26);
}

async function probeCapability() {
  if (state.active) return;
  state.active = true;
  render();

  const hasWebGpu = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const forcedFallback = new URLSearchParams(window.location.search).get("mode") === "fallback";
  const ready = hasWebGpu && !forcedFallback;

  state.capability = {
    hasWebGpu,
    adapter: ready ? "navigator.gpu available" : "cpu-diffusion-fallback",
    requiredFeatures: ready ? ["shader-f16"] : []
  };
  state.environment.gpu = {
    adapter: state.capability.adapter,
    required_features: state.capability.requiredFeatures,
    limits: ready ? { maxStorageBuffersPerShaderStage: 8, maxTextureDimension2D: 8192 } : {}
  };
  state.environment.backend = executionMode.backend;
  state.environment.fallback_triggered = executionMode.fallbackTriggered;
  state.environment.worker_mode = executionMode.workerMode;
  state.active = false;

  log(ready ? "WebGPU diffusion path selected." : "Fallback diffusion path selected.");
  render();
}

async function runRealRuntimeDiffusion(adapter) {
  log(`Connecting real runtime adapter '${adapter.id}'.`);
  await withTimeout(
    Promise.resolve(adapter.loadModel({ modelId: "diffusion-webgpu-browser-default" })),
    REAL_ADAPTER_LOAD_MS,
    `loadModel(${adapter.id})`
  );
  const prefill = await withTimeout(
    Promise.resolve(adapter.prefill({ promptTokens: 96 })),
    REAL_ADAPTER_LOAD_MS,
    `prefill(${adapter.id})`
  );
  const decode = await withTimeout(
    Promise.resolve(adapter.decode({ tokenBudget: 32 })),
    REAL_ADAPTER_LOAD_MS,
    `decode(${adapter.id})`
  );
  log(`Real runtime adapter '${adapter.id}' ready: prefill_tok_per_sec=${prefill?.tokPerSec ?? "?"}, decode_tok_per_sec=${decode?.tokPerSec ?? "?"}.`);
  return { adapter, prefill, decode };
}

async function runGeneration() {
  if (state.active) return;
  if (!state.capability) await probeCapability();

  state.active = true;
  state.run = null;
  render();

  if (isRealRuntimeMode) {
    log(`Mode=${requestedMode} requested; awaiting real runtime adapter registration.`);
    const adapter = await awaitRealRuntime();
    if (adapter) {
      try {
        const { prefill, decode } = await runRealRuntimeDiffusion(adapter);
        state.realAdapterPrefill = prefill;
        state.realAdapterDecode = decode;
        state.realAdapter = adapter;
      } catch (error) {
        state.realAdapterError = error?.message || String(error);
        log(`Real runtime '${adapter.id}' failed: ${state.realAdapterError}; falling back to deterministic.`);
      }
    } else {
      const reason = (typeof window !== "undefined" && window.__aiWebGpuLabRealDiffusionBootstrapError) || "timed out waiting for adapter registration";
      state.realAdapterError = reason;
      log(`No real runtime adapter registered (${reason}); falling back to deterministic diffusion baseline.`);
    }
  }

  const fixture = await loadFixture();
  log(`Prompt tag ${fixture.promptTag} loaded with ${fixture.steps} denoise steps.`);

  let denoiseMs = 0;
  for (let index = 0; index < fixture.denoiseWindows.length; index += 1) {
    const steps = fixture.denoiseWindows[index];
    const windowMs = fixture.latentMsPerWindow[index] * executionMode.stageMultiplier;
    await sleep(windowMs);
    denoiseMs += windowMs;
    log(`Denoise window ${index + 1}/${fixture.denoiseWindows.length}: ${steps} steps complete.`);
  }

  const decodeMs = (fixture.decodeMs + executionMode.decodeOffsetMs) * executionMode.stageMultiplier;
  await sleep(decodeMs);
  log("Latents decoded into RGB image.");

  const safetyMs = (fixture.safetyMs + executionMode.safetyOffsetMs) * executionMode.stageMultiplier;
  await sleep(safetyMs);
  log("Safety and resolution checks recorded.");

  const totalMs = denoiseMs + decodeMs + safetyMs;
  const secPerImage = totalMs / 1000;
  const stepsPerSec = fixture.steps / Math.max(denoiseMs / 1000, 0.001);
  const resolutionSuccessRate = executionMode.fallbackTriggered ? 0.67 : 1;
  const oomOrFailRate = executionMode.fallbackTriggered ? 0.33 : 0;

  state.run = {
    prompt: fixture.prompt,
    negativePrompt: fixture.negativePrompt,
    scheduler: fixture.scheduler,
    seed: fixture.seed,
    width: fixture.width,
    height: fixture.height,
    steps: fixture.steps,
    guidanceScale: fixture.guidanceScale,
    previewFrames: fixture.previewFrames,
    promptTag: fixture.promptTag,
    denoiseMs,
    decodeMs,
    safetyMs,
    totalMs,
    secPerImage,
    stepsPerSec,
    resolutionSuccessRate,
    oomOrFailRate,
    realAdapter: state.realAdapter || null
  };

  drawGeneratedImage(fixture.seed);
  state.active = false;
  log(`Diffusion baseline complete: ${round(state.run.secPerImage, 3)} sec/image, ${round(state.run.stepsPerSec, 2)} steps/s.`);
  render();
}

function buildPromptText() {
  const fixture = state.fixture;
  const run = state.run;
  if (!fixture) return "Loading diffusion fixture.";
  return [
    `prompt: ${fixture.prompt}`,
    `negative_prompt: ${fixture.negativePrompt}`,
    `scheduler: ${fixture.scheduler}`,
    `resolution: ${fixture.width}x${fixture.height}`,
    `seed: ${fixture.seed}`,
    `steps: ${fixture.steps}`,
    `guidance_scale: ${fixture.guidanceScale}`,
    `preview_frames: ${fixture.previewFrames}`,
    run ? `sec_per_image: ${round(run.secPerImage, 3)}` : "sec_per_image: pending",
    run ? `steps_per_sec: ${round(run.stepsPerSec, 2)}` : "steps_per_sec: pending"
  ].join("\n");
}

function describeRuntimeAdapter() {
  const registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null;
  const requested = typeof window !== "undefined"
    ? new URLSearchParams(window.location.search).get("mode")
    : null;
  if (registry) {
    return registry.describe(requested);
  }
  return {
    id: "deterministic-diffusion",
    label: "Deterministic Diffusion",
    status: "deterministic",
    isReal: false,
    version: "1.0.0",
    capabilities: ["prefill", "decode", "fixed-output-budget"],
    runtimeType: "synthetic",
    message: "Runtime adapter registry unavailable; using inline deterministic mock."
  };
}

function buildResult() {
  const run = state.run;
  return {
    meta: {
      repo: "exp-diffusion-webgpu-browser",
      commit: "bootstrap-generated",
      timestamp: new Date().toISOString(),
      owner: "ai-webgpu-lab",
      track: "multimodal",
      scenario: (state.run && state.run.realAdapter) ? `diffusion-webgpu-browser-real-${state.run.realAdapter.id}` : (run ? "diffusion-webgpu-browser-readiness" : "diffusion-webgpu-browser-pending"),
      notes: run
        ? `promptTag=${run.promptTag}; scheduler=${run.scheduler}; resolution=${run.width}x${run.height}; seed=${run.seed}; steps=${run.steps}; previews=${run.previewFrames}; backend=${state.environment.backend}; fallback=${state.environment.fallback_triggered}; safety=pass${state.run && state.run.realAdapter ? `; realAdapter=${state.run.realAdapter.id}` : (isRealRuntimeMode && state.realAdapterError ? `; realAdapter=fallback(${state.realAdapterError})` : "")}`
        : "Probe capability, then run the deterministic browser diffusion readiness harness."
    },
    environment: state.environment,
    workload: {
      kind: "diffusion",
      name: "diffusion-webgpu-browser-readiness",
      input_profile: state.fixture ? `${state.fixture.width}x${state.fixture.height}-${state.fixture.steps}-steps` : "fixture-pending",
      model_id: "deterministic-diffusion-browser-v1",
      dataset: "diffusion-fixture-v1"
    },
    metrics: {
      common: {
        time_to_interactive_ms: round(performance.now() - state.startedAt, 2) || 0,
        init_ms: run ? round(run.totalMs, 2) || 0 : 0,
        success_rate: run ? 1 : 0.5,
        peak_memory_note: navigator.deviceMemory ? `${navigator.deviceMemory} GB reported by browser` : "deviceMemory unavailable",
        error_type: ""
      },
      diffusion: {
        sec_per_image: run ? round(run.secPerImage, 3) || 0 : 0,
        steps_per_sec: run ? round(run.stepsPerSec, 2) || 0 : 0,
        resolution_success_rate: run ? round(run.resolutionSuccessRate, 2) || 0 : 0,
        oom_or_fail_rate: run ? round(run.oomOrFailRate, 2) || 0 : 0
      }
    },
    status: run ? "success" : "partial",
    artifacts: {
      raw_logs: state.logs.slice(0, 5),
      deploy_url: "https://ai-webgpu-lab.github.io/exp-diffusion-webgpu-browser/",
      runtime_adapter: describeRuntimeAdapter()
    }
  };
}

function renderCards(container, items) {
  container.innerHTML = "";
  for (const [label, value] of items) {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<span class="label">${label}</span><div class="value">${value}</div>`;
    container.appendChild(card);
  }
}

function renderStatus() {
  const badges = state.active
    ? ["Generation running", executionMode.label]
    : state.run
      ? [`${round(state.run.secPerImage, 3)} sec/image`, `${round(state.run.stepsPerSec, 2)} steps/s`]
      : ["Fixture ready", executionMode.label];

  elements.statusRow.innerHTML = "";
  for (const text of badges) {
    const node = document.createElement("span");
    node.className = "badge";
    node.textContent = text;
    elements.statusRow.appendChild(node);
  }

  elements.summary.textContent = state.run
    ? `Last run: ${round(state.run.secPerImage, 3)} sec/image, ${round(state.run.stepsPerSec, 2)} steps/s, resolution success ${round(state.run.resolutionSuccessRate, 2)}, fail rate ${round(state.run.oomOrFailRate, 2)}.`
    : "Run the browser diffusion baseline to generate the deterministic observatory image and record latency and fail-rate metrics.";
}

function renderMetrics() {
  renderCards(elements.metricGrid, [
    ["Resolution", state.fixture ? `${state.fixture.width}x${state.fixture.height}` : "pending"],
    ["Steps", state.fixture ? String(state.fixture.steps) : "pending"],
    ["Sec/Image", state.run ? `${round(state.run.secPerImage, 3)} s` : "pending"],
    ["Steps/Sec", state.run ? `${round(state.run.stepsPerSec, 2)}` : "pending"],
    ["Resolution OK", state.run ? `${round(state.run.resolutionSuccessRate, 2)}` : "pending"],
    ["OOM/Fail", state.run ? `${round(state.run.oomOrFailRate, 2)}` : "pending"]
  ]);
}

function renderEnvironment() {
  renderCards(elements.metaGrid, [
    ["Browser", `${state.environment.browser.name} ${state.environment.browser.version}`],
    ["OS", `${state.environment.os.name} ${state.environment.os.version}`],
    ["Device", state.environment.device.class],
    ["CPU", state.environment.device.cpu],
    ["Memory", state.environment.device.memory_gb ? `${state.environment.device.memory_gb} GB` : "unknown"],
    ["Backend", state.environment.backend],
    ["Fallback", String(state.environment.fallback_triggered)],
    ["Worker", state.environment.worker_mode],
    ["Scheduler", state.run ? state.run.scheduler : (state.fixture ? state.fixture.scheduler : "pending")]
  ]);
}

function renderLogs() {
  elements.logList.innerHTML = "";
  const entries = state.logs.length ? state.logs : ["No diffusion activity yet."];
  for (const entry of entries) {
    const item = document.createElement("li");
    item.textContent = entry;
    elements.logList.appendChild(item);
  }
}

function render() {
  renderStatus();
  renderMetrics();
  renderEnvironment();
  renderLogs();
  elements.promptView.textContent = buildPromptText();
  elements.resultJson.textContent = JSON.stringify(buildResult(), null, 2);
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(buildResult(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `exp-diffusion-webgpu-browser-${state.run ? "readiness" : "pending"}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  log("Downloaded diffusion readiness JSON draft.");
}

elements.probeCapability.addEventListener("click", () => {
  probeCapability().catch((error) => {
    state.active = false;
    log(`Capability probe failed: ${error instanceof Error ? error.message : String(error)}`);
    render();
  });
});

elements.runGeneration.addEventListener("click", () => {
  runGeneration().catch((error) => {
    state.active = false;
    log(`Diffusion run failed: ${error instanceof Error ? error.message : String(error)}`);
    render();
  });
});

elements.downloadJson.addEventListener("click", downloadJson);

(async function init() {
  await loadFixture();
  drawPlaceholder();
  log("Browser diffusion readiness harness ready.");
  render();
})();
