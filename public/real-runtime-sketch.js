// Real diffusion runtime integration sketch for exp-diffusion-webgpu-browser.
//
// Gated by ?mode=real-diffusion. Default deterministic harness path is untouched.
// `loadDiffuserFromCdn` is parameterized so tests can inject a stub.

const DEFAULT_TRANSFORMERS_VERSION = "3.0.0";
const DEFAULT_TRANSFORMERS_CDN = (version) => `https://esm.sh/@huggingface/transformers@${version}`;
const DEFAULT_MODEL_ID = "Xenova/sd-turbo";
const DEFAULT_TASK = "text-to-image";

export async function loadDiffuserFromCdn({ version = DEFAULT_TRANSFORMERS_VERSION } = {}) {
  const transformers = await import(/* @vite-ignore */ DEFAULT_TRANSFORMERS_CDN(version));
  if (!transformers || typeof transformers.pipeline !== "function") {
    throw new Error("transformers module did not expose pipeline()");
  }
  return {
    transformers,
    pipeline: transformers.pipeline,
    env: transformers.env
  };
}

export function buildRealDiffusionAdapter({
  pipeline,
  env,
  version = DEFAULT_TRANSFORMERS_VERSION,
  modelId = DEFAULT_MODEL_ID,
  task = DEFAULT_TASK
}) {
  if (typeof pipeline !== "function") {
    throw new Error("buildRealDiffusionAdapter requires a callable pipeline");
  }
  const sanitized = modelId.replace(/[^A-Za-z0-9]/g, "-").toLowerCase();
  const id = `diffusion-${sanitized}-${version.replace(/[^0-9]/g, "")}`;
  let runtime = null;

  return {
    id,
    label: `Diffusion ${modelId} (Transformers.js ${version})`,
    version,
    capabilities: ["prefill", "decode", "image-generation", "fixed-output-budget"],
    loadType: "async",
    backendHint: "webgpu",
    isReal: true,
    async loadRuntime({ device = "webgpu", dtype = "fp16" } = {}) {
      if (env && typeof env === "object") {
        env.allowRemoteModels = true;
      }
      runtime = await pipeline(task, modelId, { device, dtype });
      return runtime;
    },
    async prefill(_runtime, prompt) {
      const startedAt = performance.now();
      const text = String(prompt || "");
      const promptTokens = text.trim().split(/\s+/).filter(Boolean).length;
      const prefillMs = performance.now() - startedAt;
      return { promptTokens, prefillMs, text };
    },
    async decode(activeRuntime, prefillResult, outputTokenBudget = 4) {
      const target = activeRuntime || runtime;
      if (!target) {
        throw new Error("real diffusion adapter requires loadRuntime() before decode()");
      }
      const text = (prefillResult && prefillResult.text) || "a serene blackhole observatory";
      const startedAt = performance.now();
      const output = await target(text, {
        num_inference_steps: outputTokenBudget,
        guidance_scale: 1.0,
        height: 256,
        width: 256
      });
      const decodeMs = performance.now() - startedAt;
      const image = Array.isArray(output) ? output[0] : output;
      const widthPx = image && image.width ? image.width : 256;
      const heightPx = image && image.height ? image.height : 256;
      const tokens = outputTokenBudget;
      return {
        tokens,
        decodeMs,
        text,
        widthPx,
        heightPx,
        ttftMs: decodeMs,
        decodeTokPerSec: tokens / Math.max(decodeMs / 1000, 0.001)
      };
    }
  };
}

export async function connectRealDiffusion({
  registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null,
  loader = loadDiffuserFromCdn,
  version = DEFAULT_TRANSFORMERS_VERSION,
  modelId = DEFAULT_MODEL_ID,
  task = DEFAULT_TASK
} = {}) {
  if (!registry) {
    throw new Error("runtime registry not available");
  }
  const { pipeline, env } = await loader({ version });
  if (typeof pipeline !== "function") {
    throw new Error("loaded pipeline is not callable");
  }
  const adapter = buildRealDiffusionAdapter({ pipeline, env, version, modelId, task });
  registry.register(adapter);
  return { adapter, pipeline, env };
}

if (typeof window !== "undefined" && window.location && typeof window.location.search === "string") {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "real-diffusion" && !window.__aiWebGpuLabRealDiffusionBootstrapping) {
    window.__aiWebGpuLabRealDiffusionBootstrapping = true;
    connectRealDiffusion().catch((error) => {
      console.warn(`[real-diffusion] bootstrap failed: ${error.message}`);
      window.__aiWebGpuLabRealDiffusionBootstrapError = error.message;
    });
  }
}
