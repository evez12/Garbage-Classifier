/* ==========================================================================
   Garbage Classifier — demo page
   --------------------------------------------------------------------------
   Inference happens in the visitor's browser with onnxruntime-web. The ONNX
   graph is the plain network: it takes an already-normalised (1,3,224,224)
   tensor and returns raw logits. Everything around it — resize, centre crop,
   normalise, the temperature division, the review threshold — is reproduced
   here from docs/model/model_meta.json, exactly as scripts/predict.py does it
   in Python.
   ========================================================================== */

import * as ort from "../vendor/onnxruntime/ort.wasm.min.mjs";

// Vendored rather than loaded from a CDN: the page then makes no external
// requests at all, and cannot break when someone else's CDN does.
// ort.wasm.min.mjs is the CPU-only build, so the runtime asks for
// ort-wasm-simd-threaded.{mjs,wasm} and never the much larger WebGPU variant.
// wasmPaths must be an absolute URL: the runtime resolves it as a module
// specifier, not against the document.
const ORT_DIST = new URL("../vendor/onnxruntime/", import.meta.url).href;
const MODEL_CACHE = "garbage-classifier-model-v1";

// GitHub Pages cannot send the COOP/COEP headers that SharedArrayBuffer needs,
// so multi-threaded wasm is unavailable. One thread plus SIMD it is.
ort.env.wasm.wasmPaths = ORT_DIST;
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = "error";

const $ = (id) => document.getElementById(id);

/* ---------- Theme -------------------------------------------------------- */
(function initTheme() {
  const root = document.documentElement;
  let stored = null;
  try {
    stored = localStorage.getItem("gc-theme");
  } catch {
    /* private mode or blocked storage — fall back to the OS preference */
  }
  if (stored === "light" || stored === "dark") root.dataset.theme = stored;

  $("theme-toggle")?.addEventListener("click", () => {
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const current = root.dataset.theme === "light" || root.dataset.theme === "dark"
      ? root.dataset.theme
      : (dark ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("gc-theme", next);
    } catch {
      /* not persisting the choice is acceptable */
    }
  });
})();

/* ---------- Formatting --------------------------------------------------- */
const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
const titled = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const classVar = (name) => `var(--class-${name})`;

/* ---------- Results sections -------------------------------------------- */
// Every class scores above this, so the F1 dot scale starts here rather than
// at zero; the header states the range so nothing is implied by omission.
const F1_FLOOR = 0.85;

const METRIC_ORDER = [
  ["accuracy", "Accuracy"],
  ["macro_f1", "Macro-F1"],
  ["balanced_accuracy", "Balanced accuracy"],
  ["mcc", "Matthews correlation"],
];

function renderMetrics(data) {
  const host = $("metrics");
  if (!host) return;
  // The CI whisker is drawn on a shared local scale so the widths are
  // comparable between tiles.
  const present = METRIC_ORDER.filter(([k]) => data.headline[k]);
  if (!present.length) return;
  const lo = Math.min(...present.map(([k]) => data.headline[k].ci[0])) - 0.004;
  const hi = Math.max(...present.map(([k]) => data.headline[k].ci[1])) + 0.004;
  const place = (v) => `${((v - lo) / (hi - lo)) * 100}%`;

  host.innerHTML = METRIC_ORDER.filter(([key]) => data.headline[key]).map(([key, label]) => {
    const m = data.headline[key];
    const delta = (m.value - m.baseline) * 100;
    return `
      <div class="metric">
        <span class="metric__name">${label}</span>
        <span class="metric__value">${pct(m.value, 2)}</span>
        <div class="ci">
          <div class="ci__track" role="img"
               aria-label="95% confidence interval from ${pct(m.ci[0], 2)} to ${pct(m.ci[1], 2)}">
            <span class="ci__span" style="left:${place(m.ci[0])};right:${100 - parseFloat(place(m.ci[1]))}%"></span>
            <span class="ci__point" style="left:${place(m.value)}"></span>
          </div>
          <div class="ci__scale"><span>${pct(m.ci[0], 1)}</span><span>${pct(m.ci[1], 1)}</span></div>
        </div>
        <span class="metric__baseline"><b>+${delta.toFixed(2)} pp</b> over the linear probe</span>
      </div>`;
  }).join("");
}

function renderPerClass(data) {
  const body = document.querySelector("#per-class tbody");
  if (!body) return;
  body.innerHTML = data.per_class.map((r) => `
    <tr>
      <th scope="row" style="font-weight:600">
        <span class="cell-class">
          <span class="cell-class__swatch" style="background:${classVar(r.class)}"></span>${titled(r.class)}
        </span>
      </th>
      <td class="num">${r.support}</td>
      <td>
        <span class="dot-scale">
          <span class="dot-scale__track">
            <span class="dot-scale__dot" style="left:${((r.f1 - F1_FLOOR) / (1 - F1_FLOOR)) * 100}%"></span>
          </span>
          <span class="dot-scale__num">${r.f1.toFixed(3)}</span>
        </span>
      </td>
      <td class="num">${r.precision.toFixed(3)}</td>
      <td class="num">${r.recall.toFixed(3)}</td>
      <td class="num">${r.missed}</td>
      <td class="num">${r.false_alarms}</td>
      <td class="num" style="color:var(--ok)">${r.delta_f1_pp == null ? "—" : `+${r.delta_f1_pp.toFixed(1)}`}</td>
    </tr>`).join("");
}

function renderMatrix(data) {
  const table = $("matrix");
  if (!table) return;
  const classes = data.classes;
  const cm = data.confusion_matrix;

  // Sequential ramp over off-diagonal magnitude: the diagonal dominates by two
  // orders of magnitude, so scaling to it would flatten every mistake to zero.
  let maxOff = 0;
  cm.forEach((row, i) => row.forEach((v, j) => {
    if (i !== j) maxOff = Math.max(maxOff, v);
  }));
  const level = (v, isDiag) => {
    if (isDiag) return 5;
    if (v === 0) return 0;
    return Math.min(4, 1 + Math.floor((v / maxOff) * 3.999));
  };

  const head = `<thead><tr><th><span class="visually-hidden">True stream</span></th>${
    classes.map((c) => `<th scope="col"><span>${titled(c)}</span></th>`).join("")
  }</tr></thead>`;

  const rows = classes.map((trueClass, i) => {
    const total = cm[i].reduce((a, b) => a + b, 0);
    const cells = classes.map((predClass, j) => {
      const v = cm[i][j];
      const isDiag = i === j;
      const share = total ? (v / total) * 100 : 0;
      const title = isDiag
        ? `${v} of ${total} ${trueClass} images classified correctly (${share.toFixed(1)}%)`
        : `${v} ${trueClass} image${v === 1 ? "" : "s"} classified as ${predClass}`;
      return `<td data-level="${level(v, isDiag)}"${isDiag ? ' class="is-diagonal"' : ""} title="${title}">${
        v === 0 ? "·" : v
      }</td>`;
    }).join("");
    return `<tr><th scope="row"><span class="cell-class">
      <span class="cell-class__swatch" style="background:${classVar(trueClass)}"></span>${titled(trueClass)}
    </span></th>${cells}</tr>`;
  }).join("");

  table.innerHTML = `${head}<tbody>${rows}</tbody>`;
  const keyMax = $("matrix-key-max");
  if (keyMax) keyMax.textContent = `0 → ${maxOff} off-diagonal, diagonal = correct`;

  // Narrative note built from the matrix itself, so it cannot drift.
  const off = [];
  cm.forEach((row, i) => row.forEach((v, j) => {
    if (i !== j && v > 0) off.push({ t: classes[i], p: classes[j], v });
  }));
  off.sort((a, b) => b.v - a.v);
  const note = $("matrix-note");
  if (note && data.per_class && data.per_class.length) {
    const worst = data.per_class[0];
    note.innerHTML = `
      <p><b>The biggest confusions</b></p>
      <p>${off.slice(0, 4).map((o) => `${titled(o.t)} read as ${o.p} — ${o.v}`).join("<br>")}</p>
      <p>
        Almost all of the remaining error is between materials that look alike in a photograph:
        paper against cardboard, plastic against glass. <b>${titled(worst.class)}</b> is the weakest
        stream (F1 ${worst.f1.toFixed(3)}), which is unsurprising — it is the residual category, it has
        the fewest training images, and a crushed item of almost anything can end up in it.
      </p>`;
  }
}

function renderRuns(data) {
  const body = document.querySelector("#runs tbody");
  if (!body) return;
  body.innerHTML = data.experiments.map((r) => {
    const selected = r.name === data.selected_model;
    return `
      <tr class="${selected ? "is-selected" : ""}">
        <th scope="row" style="font-weight:600">${r.name}${selected ? '<span class="tag">selected</span>' : ""}</th>
        <td style="white-space:normal;color:var(--ink-2)">${r.description}</td>
        <td class="num">${r.trainable_m < 0.01 ? "8.9K" : `${r.trainable_m.toFixed(1)}M`}</td>
        <td class="num">${r.epochs_run}</td>
        <td class="num">${r.train_time_min == null ? "—" : `${r.train_time_min.toFixed(0)} min`}</td>
        <td class="num">${r.val_macro_f1.toFixed(4)}</td>
        <td class="num">${r.test_macro_f1.toFixed(4)}</td>
      </tr>`;
  }).join("");
}

function renderSpec(data) {
  const left = $("spec-left");
  const right = $("spec-right");
  if (!left || !right) return;
  const row = (k, v) => `<div class="spec__row"><span class="spec__key">${k}</span><span class="spec__val">${v}</span></div>`;
  // Robustness, latency and the efficiency figures come from notebook sections
  // the scripts do not reimplement, so a script-only run may not have them.
  const lat = (data.latency || []).find((l) => l.batch === 1 && l.precision === "fp32");
  const rob = (data.robustness || {})[data.selected_model];
  const eff = data.efficiency || {};
  const env = data.environment || {};
  const review = (data.selective_prediction || [])
    .find((s) => s["target accuracy"] === data.recipe.review_target);
  const keep = (rows) => rows.filter(Boolean).join("");

  const sizes = data.split_sizes || (data.dataset && data.dataset.split_sizes) || {};
  left.innerHTML = keep([
    row("Selected run", data.selected_model),
    row("Backbone", "RegNetY-1.6GF, ImageNet-1k V2"),
    eff["parameters (M)"] &&
      row("Parameters", `${eff["parameters (M)"].toFixed(2)}M · ${eff["GFLOPs @224 (torchvision meta)"]} GFLOPs`),
    data.dataset && row("Images", `${data.dataset.images_audited.toLocaleString()} audited`),
    sizes.train && row("Split", `${sizes.train.toLocaleString()} / ${sizes.val.toLocaleString()} / ${sizes.test.toLocaleString()}`),
    env.seed != null && row("Seed", env.seed),
  ]);

  right.innerHTML = keep([
    env.gpu && row("Trained on", env.gpu),
    env.torch && row("PyTorch", `${env.torch} · bf16 autocast`),
    row("Temperature", data.recipe.temperature.toFixed(4)),
    row("Flip TTA", data.recipe.tta_hflip ? "on" : "rejected — not significant"),
    data.recipe.review_threshold != null &&
      row("Review threshold", `${data.recipe.review_threshold.toFixed(3)} → ${pct(review ? review["test coverage"] : 0)} handled automatically`),
    lat && row("GPU latency", `${lat["latency (ms / batch)"].toFixed(1)} ms per image`),
    rob && row("Corruption retention", rob.retention.toFixed(3)),
  ]);
}

/* ---------- Probability readout ------------------------------------------ */
let readoutRows = null;

function buildReadout(classes) {
  const host = $("readout");
  if (!host) return;
  host.innerHTML = classes.map((c) => `
    <div class="readout__row" data-class="${c}">
      <span class="readout__swatch" style="background:${classVar(c)}"></span>
      <span class="readout__name">${titled(c)}</span>
      <span class="readout__track"><span class="readout__bar"></span></span>
      <span class="readout__value">—</span>
    </div>`).join("");
  readoutRows = new Map(
    [...host.querySelectorAll(".readout__row")].map((el) => [el.dataset.class, el])
  );
}

function paintReadout(classes, probs) {
  if (!readoutRows) return;
  const order = [...classes.keys()].sort((a, b) => probs[b] - probs[a]);
  const host = $("readout");
  order.forEach((idx, rank) => {
    const name = classes[idx];
    const row = readoutRows.get(name);
    if (!row) return;
    row.classList.toggle("is-top", rank === 0);
    row.querySelector(".readout__bar").style.width = `${Math.max(probs[idx] * 100, probs[idx] > 0 ? 0.6 : 0)}%`;
    const v = probs[idx];
    row.querySelector(".readout__value").textContent = v < 0.001 ? "<0.1%" : pct(v, 1);
    host.appendChild(row); // reorder: most likely first
  });
}

function clearReadout(classes) {
  if (!readoutRows) return;
  classes.forEach((c) => {
    const row = readoutRows.get(c);
    row.classList.remove("is-top");
    row.querySelector(".readout__bar").style.width = "0%";
    row.querySelector(".readout__value").textContent = "—";
  });
}

/* ---------- Status and errors ------------------------------------------- */
function setStatus(state, text, progress) {
  const host = $("model-status");
  if (!host) return;
  host.dataset.state = state;
  $("model-status-text").textContent = text;
  const bar = $("progress");
  if (progress == null) {
    bar.style.visibility = "hidden";
  } else {
    bar.style.visibility = "visible";
    $("progress-fill").style.width = `${Math.round(progress * 100)}%`;
  }
}

function showAlert(message) {
  $("alert-text").textContent = message;
  $("alert").hidden = false;
}
const hideAlert = () => { $("alert").hidden = true; };

/* ---------- Preprocessing ----------------------------------------------- */
/**
 * Reproduce torchvision's eval transform on a canvas: resize the shorter side
 * to `resize`, centre-crop `crop`, scale to [0,1] and normalise. Large
 * reductions are done by repeated halving, because a single huge drawImage
 * step aliases badly and would shift the prediction.
 */
function preprocess(image, meta) {
  const resizeTo = meta.preprocess.resize[0];
  const crop = meta.preprocess.crop[0];
  const { mean, std } = meta.preprocess;

  const scale = resizeTo / Math.min(image.naturalWidth, image.naturalHeight);
  const targetW = Math.max(crop, Math.round(image.naturalWidth * scale));
  const targetH = Math.max(crop, Math.round(image.naturalHeight * scale));

  let srcCanvas = document.createElement("canvas");
  srcCanvas.width = image.naturalWidth;
  srcCanvas.height = image.naturalHeight;
  let ctx = srcCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);

  // Step down by halves while we are more than 2x away from the target.
  let w = srcCanvas.width;
  let h = srcCanvas.height;
  while (w / 2 >= targetW && h / 2 >= targetH) {
    const next = document.createElement("canvas");
    next.width = Math.max(targetW, Math.floor(w / 2));
    next.height = Math.max(targetH, Math.floor(h / 2));
    const nctx = next.getContext("2d", { willReadFrequently: true });
    nctx.imageSmoothingEnabled = true;
    nctx.imageSmoothingQuality = "high";
    nctx.drawImage(srcCanvas, 0, 0, next.width, next.height);
    srcCanvas = next;
    w = next.width;
    h = next.height;
  }

  const resized = document.createElement("canvas");
  resized.width = targetW;
  resized.height = targetH;
  const rctx = resized.getContext("2d", { willReadFrequently: true });
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = "high";
  rctx.drawImage(srcCanvas, 0, 0, targetW, targetH);

  const left = Math.round((targetW - crop) / 2);
  const top = Math.round((targetH - crop) / 2);
  const { data } = rctx.getImageData(left, top, crop, crop);

  const out = new Float32Array(3 * crop * crop);
  const plane = crop * crop;
  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    out[i] = (data[p] / 255 - mean[0]) / std[0];
    out[i + plane] = (data[p + 1] / 255 - mean[1]) / std[1];
    out[i + 2 * plane] = (data[p + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", out, [1, 3, crop, crop]);
}

function softmax(logits, temperature) {
  const scaled = Array.from(logits, (v) => v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/* ---------- Model loading ------------------------------------------------ */
let session = null;
let modelMeta = null;
let loadPromise = null;

async function fetchModelBytes(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) {
      onProgress(1);
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {
    /* Cache Storage unavailable — download without caching */
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} fetching the model`);

  try {
    if (cache) await cache.put(url, response.clone());
  } catch {
    /* over quota; the download below still succeeds */
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

function loadModel() {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    setStatus("loading", "Loading the model…", 0);
    modelMeta = await (await fetch("model/model_meta.json")).json();
    const sizeMb = (modelMeta.size_bytes / 1e6).toFixed(0);
    const bytes = await fetchModelBytes(`model/${modelMeta.model_file}`, (p) => {
      setStatus("loading", `Loading the model… ${Math.round(p * 100)}% of ${sizeMb} MB`, p);
    });
    setStatus("loading", "Starting the runtime…", 1);
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    setStatus("ready", `Model ready — ${sizeMb} MB, cached for next time`, null);
    return session;
  })().catch((err) => {
    loadPromise = null;
    setStatus("error", "The model could not be loaded", null);
    showAlert(
      "The model failed to download, so classifying is unavailable. Check your connection and reload the page. " +
      "The results below are unaffected — they come from the training run, not from your browser."
    );
    console.error(err);
    throw err;
  });
  return loadPromise;
}

/* ---------- Classify flow ----------------------------------------------- */
const imageEl = new Image();
let hasImage = false;
let busy = false;

function setImageSource(src, revokeAfterLoad) {
  return new Promise((resolve, reject) => {
    imageEl.onload = () => {
      if (revokeAfterLoad) URL.revokeObjectURL(src);
      resolve();
    };
    imageEl.onerror = () => {
      if (revokeAfterLoad) URL.revokeObjectURL(src);
      reject(new Error("decode failed"));
    };
    imageEl.src = src;
  });
}

async function acceptImage(src, { revoke = false, label = "" } = {}) {
  hideAlert();
  try {
    await setImageSource(src, revoke);
  } catch {
    showAlert("That file could not be read as an image. Try a JPEG, PNG or WebP.");
    return;
  }
  $("preview").src = imageEl.src;
  $("preview").alt = label ? `Selected sample: ${label}` : "The image you selected";
  $("dropzone").classList.add("has-image");
  $("classify").disabled = false;
  $("reset").disabled = false;
  hasImage = true;
  clearReadout(modelMeta?.classes ?? CLASSES_FALLBACK);
  resetVerdict();
  $("timing").textContent = "";
  loadModel().catch(() => {});
}

function resetVerdict() {
  $("verdict").classList.add("is-empty");
  $("verdict-class").textContent = "Ready when you are";
  $("verdict-confidence").textContent = "";
  $("verdict-swatch").hidden = true;
  $("review-badge").hidden = true;
}

async function classify() {
  if (!hasImage || busy) return;
  busy = true;
  const button = $("classify");
  const original = button.textContent;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Classifying';
  hideAlert();

  try {
    await loadModel();
    const tensor = preprocess(imageEl, modelMeta);
    const started = performance.now();
    const output = await session.run({ [modelMeta.input.name]: tensor });
    const logitsData = output[Object.keys(output)[0]].data;

    // The exported recipe may ask for horizontal-flip TTA; this run did not.
    let logits = Array.from(logitsData);
    if (modelMeta.tta_hflip) {
      const flipped = flipTensor(tensor, modelMeta.input.width);
      const out2 = await session.run({ [modelMeta.input.name]: flipped });
      const second = out2[Object.keys(out2)[0]].data;
      logits = logits.map((v, i) => 0.5 * (v + second[i]));
    }

    const probs = softmax(logits, modelMeta.temperature);
    const elapsed = performance.now() - started;

    const classes = modelMeta.classes;
    const top = probs.indexOf(Math.max(...probs));
    const label = classes[top];

    $("verdict").classList.remove("is-empty");
    $("verdict-class").textContent = titled(label);
    $("verdict-confidence").textContent = pct(probs[top], 1);
    const swatch = $("verdict-swatch");
    swatch.hidden = false;
    swatch.style.background = classVar(label);

    const threshold = modelMeta.review_threshold;
    $("review-badge").hidden = !(threshold != null && probs[top] < threshold);

    paintReadout(classes, probs);
    $("timing").textContent = `${elapsed.toFixed(0)} ms on your device`;
  } catch (err) {
    if (session) {
      showAlert("Classifying failed on this image. Try another file.");
      console.error(err);
    }
  } finally {
    busy = false;
    button.disabled = !hasImage;
    button.textContent = original;
  }
}

function flipTensor(tensor, width) {
  const src = tensor.data;
  const out = new Float32Array(src.length);
  const plane = width * width;
  for (let c = 0; c < 3; c += 1) {
    for (let y = 0; y < width; y += 1) {
      for (let x = 0; x < width; x += 1) {
        out[c * plane + y * width + x] = src[c * plane + y * width + (width - 1 - x)];
      }
    }
  }
  return new ort.Tensor("float32", out, tensor.dims);
}

/* ---------- Wiring ------------------------------------------------------- */
const CLASSES_FALLBACK = [
  "battery", "biological", "cardboard", "clothes", "glass",
  "metal", "paper", "plastic", "shoes", "trash",
];

function wireInput() {
  const dropzone = $("dropzone");
  const fileInput = $("file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    })
  );
  ["dragleave", "dragend", "drop"].forEach((type) =>
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragover"))
  );

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) handleFile(file);
    fileInput.value = "";
  });

  $("classify").addEventListener("click", classify);
  $("reset").addEventListener("click", () => {
    hasImage = false;
    $("dropzone").classList.remove("has-image");
    $("preview").removeAttribute("src");
    $("classify").disabled = true;
    $("reset").disabled = true;
    $("timing").textContent = "";
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
    clearReadout(modelMeta?.classes ?? CLASSES_FALLBACK);
    resetVerdict();
    hideAlert();
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showAlert("That is not an image file. Try a JPEG, PNG or WebP.");
      return;
    }
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
    acceptImage(URL.createObjectURL(file), { revoke: true });
  }
}

async function wireSamples() {
  const host = $("samples");
  if (!host) return;
  let samples = [];
  try {
    samples = await (await fetch("samples/samples.json")).json();
  } catch {
    host.innerHTML = '<p class="samples__label">Sample images are unavailable right now.</p>';
    return;
  }

  host.innerHTML = samples.map((s) => `
    <button class="sample" type="button" data-file="${s.file}" data-label="${s.label}"
            aria-pressed="false" title="${titled(s.label)} — from the test split">
      <img src="samples/${s.file}" alt="${titled(s.label)}" loading="lazy" decoding="async">
      <span class="sample__tag">${titled(s.label)}</span>
    </button>`).join("");

  host.querySelectorAll(".sample").forEach((button) => {
    button.addEventListener("click", () => {
      host.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
      button.setAttribute("aria-pressed", "true");
      acceptImage(`samples/${button.dataset.file}`, { label: button.dataset.label });
    });
  });
}

async function wireResults() {
  let data;
  try {
    data = await (await fetch("data/metrics.json")).json();
  } catch (err) {
    console.error("metrics.json could not be loaded", err);
    return;
  }
  buildReadout(data.classes);
  renderMetrics(data);
  renderPerClass(data);
  renderMatrix(data);
  renderRuns(data);
  renderSpec(data);
}

function announceModelPlan() {
  // Downloading 42 MB unprompted is rude on a metered connection, so on a slow
  // or data-saving link the model waits until an image is actually chosen.
  const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const slow = conn && (conn.saveData || /^(slow-)?2g$/.test(conn.effectiveType || ""));
  if (slow) {
    setStatus("idle", "The model (42 MB) downloads when you pick an image", null);
    return;
  }
  setStatus("idle", "Fetching the model…", 0);
  loadModel().catch(() => {});
}

resetVerdict();
buildReadout(CLASSES_FALLBACK);
wireInput();
wireSamples();
wireResults();
announceModelPlan();
