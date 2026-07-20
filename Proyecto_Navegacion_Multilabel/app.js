"use strict";

/*
 * Orden confirmado en LABEL_VOCAB del notebook de entrenamiento.
 * Cada etiqueta usa el threshold óptimo calculado sobre el conjunto de test.
 */
const LABELS = [
  {
    id: "letrero_pared",
    name: "letrero o rótulo en la pared",
    threshold: 0.45,
  },
  {
    id: "puerta",
    name: "puerta",
    threshold: 0.70,
  },
  {
    id: "escalera",
    name: "escalera",
    threshold: 0.35,
  },
  {
    id: "obstaculo",
    name: "posible obstáculo",
    threshold: 0.60,
  },
  {
    id: "pasillo",
    name: "pasillo o corredor",
    threshold: 0.25,
  },
];

const ONNX_RUNTIME_VERSION = "1.26.0";

/*
 * Para publicar una versión futura:
 * 1. Copia el archivo dentro de model/.
 * 2. Agrega otra entrada en este registro.
 * 3. Agrega su <option> correspondiente en index.html.
 * Las versiones deben conservar la entrada 224x224x3 y el orden de LABELS.
 */
const MODEL_REGISTRY = Object.freeze({
  "onnx-v1": Object.freeze({
    id: "onnx-v1",
    name: "ONNX — modelo actual",
    runtime: "onnx",
    url: "./model/modelo_navegacion_multilabel.onnx",
    inputName: "inputs",
    outputName: "output_0",
    layout: "NHWC",
    description: "Ejecuta el archivo ONNX mediante WebAssembly en este dispositivo.",
  }),
});

const CONFIG = Object.freeze({
  inputSize: 224,
  stablePredictionsRequired: 3,
  negativePredictionsToClear: 4,
  inferenceIntervalMs: 350,
  announcementCooldownMs: 8_000,
  speechLanguage: "es-EC",
});

const elements = {
  video: document.querySelector("#camera"),
  startButton: document.querySelector("#start-button"),
  stopButton: document.querySelector("#stop-button"),
  repeatButton: document.querySelector("#repeat-button"),
  modelSelect: document.querySelector("#model-select"),
  modelDescription: document.querySelector("#model-description"),
  imageUpload: document.querySelector("#image-upload"),
  analyzeUploadButton: document.querySelector("#analyze-upload-button"),
  testImageSelect: document.querySelector("#test-image-select"),
  analyzeTestButton: document.querySelector("#analyze-test-button"),
  imagePreviewPanel: document.querySelector("#image-preview-panel"),
  imagePreview: document.querySelector("#image-preview"),
  imageCaption: document.querySelector("#image-caption"),
  imageResultsPanel: document.querySelector("#image-results-panel"),
  imageScores: document.querySelector("#image-scores"),
  expectedLabels: document.querySelector("#expected-labels"),
  status: document.querySelector("#status"),
  detection: document.querySelector("#detection"),
  error: document.querySelector("#error"),
};

let activeModel = null;
const modelCache = new Map();
const modelPromises = new Map();
let mediaStream = null;
let animationFrameId = null;
let sessionId = 0;
let isRunning = false;
let isStarting = false;
let isPredicting = false;
let lastInferenceAt = 0;
let positiveStreaks = Array(LABELS.length).fill(0);
let negativeStreaks = Array(LABELS.length).fill(0);
let activeLabels = Array(LABELS.length).fill(false);
let lastAnnouncedSignature = "";
let lastAutomaticSpeechAt = 0;
let lastSpokenMessage = "";
let isAnalyzingImage = false;
let uploadedImageUrl = "";
let uploadedImageName = "";
let testImages = [];
let selectedTestImage = null;

elements.startButton.addEventListener("click", startNavigation);
elements.stopButton.addEventListener("click", () => stopNavigation(true));
elements.repeatButton.addEventListener("click", repeatLastAnnouncement);
elements.modelSelect.addEventListener("change", handleModelSelection);
elements.imageUpload.addEventListener("change", handleImageUpload);
elements.testImageSelect.addEventListener("change", handleTestImageSelection);
elements.analyzeUploadButton.addEventListener("click", () => analyzeSelectedImage("upload"));
elements.analyzeTestButton.addEventListener("click", () => analyzeSelectedImage("test"));
window.addEventListener("pagehide", cleanupPage);

restoreModelSelection();
updateModelDescription();
void loadTestImageManifest();

async function startNavigation() {
  if (isRunning || isStarting || isAnalyzingImage) return;

  const thisSession = ++sessionId;
  isStarting = true;
  clearError();
  resetDetectionState();
  setControls("starting");
  setStatus("Solicitando acceso a la cámara…");

  // Iniciar la voz desde un gesto del usuario mejora el funcionamiento en iOS.
  speak("Iniciando navegación asistida.", { remember: false });

  try {
    ensureInferenceSupport();
    ensureCameraSupport();
    mediaStream = await openRearCamera();
    if (thisSession !== sessionId) return;

    const selectedConfig = getSelectedModelConfig();
    setStatus(`Cargando ${selectedConfig.name}…`);
    activeModel = await loadNavigationModel(selectedConfig);
    if (thisSession !== sessionId) return;

    validateModel(activeModel);
    await warmUpModel(activeModel);
    if (thisSession !== sessionId) return;

    isStarting = false;
    isRunning = true;
    setControls("running");
    setStatus(`Navegación activa con ${activeModel.config.name}.`);
    speak("Navegación activa en el segundo piso. Apunta la cámara hacia el frente.", {
      remember: false,
    });
    animationFrameId = requestAnimationFrame(runInferenceLoop);
  } catch (error) {
    if (thisSession !== sessionId) return;
    console.error(error);
    const message = friendlyErrorMessage(error);
    stopNavigation(false);
    showError(message);
    setStatus("No se pudo iniciar la navegación.");
    speak(message, { remember: false });
  }
}

function ensureInferenceSupport() {
  if (!window.tf) {
    throw new Error("TFJS_UNAVAILABLE");
  }
  if (getSelectedModelConfig().runtime === "onnx" && !window.ort) {
    throw new Error("ONNX_RUNTIME_UNAVAILABLE");
  }
  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    throw new Error("SPEECH_UNSUPPORTED");
  }
}

function ensureCameraSupport() {
  if (!window.isSecureContext) {
    throw new Error("INSECURE_CONTEXT");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("CAMERA_UNSUPPORTED");
  }
}

async function loadTestImageManifest() {
  try {
    const response = await fetch("./test-images/manifest.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`TEST_MANIFEST_HTTP_${response.status}`);

    const manifest = await response.json();
    if (!Array.isArray(manifest.images) || manifest.images.length === 0) {
      throw new Error("INVALID_TEST_MANIFEST");
    }

    testImages = manifest.images;
    elements.testImageSelect.replaceChildren(
      new Option("Selecciona una imagen de prueba", ""),
      ...testImages.map((image, index) => {
        const labelNames = image.labels.map(getLabelName).join(", ");
        return new Option(`${index + 1}. ${image.filename} — ${labelNames}`, image.id);
      }),
    );
    elements.testImageSelect.disabled = false;
    refreshImageControlState();
  } catch (error) {
    console.error(error);
    elements.testImageSelect.replaceChildren(
      new Option("No se pudo cargar la galería de prueba", ""),
    );
    showError(
      "No se pudo cargar la galería de prueba. Aún puedes subir una foto o usar la cámara.",
    );
  }
}

function handleImageUpload() {
  if (isRunning || isStarting || isAnalyzingImage) return;

  const file = elements.imageUpload.files?.[0];
  if (!file) {
    clearUploadedImage();
    refreshImageControlState();
    return;
  }

  if (!file.type.startsWith("image/")) {
    clearUploadedImage();
    showError("Selecciona un archivo de imagen JPG, PNG o WebP.");
    refreshImageControlState();
    return;
  }

  if (file.size > 15 * 1024 * 1024) {
    clearUploadedImage();
    showError("La foto supera 15 MB. Elige una imagen más pequeña.");
    refreshImageControlState();
    return;
  }

  clearError();
  if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
  uploadedImageUrl = URL.createObjectURL(file);
  uploadedImageName = file.name;
  elements.imageResultsPanel.hidden = true;
  void showImagePreview(uploadedImageUrl, `Foto subida: ${file.name}`).catch(handlePreviewError);
  refreshImageControlState();
}

function handleTestImageSelection() {
  if (isRunning || isStarting || isAnalyzingImage) return;

  selectedTestImage =
    testImages.find((image) => image.id === elements.testImageSelect.value) ?? null;
  elements.imageResultsPanel.hidden = true;

  if (selectedTestImage) {
    const expected = formatSpanishList(selectedTestImage.labels.map(getLabelName));
    void showImagePreview(
      selectedTestImage.path,
      `Imagen de prueba: ${selectedTestImage.filename}. Etiquetas esperadas: ${expected}.`,
    ).catch(handlePreviewError);
  }

  refreshImageControlState();
}

async function analyzeSelectedImage(mode) {
  if (isRunning || isStarting || isAnalyzingImage) return;

  const context =
    mode === "upload"
      ? uploadedImageUrl
        ? { src: uploadedImageUrl, filename: uploadedImageName, labels: null }
        : null
      : selectedTestImage;
  if (!context) return;

  isAnalyzingImage = true;
  clearError();
  setControls("analyzing");
  setStatus("Preparando la imagen…");
  elements.imageResultsPanel.hidden = true;
  speak("Analizando imagen.", { remember: false });

  try {
    ensureInferenceSupport();
    const expected = context.labels
      ? formatSpanishList(context.labels.map(getLabelName))
      : "sin etiquetas esperadas";
    await showImagePreview(
      context.src ?? context.path,
      mode === "test"
        ? `Imagen de prueba: ${context.filename}. Etiquetas esperadas: ${expected}.`
        : `Foto subida: ${context.filename}`,
    );

    const selectedConfig = getSelectedModelConfig();
    setStatus(`Cargando ${selectedConfig.name}…`);
    activeModel = await loadNavigationModel(selectedConfig);
    validateModel(activeModel);
    await warmUpModel(activeModel);

    setStatus(`Analizando imagen con ${selectedConfig.name}…`);
    const scores =
      activeModel.config.runtime === "onnx"
        ? await predictWithOnnx(activeModel, elements.imagePreview)
        : await predictWithTfjs(activeModel, elements.imagePreview);
    renderImageResults(scores, context.labels);
    setStatus(`Imagen analizada con ${selectedConfig.name}.`);
  } catch (error) {
    console.error(error);
    const message = friendlyErrorMessage(error);
    showError(message);
    setStatus("No se pudo analizar la imagen.");
    speak(message, { remember: false });
  } finally {
    isAnalyzingImage = false;
    setControls("stopped");
  }
}

async function showImagePreview(src, caption) {
  if (elements.imagePreview.src !== new URL(src, document.baseURI).href) {
    elements.imagePreview.src = src;
  }
  elements.imageCaption.textContent = caption;
  elements.imagePreviewPanel.hidden = false;

  if (!elements.imagePreview.complete || elements.imagePreview.naturalWidth === 0) {
    await new Promise((resolve, reject) => {
      elements.imagePreview.addEventListener("load", resolve, { once: true });
      elements.imagePreview.addEventListener("error", reject, { once: true });
    });
  }
}

function handlePreviewError(error) {
  console.error(error);
  showError("No se pudo abrir la imagen seleccionada.");
}

function renderImageResults(scores, expectedLabelIds) {
  if (scores.length !== LABELS.length || scores.some((score) => !Number.isFinite(score))) {
    throw new Error("INVALID_PREDICTION");
  }

  const expectedSet = new Set(expectedLabelIds ?? []);
  const results = LABELS.map((label, index) => ({
    ...label,
    score: scores[index],
    detected: scores[index] >= label.threshold,
    expected: expectedSet.has(label.id),
  }));
  const detected = results.filter((result) => result.detected);

  elements.imageScores.replaceChildren(
    ...results.map((result) => {
      const item = document.createElement("li");
      item.className = `image-score${result.detected ? " image-score--detected" : ""}`;
      const comparison = expectedLabelIds
        ? ` Esperada: ${result.expected ? "sí" : "no"}.`
        : "";
      item.textContent = `${capitalize(result.name)}: ${Math.round(result.score * 100)} %. Detectada: ${result.detected ? "sí" : "no"}.${comparison}`;
      return item;
    }),
  );

  elements.expectedLabels.hidden = !expectedLabelIds;
  elements.expectedLabels.textContent = expectedLabelIds
    ? `Etiquetas esperadas: ${formatSpanishList(expectedLabelIds.map(getLabelName))}.`
    : "";
  elements.imageResultsPanel.hidden = false;

  if (detected.length === 0) {
    const message = "No se detectaron elementos por encima de sus umbrales.";
    elements.detection.textContent = message;
    speak(message, { remember: true });
    return;
  }

  elements.detection.textContent = detected
    .map((label) => `${capitalize(label.name)}: ${Math.round(label.score * 100)} por ciento`)
    .join(". ");
  speak(buildDetectionMessage(detected), { remember: true });
}

function getLabelName(labelId) {
  return LABELS.find((label) => label.id === labelId)?.name ?? labelId;
}

function clearUploadedImage() {
  if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
  uploadedImageUrl = "";
  uploadedImageName = "";
}

async function openRearCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  elements.video.srcObject = stream;
  await elements.video.play();
  return stream;
}

async function loadNavigationModel(config) {
  if (modelCache.has(config.id)) {
    return { config, instance: modelCache.get(config.id) };
  }

  if (!modelPromises.has(config.id)) {
    const loading = (async () => {
      let instance;
      if (config.runtime === "onnx") {
        ort.env.wasm.wasmPaths =
          `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_RUNTIME_VERSION}/dist/`;
        ort.env.wasm.numThreads = window.crossOriginIsolated
          ? Math.min(navigator.hardwareConcurrency || 1, 4)
          : 1;
        instance = await ort.InferenceSession.create(config.url, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      } else {
        await tf.ready();
        instance = await tf.loadLayersModel(config.url);
      }

      modelCache.set(config.id, instance);
      return instance;
    })().catch((error) => {
      modelPromises.delete(config.id);
      throw error;
    });
    modelPromises.set(config.id, loading);
  }

  const instance = await modelPromises.get(config.id);
  return { config, instance };
}

function validateModel(loadedModel) {
  const { config, instance } = loadedModel;

  if (config.runtime === "onnx") {
    if (!instance.inputNames.includes(config.inputName)) {
      throw new Error(`INVALID_ONNX_INPUT:${instance.inputNames.join(",")}`);
    }
    if (!instance.outputNames.includes(config.outputName)) {
      throw new Error(`INVALID_ONNX_OUTPUT:${instance.outputNames.join(",")}`);
    }
    return;
  }

  const inputShape = instance.inputs?.[0]?.shape;
  const outputShape = instance.outputs?.[0]?.shape;
  const expectedInput = [null, CONFIG.inputSize, CONFIG.inputSize, 3];

  if (!sameShape(inputShape, expectedInput)) {
    throw new Error(`INVALID_INPUT_SHAPE:${JSON.stringify(inputShape)}`);
  }

  if (!outputShape || outputShape.at(-1) !== LABELS.length) {
    throw new Error(`INVALID_OUTPUT_SHAPE:${JSON.stringify(outputShape)}`);
  }
}

function sameShape(actual, expected) {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

async function warmUpModel(loadedModel) {
  const { config, instance } = loadedModel;

  if (config.runtime === "onnx") {
    const input = new ort.Tensor(
      "float32",
      new Float32Array(CONFIG.inputSize * CONFIG.inputSize * 3),
      [1, CONFIG.inputSize, CONFIG.inputSize, 3],
    );
    const results = await instance.run({ [config.inputName]: input });
    const output = results[config.outputName];
    if (!output || output.data.length !== LABELS.length) {
      throw new Error(`INVALID_OUTPUT_SHAPE:${output?.dims}`);
    }
    return;
  }

  const output = tf.tidy(() => {
    const input = tf.zeros([1, CONFIG.inputSize, CONFIG.inputSize, 3]);
    const prediction = instance.predict(input);
    return Array.isArray(prediction) ? prediction[0] : prediction;
  });

  try {
    await output.data();
  } finally {
    output.dispose();
  }
}

function runInferenceLoop(timestamp) {
  if (!isRunning) return;

  animationFrameId = requestAnimationFrame(runInferenceLoop);

  const enoughTimePassed = timestamp - lastInferenceAt >= CONFIG.inferenceIntervalMs;
  const videoIsReady = elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
  if (!enoughTimePassed || !videoIsReady || isPredicting || document.hidden) return;

  lastInferenceAt = timestamp;
  void predictCurrentFrame(sessionId);
}

async function predictCurrentFrame(predictionSession) {
  isPredicting = true;

  try {
    const scores =
      activeModel.config.runtime === "onnx"
        ? await predictWithOnnx(activeModel, elements.video)
        : await predictWithTfjs(activeModel, elements.video);
    if (!isRunning || predictionSession !== sessionId) return;
    processScores(scores);
  } catch (error) {
    console.error(error);
    if (predictionSession === sessionId) {
      showError("Ocurrió un error al analizar la imagen. Intenta reiniciar la navegación.");
    }
  } finally {
    if (predictionSession === sessionId) isPredicting = false;
  }
}

function createFrameTensor(sourceElement, layout) {
  return tf.tidy(() => {
    const frame = tf.browser.fromPixels(sourceElement);
    let resized = tf.image
      .resizeBilinear(frame, [CONFIG.inputSize, CONFIG.inputSize], false)
      .toFloat();

    // Ambos modelos incluyen Rescaling(1/127.5, offset=-1). Se conservan 0–255.
    if (layout === "NCHW") resized = resized.transpose([2, 0, 1]);
    return resized.expandDims(0);
  });
}

async function predictWithOnnx(loadedModel, sourceElement) {
  const { config, instance } = loadedModel;
  const inputTensor = createFrameTensor(sourceElement, config.layout);

  try {
    const data = await inputTensor.data();
    const dimensions =
      config.layout === "NCHW"
        ? [1, 3, CONFIG.inputSize, CONFIG.inputSize]
        : [1, CONFIG.inputSize, CONFIG.inputSize, 3];
    const input = new ort.Tensor("float32", data, dimensions);
    const results = await instance.run({ [config.inputName]: input });
    const output = results[config.outputName];
    if (!output) throw new Error("INVALID_ONNX_OUTPUT");
    return Array.from(output.data);
  } finally {
    inputTensor.dispose();
  }
}

async function predictWithTfjs(loadedModel, sourceElement) {
  const { config, instance } = loadedModel;
  const input = createFrameTensor(sourceElement, config.layout);
  const prediction = instance.predict(input);
  const output = Array.isArray(prediction) ? prediction[0] : prediction;

  try {
    return Array.from(await output.data());
  } finally {
    input.dispose();
    tf.dispose(prediction);
  }
}

function processScores(scores) {
  if (scores.length !== LABELS.length || scores.some((score) => !Number.isFinite(score))) {
    throw new Error("INVALID_PREDICTION");
  }

  LABELS.forEach((label, index) => {
    if (scores[index] >= label.threshold) {
      positiveStreaks[index] += 1;
      negativeStreaks[index] = 0;
      if (positiveStreaks[index] >= CONFIG.stablePredictionsRequired) {
        activeLabels[index] = true;
      }
    } else {
      positiveStreaks[index] = 0;
      negativeStreaks[index] += 1;
      if (negativeStreaks[index] >= CONFIG.negativePredictionsToClear) {
        activeLabels[index] = false;
      }
    }
  });

  const detected = LABELS.map((label, index) => ({
    ...label,
    score: scores[index],
    active: activeLabels[index],
  })).filter((label) => label.active);

  if (detected.length === 0) {
    elements.detection.textContent = "Buscando elementos de navegación reconocibles…";
    lastAnnouncedSignature = "";
    return;
  }

  elements.detection.textContent = detected
    .map((label) => `${capitalize(label.name)}: ${Math.round(label.score * 100)} por ciento`)
    .join(". ");

  const signature = detected.map((label) => label.id).join("|");
  const cooldownFinished =
    Date.now() - lastAutomaticSpeechAt >= CONFIG.announcementCooldownMs;
  if (signature === lastAnnouncedSignature || !cooldownFinished) return;

  const message = buildDetectionMessage(detected);
  lastAnnouncedSignature = signature;
  lastAutomaticSpeechAt = Date.now();
  speak(message, { remember: true });
}

function buildDetectionMessage(detected) {
  const names = detected.map((label) => label.name);
  const list = formatSpanishList(names);
  const warning = detected.some((label) => label.id === "obstaculo") ? "Atención. " : "";
  return `${warning}En el entorno del segundo piso se detecta: ${list}.`;
}

function formatSpanishList(items) {
  if (items.length <= 1) return items[0] ?? "";
  if (items.length === 2) return `${items[0]} y ${items[1]}`;
  return `${items.slice(0, -1).join(", ")} y ${items.at(-1)}`;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function speak(message, { remember }) {
  if (!("speechSynthesis" in window) || !message) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.lang = CONFIG.speechLanguage;
  utterance.rate = 0.92;
  utterance.pitch = 1;
  utterance.volume = 1;

  const voices = window.speechSynthesis.getVoices();
  const spanishVoice =
    voices.find((voice) => voice.lang.toLowerCase() === CONFIG.speechLanguage.toLowerCase()) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith("es"));
  if (spanishVoice) utterance.voice = spanishVoice;

  window.speechSynthesis.speak(utterance);

  if (remember) {
    lastSpokenMessage = message;
    elements.repeatButton.disabled = false;
  }
}

function repeatLastAnnouncement() {
  if (lastSpokenMessage) speak(lastSpokenMessage, { remember: false });
}

function stopNavigation(announceStop) {
  sessionId += 1;
  isRunning = false;
  isStarting = false;
  isPredicting = false;

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  elements.video.srcObject = null;
  window.speechSynthesis?.cancel();

  resetDetectionState();
  setControls("stopped");
  setStatus("Navegación detenida.");
  elements.detection.textContent = "No se han detectado elementos.";

  if (announceStop) speak("Navegación detenida.", { remember: false });
}

function cleanupPage() {
  stopNavigation(false);
  clearUploadedImage();
}

function resetDetectionState() {
  positiveStreaks = Array(LABELS.length).fill(0);
  negativeStreaks = Array(LABELS.length).fill(0);
  activeLabels = Array(LABELS.length).fill(false);
  lastAnnouncedSignature = "";
  lastAutomaticSpeechAt = 0;
  lastInferenceAt = 0;
}

function getSelectedModelConfig() {
  const config = MODEL_REGISTRY[elements.modelSelect.value];
  if (!config) throw new Error("MODEL_NOT_CONFIGURED");
  return config;
}

function restoreModelSelection() {
  try {
    const savedModelId = window.localStorage.getItem("navigation-model-id");
    if (savedModelId && MODEL_REGISTRY[savedModelId]) {
      elements.modelSelect.value = savedModelId;
    }
  } catch {
    // El almacenamiento local es una mejora opcional; la selección ONNX sigue por defecto.
  }
}

function handleModelSelection() {
  if (isRunning || isStarting || isAnalyzingImage) return;

  activeModel = null;
  resetDetectionState();
  clearError();
  updateModelDescription();

  const config = getSelectedModelConfig();
  setStatus(`Modelo seleccionado: ${config.name}.`);
  try {
    window.localStorage.setItem("navigation-model-id", config.id);
  } catch {
    // La aplicación funciona aunque el navegador bloquee el almacenamiento local.
  }
}

function updateModelDescription() {
  const config = getSelectedModelConfig();
  elements.modelDescription.textContent = config.description;
}

function setControls(state) {
  const stopped = state === "stopped";
  elements.startButton.disabled = !stopped;
  elements.stopButton.disabled = state !== "starting" && state !== "running";
  elements.modelSelect.disabled = !stopped;
  elements.imageUpload.disabled = !stopped;
  elements.testImageSelect.disabled = !stopped || testImages.length === 0;
  refreshImageControlState();
  if (!lastSpokenMessage) elements.repeatButton.disabled = true;
}

function refreshImageControlState() {
  const available = !isRunning && !isStarting && !isAnalyzingImage;
  elements.analyzeUploadButton.disabled = !available || !uploadedImageUrl;
  elements.analyzeTestButton.disabled = !available || !selectedTestImage;
}

function setStatus(message) {
  elements.status.textContent = message;
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = false;
}

function clearError() {
  elements.error.textContent = "";
  elements.error.hidden = true;
}

function friendlyErrorMessage(error) {
  const name = error?.name ?? "";
  const message = String(error?.message ?? error);

  if (name === "NotAllowedError" || name === "SecurityError") {
    return "No se concedió acceso a la cámara. Abre los permisos del navegador y autoriza la cámara.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No se encontró una cámara disponible en este dispositivo.";
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "La cámara está siendo utilizada por otra aplicación. Ciérrala e intenta de nuevo.";
  }
  if (message.includes("INSECURE_CONTEXT")) {
    return "La cámara requiere una conexión segura. Abre la aplicación mediante HTTPS.";
  }
  if (message.includes("CAMERA_UNSUPPORTED")) {
    return "Este navegador no permite acceder a la cámara desde esta página.";
  }
  if (message.includes("SPEECH_UNSUPPORTED")) {
    return "Este navegador no admite síntesis de voz. Usa una versión reciente de Chrome o Safari.";
  }
  if (message.includes("TFJS_UNAVAILABLE")) {
    return "No se pudo cargar TensorFlow.js. Comprueba la conexión a internet e intenta nuevamente.";
  }
  if (message.includes("ONNX_RUNTIME_UNAVAILABLE")) {
    return "No se pudo cargar ONNX Runtime Web. Comprueba la conexión o selecciona TensorFlow.js.";
  }
  if (message.includes("MODEL_NOT_CONFIGURED")) {
    return "El modelo seleccionado no está configurado correctamente.";
  }
  if (
    message.includes("INVALID_INPUT_SHAPE") ||
    message.includes("INVALID_OUTPUT_SHAPE") ||
    message.includes("INVALID_ONNX_INPUT") ||
    message.includes("INVALID_ONNX_OUTPUT")
  ) {
    return "El modelo cargado no tiene la entrada 224 por 224 y cinco salidas esperadas.";
  }
  if (message.includes("INVALID_PREDICTION")) {
    return "El modelo devolvió una predicción no válida para esta imagen.";
  }
  if (/\.onnx|model\.json|fetch|404|load/i.test(message)) {
    return "No se pudo cargar el modelo seleccionado. Comprueba los archivos de la carpeta model e intenta nuevamente.";
  }

  return "No se pudo completar el análisis. Revisa la conexión y los archivos del modelo.";
}
