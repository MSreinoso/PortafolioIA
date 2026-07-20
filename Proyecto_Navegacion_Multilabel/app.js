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
  "tfjs-v1": Object.freeze({
    id: "tfjs-v1",
    name: "TensorFlow.js — modelo actual",
    runtime: "tfjs",
    url: "./model/model.json",
    layout: "NHWC",
    description: "Usa el modelo Keras convertido a model.json como opción de respaldo.",
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

elements.startButton.addEventListener("click", startNavigation);
elements.stopButton.addEventListener("click", () => stopNavigation(true));
elements.repeatButton.addEventListener("click", repeatLastAnnouncement);
elements.modelSelect.addEventListener("change", handleModelSelection);
window.addEventListener("pagehide", () => stopNavigation(false));

restoreModelSelection();
updateModelDescription();

async function startNavigation() {
  if (isRunning || isStarting) return;

  const thisSession = ++sessionId;
  isStarting = true;
  clearError();
  resetDetectionState();
  setControls("starting");
  setStatus("Solicitando acceso a la cámara…");

  // Iniciar la voz desde un gesto del usuario mejora el funcionamiento en iOS.
  speak("Iniciando navegación asistida.", { remember: false });

  try {
    ensureBrowserSupport();
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

function ensureBrowserSupport() {
  if (!window.isSecureContext) {
    throw new Error("INSECURE_CONTEXT");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("CAMERA_UNSUPPORTED");
  }
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
        ? await predictWithOnnx(activeModel)
        : await predictWithTfjs(activeModel);
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

function createFrameTensor(layout) {
  return tf.tidy(() => {
    const frame = tf.browser.fromPixels(elements.video);
    let resized = tf.image
      .resizeBilinear(frame, [CONFIG.inputSize, CONFIG.inputSize], false)
      .toFloat();

    // Ambos modelos incluyen Rescaling(1/127.5, offset=-1). Se conservan 0–255.
    if (layout === "NCHW") resized = resized.transpose([2, 0, 1]);
    return resized.expandDims(0);
  });
}

async function predictWithOnnx(loadedModel) {
  const { config, instance } = loadedModel;
  const inputTensor = createFrameTensor(config.layout);

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

async function predictWithTfjs(loadedModel) {
  const { config, instance } = loadedModel;
  const input = createFrameTensor(config.layout);
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
  if (isRunning || isStarting) return;

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
  elements.stopButton.disabled = stopped;
  elements.modelSelect.disabled = !stopped;
  if (!lastSpokenMessage) elements.repeatButton.disabled = true;
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
  if (/\.onnx|model\.json|fetch|404|load/i.test(message)) {
    return "No se pudo cargar el modelo seleccionado. Comprueba los archivos de la carpeta model o elige el modelo de respaldo.";
  }

  return "No se pudo iniciar la navegación. Revisa la cámara, la conexión y los archivos del modelo.";
}
