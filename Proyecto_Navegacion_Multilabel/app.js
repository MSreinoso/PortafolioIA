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

const CONFIG = Object.freeze({
  modelUrl: "./model/model.json",
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
  status: document.querySelector("#status"),
  detection: document.querySelector("#detection"),
  error: document.querySelector("#error"),
};

let model = null;
let modelPromise = null;
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
window.addEventListener("pagehide", () => stopNavigation(false));

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

    setStatus("Cargando el modelo de inteligencia artificial…");
    model = await loadNavigationModel();
    if (thisSession !== sessionId) return;

    validateModel(model);
    await warmUpModel(model);
    if (thisSession !== sessionId) return;

    isStarting = false;
    isRunning = true;
    setControls("running");
    setStatus("Navegación activa. Analizando el entorno.");
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

async function loadNavigationModel() {
  if (model) return model;

  if (!modelPromise) {
    modelPromise = (async () => {
      await tf.ready();
      return tf.loadLayersModel(CONFIG.modelUrl);
    })().catch((error) => {
      modelPromise = null;
      throw error;
    });
  }

  return modelPromise;
}

function validateModel(loadedModel) {
  const inputShape = loadedModel.inputs?.[0]?.shape;
  const outputShape = loadedModel.outputs?.[0]?.shape;
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
  const output = tf.tidy(() => {
    const input = tf.zeros([1, CONFIG.inputSize, CONFIG.inputSize, 3]);
    const prediction = loadedModel.predict(input);
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
  let output = null;

  try {
    output = tf.tidy(() => {
      const frame = tf.browser.fromPixels(elements.video);
      const resized = tf.image.resizeBilinear(
        frame,
        [CONFIG.inputSize, CONFIG.inputSize],
        false,
      );

      // El modelo .h5 ya incluye Rescaling(1/127.5, offset=-1).
      // Por ello se entregan píxeles float en 0–255, sin normalizar otra vez aquí.
      const input = resized.toFloat().expandDims(0);
      const prediction = model.predict(input);
      return Array.isArray(prediction) ? prediction[0] : prediction;
    });

    const scores = Array.from(await output.data());
    if (!isRunning || predictionSession !== sessionId) return;
    processScores(scores);
  } catch (error) {
    console.error(error);
    if (predictionSession === sessionId) {
      showError("Ocurrió un error al analizar la imagen. Intenta reiniciar la navegación.");
    }
  } finally {
    output?.dispose();
    if (predictionSession === sessionId) isPredicting = false;
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

function setControls(state) {
  const stopped = state === "stopped";
  elements.startButton.disabled = !stopped;
  elements.stopButton.disabled = stopped;
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
  if (message.includes("INVALID_INPUT_SHAPE") || message.includes("INVALID_OUTPUT_SHAPE")) {
    return "El modelo cargado no tiene la entrada 224 por 224 y cinco salidas esperadas.";
  }
  if (/model\.json|fetch|404|load/i.test(message)) {
    return "No se pudo cargar el modelo. Comprueba que model.json y todos los archivos .bin estén dentro de la carpeta model.";
  }

  return "No se pudo iniciar la navegación. Revisa la cámara, la conexión y los archivos del modelo.";
}
