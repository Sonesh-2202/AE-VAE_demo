/**
 * AE & VAE Interactive Demo — Main Application
 * ==============================================
 * Loads ONNX models into the browser and provides:
 * 1. Drawing canvas for user input
 * 2. AE and VAE reconstruction comparison
 * 3. VAE latent space exploration
 */

// ─── GLOBALS ────────────────────────────────────────────────
let aeSession = null;
let vaeSession = null;
let vaeDecoderSession = null;
let isDrawing = false;
let hasDrawn = false;

const CANVAS_SIZE = 280;
const MODEL_SIZE = 28;
const LATENT_RANGE = 3.0;

// ─── INITIALIZATION ─────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
    setupDrawingCanvas();
    setupButtons();
    setupLatentExplorer();
    await loadModels();
});

// ─── MODEL LOADING ──────────────────────────────────────────
async function loadModelFromURL(url) {
    // Fetch model as ArrayBuffer — more reliable than passing URL to ort
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
    const buffer = await response.arrayBuffer();
    return ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
    });
}

async function loadModels() {
    const overlay = document.getElementById("loadingOverlay");
    const loadText = overlay.querySelector(".loading-text");
    const loadSub = overlay.querySelector(".loading-subtext");
    const runBtn = document.getElementById("runBtn");

    try {
        loadText.textContent = "Loading neural networks...";
        loadSub.textContent = "Downloading model files (~2 MB total)";

        const basePath = "./models";

        // Load models sequentially to avoid overwhelming the browser
        loadSub.textContent = "Loading Autoencoder (1/3)...";
        aeSession = await loadModelFromURL(`${basePath}/autoencoder.onnx`);

        loadSub.textContent = "Loading VAE (2/3)...";
        vaeSession = await loadModelFromURL(`${basePath}/vae.onnx`);

        loadSub.textContent = "Loading VAE Decoder (3/3)...";
        vaeDecoderSession = await loadModelFromURL(`${basePath}/vae_decoder.onnx`);

        runBtn.disabled = false;
        overlay.classList.add("hidden");
        console.log("✅ All models loaded successfully!");
    } catch (err) {
        console.error("Model loading failed:", err);
        loadText.textContent = "Failed to load models";
        loadSub.textContent = err.message;
    }
}

// ─── DRAWING CANVAS ─────────────────────────────────────────
function setupDrawingCanvas() {
    const canvas = document.getElementById("drawCanvas");
    const ctx = canvas.getContext("2d");
    const guide = canvas.parentElement.querySelector(".canvas-guide");

    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    clearCanvas(canvas, ctx);

    let lastX = 0, lastY = 0;

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = CANVAS_SIZE / rect.width;
        const scaleY = CANVAS_SIZE / rect.height;
        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY,
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPos(e);
        lastX = pos.x;
        lastY = pos.y;
        if (!hasDrawn) {
            hasDrawn = true;
            guide.classList.add("hidden");
        }
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        const brushSize = parseInt(document.getElementById("brushSize").value);
        ctx.beginPath();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = brushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDraw() { isDrawing = false; }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mouseleave", stopDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDraw);
}

function clearCanvas(canvas, ctx) {
    ctx = ctx || canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// ─── BUTTONS & CONTROLS ────────────────────────────────────
function setupButtons() {
    document.getElementById("runBtn").addEventListener("click", runInference);

    document.getElementById("clearBtn").addEventListener("click", () => {
        const canvas = document.getElementById("drawCanvas");
        clearCanvas(canvas);
        hasDrawn = false;
        canvas.parentElement.querySelector(".canvas-guide").classList.remove("hidden");
        ["aeOutput", "vaeOutput"].forEach((id) => {
            const c = document.getElementById(id);
            clearCanvas(c);
            c.parentElement.querySelector(".output-placeholder").classList.remove("hidden");
        });
    });

    document.getElementById("uploadInput").addEventListener("change", handleImageUpload);
}

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
        const canvas = document.getElementById("drawCanvas");
        const ctx = canvas.getContext("2d");
        clearCanvas(canvas, ctx);
        const scale = Math.min(CANVAS_SIZE / img.width, CANVAS_SIZE / img.height) * 0.8;
        const w = img.width * scale;
        const h = img.height * scale;
        ctx.drawImage(img, (CANVAS_SIZE - w) / 2, (CANVAS_SIZE - h) / 2, w, h);
        hasDrawn = true;
        canvas.parentElement.querySelector(".canvas-guide").classList.add("hidden");
        runInference();
    };
    img.src = URL.createObjectURL(file);
    e.target.value = "";
}

// ─── INFERENCE ──────────────────────────────────────────────
async function runInference() {
    if (!aeSession || !vaeSession || !hasDrawn) return;

    const runBtn = document.getElementById("runBtn");
    runBtn.disabled = true;
    runBtn.querySelector(".btn-icon").textContent = "⏳";

    try {
        const inputTensor = preprocessCanvas("drawCanvas");

        // Run AE
        const aeResult = await aeSession.run({ input: inputTensor });
        renderOutput("aeOutput", aeResult.output.data);

        // Run VAE
        const vaeResult = await vaeSession.run({ input: inputTensor });
        renderOutput("vaeOutput", vaeResult.output.data);

        document.querySelectorAll(".output-placeholder").forEach((el) => el.classList.add("hidden"));
    } catch (err) {
        console.error("Inference failed:", err);
    }

    runBtn.disabled = false;
    runBtn.querySelector(".btn-icon").textContent = "⚡";
}

function preprocessCanvas(canvasId) {
    const canvas = document.getElementById(canvasId);
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = MODEL_SIZE;
    tempCanvas.height = MODEL_SIZE;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(canvas, 0, 0, MODEL_SIZE, MODEL_SIZE);
    const imageData = tempCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const float32 = new Float32Array(MODEL_SIZE * MODEL_SIZE);
    for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
        const r = imageData.data[i * 4];
        const g = imageData.data[i * 4 + 1];
        const b = imageData.data[i * 4 + 2];
        float32[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
    }
    return new ort.Tensor("float32", float32, [1, 1, MODEL_SIZE, MODEL_SIZE]);
}

function renderOutput(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = MODEL_SIZE;
    tempCanvas.height = MODEL_SIZE;
    const tempCtx = tempCanvas.getContext("2d");
    const imageData = tempCtx.createImageData(MODEL_SIZE, MODEL_SIZE);

    for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
        const val = Math.max(0, Math.min(1, data[i])) * 255;
        imageData.data[i * 4] = val;
        imageData.data[i * 4 + 1] = val;
        imageData.data[i * 4 + 2] = val;
        imageData.data[i * 4 + 3] = 255;
    }

    tempCtx.putImageData(imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

// ─── LATENT SPACE EXPLORER ──────────────────────────────────
function setupLatentExplorer() {
    const canvas = document.getElementById("latentCanvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 400;
    canvas.height = 400;

    ctx.fillStyle = "#0a0a12";
    ctx.fillRect(0, 0, 400, 400);
    drawLatentGrid(ctx);

    let exploring = false;

    canvas.addEventListener("mousedown", (e) => { exploring = true; exploreLatent(e, canvas); });
    canvas.addEventListener("mousemove", (e) => { if (exploring) exploreLatent(e, canvas); });
    canvas.addEventListener("mouseup", () => (exploring = false));
    canvas.addEventListener("mouseleave", () => (exploring = false));
    canvas.addEventListener("touchstart", (e) => { e.preventDefault(); exploring = true; exploreLatent(e, canvas); }, { passive: false });
    canvas.addEventListener("touchmove", (e) => { e.preventDefault(); if (exploring) exploreLatent(e, canvas); }, { passive: false });
    canvas.addEventListener("touchend", () => (exploring = false));
}

function drawLatentGrid(ctx) {
    ctx.strokeStyle = "rgba(79, 125, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 400; i += 40) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 400); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(400, i); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.beginPath(); ctx.moveTo(200, 0); ctx.lineTo(200, 400); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, 200); ctx.lineTo(400, 200); ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText("0", 204, 196);
    ctx.fillText(`-${LATENT_RANGE}`, 4, 196);
    ctx.fillText(`+${LATENT_RANGE}`, 364, 196);
    ctx.fillText(`+${LATENT_RANGE}`, 204, 14);
    ctx.fillText(`-${LATENT_RANGE}`, 204, 396);
}

async function exploreLatent(e, canvas) {
    if (!vaeDecoderSession) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const px = (clientX - rect.left) / rect.width;
    const py = (clientY - rect.top) / rect.height;

    const z1 = (px - 0.5) * 2 * LATENT_RANGE;
    const z2 = (0.5 - py) * 2 * LATENT_RANGE;

    const crosshair = document.getElementById("crosshair");
    crosshair.style.left = `${px * 100}%`;
    crosshair.style.top = `${py * 100}%`;

    document.getElementById("latentCoords").textContent = `z = [${z1.toFixed(2)}, ${z2.toFixed(2)}]`;

    try {
        const latentTensor = new ort.Tensor("float32", new Float32Array([z1, z2]), [1, 2]);
        const result = await vaeDecoderSession.run({ latent: latentTensor });
        renderOutput("explorerOutput", result.output.data);
    } catch (err) {
        console.error("Latent exploration failed:", err);
    }
}
