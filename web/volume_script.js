const audioPlayer = document.getElementById('audioPlayer');
audioPlayer.volume = 0;

// --- Main App Logic ---
const canvases = [document.getElementById('canvas1'), document.getElementById('canvas2'), document.getElementById('canvas3')];
const contexts = canvases.map(c => c.getContext('2d'));
const predictions = [document.getElementById('pred1'), document.getElementById('pred2'), document.getElementById('pred3')];
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const volumeFill = document.getElementById('volume-bar-fill');
const volumeValue = document.getElementById('volume-value');
let ortSession;
let isDrawing = false;
let activeCanvasIndex = -1;

function preprocessCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const alpha = data[(y * canvas.width + x) * 4 + 3];
        if (alpha > 0) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    if (maxX === -1) { return new Float32Array(28 * 28); }
    const boxWidth = maxX - minX, boxHeight = maxY - minY, maxDim = Math.max(boxWidth, boxHeight);
    const padding = 40, newSize = maxDim + padding;
    const centerCanvas = document.createElement('canvas');
    centerCanvas.width = newSize; centerCanvas.height = newSize;
    const centerCtx = centerCanvas.getContext('2d');
    centerCtx.fillStyle = 'black'; centerCtx.fillRect(0, 0, newSize, newSize);
    const pasteX = (newSize - boxWidth) / 2, pasteY = (newSize - boxHeight) / 2;
    centerCtx.drawImage(canvas, minX, minY, boxWidth, boxHeight, pasteX, pasteY, boxWidth, boxHeight);
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 28; finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.drawImage(centerCanvas, 0, 0, 28, 28);
    const finalImageData = finalCtx.getImageData(0, 0, 28, 28);
    const float32Data = new Float32Array(28 * 28);
    for (let i = 0; i < finalImageData.data.length; i += 4) {
      const grayscale = finalImageData.data[i] / 255;
      float32Data[i / 4] = grayscale * 2 - 1;
    }
    return float32Data;
}

function startDrawing(e, index) { isDrawing = true; activeCanvasIndex = index; draw(e); }
function stopDrawing() { if (isDrawing) { contexts[activeCanvasIndex].beginPath(); isDrawing = false; activeCanvasIndex = -1; } }
function draw(e) {
  if (!isDrawing || activeCanvasIndex === -1) return;
  const canvas = canvases[activeCanvasIndex];
  const ctx = contexts[activeCanvasIndex];
  const rect = canvas.getBoundingClientRect();
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}
function isCanvasBlank(canvas) {
  const context = canvas.getContext('2d');
  const p = new Uint32Array(context.getImageData(0, 0, canvas.width, canvas.height).data.buffer);
  return !p.some(pixel => pixel !== 0);
}
async function runPrediction() {
  if (!ortSession) { alert("Model is not loaded yet."); return; }
  for (let i = 0; i < canvases.length; i++) {
    if (isCanvasBlank(canvases[i])) {
      alert(`Please draw a digit in box #${i + 1}.`);
      return;
    }
  }

  try {
    const finalPredictions = [];

    for (let i = 0; i < canvases.length; i++) {
      const canvasData = preprocessCanvas(canvases[i]);
      const inputTensor = new ort.Tensor('float32', canvasData, [1, 1, 28, 28]);

      const feeds = { input: inputTensor };
      const results = await ortSession.run(feeds);
      const digitLogits = results.output.data;  // 10 outputs
      const predictedIndex = digitLogits.indexOf(Math.max(...digitLogits));

      finalPredictions.push(predictedIndex);
      predictions[i].textContent = predictedIndex;
    }

    let volumeStr = finalPredictions.join('');
    let volumeNum = parseInt(volumeStr, 10);
    if (volumeNum > 100) { volumeNum = 100; volumeStr = '100'; }
    volumeValue.textContent = volumeStr.padStart(3, '0');
    volumeFill.style.height = `${volumeNum}%`;
    audioPlayer.volume = volumeNum / 100;

  } catch (error) {
    console.error("Error during inference:", error);
    alert("An error occurred during prediction.");
  }
}

async function main() {
  try {
    ortSession = await ort.InferenceSession.create('./mnist_cnn.onnx', { executionProviders: ['wasm'] });
    predictBtn.disabled = false; predictBtn.textContent = 'Predict';
    console.log("ONNX model loaded successfully.");
  } catch (error) {
    console.error("Failed to load ONNX model:", error);
    alert("Failed to load model.");
  }
}
canvases.forEach((c, i) => { c.addEventListener('mousedown', (e) => startDrawing(e, i)); c.addEventListener('mousemove', draw); });
window.addEventListener('mouseup', stopDrawing);
predictBtn.addEventListener('click', runPrediction);
clearBtn.addEventListener('click', () => { contexts.forEach(ctx => ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)); predictions.forEach(p => p.textContent = '-'); volumeValue.textContent = '000'; volumeFill.style.height = '0%'; audioPlayer.volume = 0; });
predictBtn.disabled = true; predictBtn.textContent = 'Loading Model...';
main();

// --- Info Overlay & Audio Logic ---
(() => {
  const overlay = document.getElementById('infoOverlay');
  const nextBtn = document.getElementById('nextInfoBtn');
  const prevBtn = document.getElementById('prevInfoBtn');
  const slides = document.querySelectorAll('.info-slide');
  const dotsContainer = document.getElementById('infoDots');
  const uploadAudioBtn = document.getElementById('uploadAudioBtn');
  const defaultAudioBtn = document.getElementById('defaultAudioBtn');
  const audioFileInput = document.getElementById('audioFileInput');
  const trackInfo = document.getElementById('track-info');

  if (!overlay) return;

  let currentIndex = 0;
  const totalSlides = slides.length;

  for (let i = 0; i < totalSlides; i++) {
    const dot = document.createElement('div');
    dot.classList.add('info-dot');
    dotsContainer.appendChild(dot);
  }
  const dots = document.querySelectorAll('.info-dot');

  function updateSlides() {
    slides.forEach((s, i) => s.classList.toggle('active', i === currentIndex));
    dots.forEach((d, i) => d.classList.toggle('active', i === currentIndex));
    prevBtn.disabled = currentIndex === 0;
    if (currentIndex === totalSlides - 1) {
      nextBtn.textContent = 'Exit';
    } else {
      nextBtn.textContent = 'Next →';
    }
  }

  function closeOverlay() { overlay.classList.remove('show'); }

  function loadAudio(file, name) {
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    trackInfo.textContent = `Now Playing: ${name}`;
    audioPlayer.play();
  }

  uploadAudioBtn.addEventListener('click', () => audioFileInput.click());
  audioFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) { loadAudio(file, file.name); }
  });

  defaultAudioBtn.addEventListener('click', async () => {
    try {
      const response = await fetch('../assets/audio/default-music.mp3');
      const blob = await response.blob();
      const file = new File([blob], 'default-music.mp3', {type: 'audio/mpeg'});
      loadAudio(file, 'Space Atmosphere');
    } catch(e) {
      alert('Could not load default audio.');
      console.error(e);
    }
  });

  nextBtn.addEventListener('click', () => {
    if (currentIndex === totalSlides - 1) { closeOverlay(); }
    else { currentIndex++; updateSlides(); }
  });
  prevBtn.addEventListener('click', () => { if (currentIndex > 0) { currentIndex--; updateSlides(); } });

  updateSlides();
})();

// --- Fullscreen Image Modal ---
const cnnImage = document.querySelector('.info-slide img');
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const closeImageModal = document.getElementById('closeImageModal');

if (cnnImage) {
  cnnImage.style.cursor = "pointer";
  cnnImage.addEventListener('click', () => {
    modalImage.src = cnnImage.src;
    imageModal.style.display = "block";
  });
}

closeImageModal.addEventListener('click', () => {
  imageModal.style.display = "none";
});

window.addEventListener('click', (e) => {
  if (e.target === imageModal) {
    imageModal.style.display = "none";
  }
});
