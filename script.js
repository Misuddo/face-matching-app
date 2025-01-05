// Load Face API models from the models directory
const MODEL_URL = './models';

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
  faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
  faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
]).then(() => {
  console.log("Models loaded successfully.");
});

// DOM elements
const referenceInput = document.getElementById("referenceImage");
const compareInput = document.getElementById("compareImage");
const canvas = document.getElementById("canvas");
const matchResult = document.getElementById("matchResult");

// Event listeners
referenceInput.addEventListener("change", handleReferenceUpload);
compareInput.addEventListener("change", handleCompareUpload);

let referenceDescriptor = null; // Stores reference face descriptor

async function handleReferenceUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const image = await faceapi.bufferToImage(file);
  const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();

  if (!detection) {
    alert("No face detected in the reference image.");
    return;
  }

  referenceDescriptor = detection.descriptor;
  matchResult.textContent = "Reference image uploaded successfully!";
}

async function handleCompareUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  if (!referenceDescriptor) {
    alert("Please upload a reference image first.");
    return;
  }

  const image = await faceapi.bufferToImage(file);
  const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();

  if (!detection) {
    alert("No face detected in the comparison image.");
    return;
  }

  const distance = faceapi.euclideanDistance(referenceDescriptor, detection.descriptor);

  // Draw bounding box on the canvas
  const displaySize = { width: image.width, height: image.height };
  faceapi.matchDimensions(canvas, displaySize);

  const resizedDetection = faceapi.resizeResults(detection, displaySize);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  faceapi.draw.drawDetections(canvas, resizedDetection);

  // Display match result
  matchResult.textContent = `Similarity Score: ${(1 - distance).toFixed(2)}`;
}
