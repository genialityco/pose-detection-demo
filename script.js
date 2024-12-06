import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");
let poseLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "660px";
const videoWidth = "780px";

// Cargar imagen
const gloveImage = new Image();
gloveImage.src = "glove.png";

// Create the pose landmarker
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU",
    },
    runningMode,
    numPoses: 1,
  });
  demosSection.classList.remove("invisible");
};
createPoseLandmarker();

// Ball object
const ball = {
  x: 240, // Starting position (centered)
  y: 180,
  radius: 20,
  vx: (Math.random() * 2 - 1) * 5, // Randomized velocity
  vy: (Math.random() * 2 - 1) * 5,
};

// Video and canvas elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Enable webcam
if (navigator.mediaDevices?.getUserMedia) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
}

function enableCam() {
  if (!poseLandmarker) {
    console.log("PoseLandmarker not loaded yet.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE WEBCAM";
    const stream = video.srcObject;
    const tracks = stream?.getTracks();
    if (tracks) {
      tracks.forEach((track) => track.stop());
    }
    video.srcObject = null;
    return;
  }

  const constraints = {
    video: {
      width: 1280,
      height: 720,
    },
  };

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
      webcamRunning = true;
      enableWebcamButton.innerText = "DISABLE WEBCAM";
    })
    .catch((err) => {
      console.error("Error accessing webcam:", err);
    });
}

// Draw the ball on the canvas
function drawBall() {
  canvasCtx.beginPath();
  canvasCtx.arc(ball.x, ball.y, ball.radius, 0, 2 * Math.PI);
  canvasCtx.fillStyle = "#FF0000"; // Red ball
  canvasCtx.fill();
  canvasCtx.closePath();
}

// Update the ball's position
function updateBall() {
  ball.x += ball.vx;
  ball.y += ball.vy;

  // Bounce off the canvas edges
  if (ball.x + ball.radius > canvasElement.width || ball.x - ball.radius < 0) {
    ball.vx = -ball.vx;
  }
  if (ball.y + ball.radius > canvasElement.height || ball.y - ball.radius < 0) {
    ball.vy = -ball.vy;
  }
}

// Dibujar guantes sobre las manos
function drawGloves(landmarks) {
  if (!gloveImage.complete) return; // Asegurarse de que la imagen está cargada

  const LEFT_WRIST = 15;
  const RIGHT_WRIST = 16;

  // Dibujar guantes en las muñecas
  if (landmarks[LEFT_WRIST]) {
    const leftWrist = landmarks[LEFT_WRIST];
    canvasCtx.drawImage(
      gloveImage,
      leftWrist.x * canvasElement.width - 25,
      leftWrist.y * canvasElement.height - 25,
      50,
      50
    );
  }

  if (landmarks[RIGHT_WRIST]) {
    const rightWrist = landmarks[RIGHT_WRIST];
    canvasCtx.drawImage(
      gloveImage,
      rightWrist.x * canvasElement.width - 25,
      rightWrist.y * canvasElement.height - 25,
      50,
      50
    );
  }
}

// Check for interaction with landmarks
function checkInteraction(landmarks) {
  for (const landmark of landmarks) {
    const dx = ball.x - landmark.x * canvasElement.width;
    const dy = ball.y - landmark.y * canvasElement.height;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < ball.radius) {
      // Reverse direction and increase speed slightly
      ball.vx = -ball.vx * 1.05;
      ball.vy = -ball.vy * 1.05;

      // Optionally, change ball color on impact
      canvasCtx.fillStyle = `#${Math.floor(Math.random() * 16777215).toString(
        16
      )}`;
    }
  }
}

// Predict poses and update the canvas
let lastFrameTime = 0;
async function predictWebcam(timestamp) {
  if (!webcamRunning) return;

  // Limit frame rate to 30 FPS
  if (timestamp - lastFrameTime < 33) {
    requestAnimationFrame(predictWebcam);
    return;
  }
  lastFrameTime = timestamp;

  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  if (video.readyState >= 2) {
    const startTimeMs = performance.now();
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      // Update and draw ball
      updateBall();
      drawBall();

      // Draw pose landmarks and check interaction
      if (result?.landmarks?.length) {
        for (const landmark of result.landmarks) {
          drawingUtils.drawLandmarks(landmark, {
            radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
          });
          drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        }

        // Check interaction with key points (hands, feet, etc.)
        const keyLandmarks = result.landmarks[0]; // Assuming single pose
        drawGloves(keyLandmarks);
        checkInteraction(keyLandmarks);
      }
    });
  }

  if (webcamRunning) {
    requestAnimationFrame(predictWebcam);
  }
}
