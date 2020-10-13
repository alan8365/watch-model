import {isCanvas, isVideo} from "./tfjs-model";
import {
  InputResolution,
  MobileNetMultiplier,
  PoseNetArchitecture,
  PoseNetOutputStride,
  PoseNetQuantBytes
} from "@tensorflow-models/posenet/dist/types";
import {drawBoundingBox, drawKeypoints, toggleLoadingUI} from "./util";
import Stats from 'stats.js';
import * as posenet from "@tensorflow-models/posenet";

// Setup start

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();
const flipPoseHorizontal = true;

/**
 net start
 */
const defaultQuantBytes: PoseNetQuantBytes = 2;

const defaultArchitecture: PoseNetArchitecture = 'MobileNetV1'
const defaultMobileNetMultiplier: MobileNetMultiplier = 0.75;
const defaultMobileNetStride: PoseNetOutputStride = 16;
const defaultMobileNetInputResolution: InputResolution = 500;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const guiState = {
  algorithm: 'single-pose',
  input: {
    architecture: defaultArchitecture,
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 1,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: false,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
  camera: undefined,
  architecture: undefined,
  inputResolution: undefined,
  changeToInputResolution: undefined,
  outputStride: undefined,
  multiplier: undefined,
  changeToOutputStride: undefined,
  changeToMultiplier: undefined,
  quantBytes: undefined,
  changeToQuantBytes: undefined,
  changeToArchitecture: undefined
};

/**
 * Loads a the camera to be used in the demo
 * @param videoElementId
 */
export async function setupCamera(videoElementId) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById(videoElementId);

  if (isVideo(video)) {
    video.width = videoWidth;
    video.height = videoHeight;

    video.srcObject = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': {
        facingMode: 'user',
        width: videoWidth,
        height: videoHeight,
      },
    });
  }

  // return new Promise((resolve) => {
  //   video.onloadedmetadata = () => {
  //     resolve(video);
  //   };
  // });

  return video;
}

/**
 * Loads a the camera to be used in the demo
 * @param videoElementId
 */
export async function loadVideo(videoElementId) {
  const video = await setupCamera(videoElementId);

  if (isVideo(video)) {
    await video.play();
  }

  return video;
}

// Setup end

/**
 * Bind page with two model.
 * @param videoElementId
 * @param canvasElementId
 */
export async function bindPage(videoElementId: string, canvasElementId: string) {
  toggleLoadingUI(true);

  const net = await posenet.load({
    architecture: guiState.input.architecture,
    outputStride: guiState.input.outputStride,
    inputResolution: guiState.input.inputResolution,
    multiplier: guiState.input.multiplier,
    quantBytes: guiState.input.quantBytes
  });

  toggleLoadingUI(false);

  let video;
  let canvas = document.getElementById(canvasElementId);

  try {
    video = await loadVideo(videoElementId);
  } catch (e) {
    // FIXME general
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  guiState.net = net;
  detect(video, canvas);
}

/**
 *
 * @param video
 * @param canvas
 */
export function detect(video, canvas) {
  if (isCanvas(canvas)) {
    const ctx = canvas.getContext('2d');

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    // since images are being fed from a webcam, we want to feed in the
    // original image and then just flip the keypoints' x coordinates. If instead
    // we flip the image, then correcting left-right keypoint pairs requires a
    // permutation on all the keypoints.
    const flipPoseHorizontal = true;

    async function poseDetectionFrame() {
      // Begin monitoring code for frames per second
      stats.begin();

      const pose = await guiState.net.estimatePoses(video, {
        flipHorizontal: flipPoseHorizontal,
        decodingMethod: 'single-person'
      });

      let minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      let minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

      ctx.clearRect(0, 0, videoWidth, videoHeight);

      if (guiState.output.showVideo) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
        // ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        ctx.restore();
      }

      pose.forEach(({score, keypoints}) => {
        if (score >= minPoseConfidence) {
          // TODO draw position here
          drawKeypoints(keypoints, minPartConfidence, ctx);
          // TODO use position info here
          // parsePosition(keypoints)
        }
      });

      // End monitoring code for frames per second
      stats.end();

      requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
  }
}
