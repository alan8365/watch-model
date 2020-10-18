import {isCanvas, isVideo} from './tfjs-model';
import {
  InputResolution,
  MobileNetMultiplier,
  PoseNetArchitecture,
  PoseNetOutputStride,
  PoseNetQuantBytes
} from '@tensorflow-models/posenet/dist/types';
import {drawBoundingBox, drawKeypoints, toggleLoadingUI} from './util';
import Stats from 'stats.js';
import * as posenet from '@tensorflow-models/posenet';
import clm from 'clmtrackr';

// Setup start

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();
const flipPoseHorizontal = true;

/*
 net start
 */
const defaultQuantBytes: PoseNetQuantBytes = 2;

const defaultArchitecture: PoseNetArchitecture = 'MobileNetV1';
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


const trackingStarted = false;
const ctrack = new clm.tracker({
  faceDetection: {
    useWebWorkers: false,
  },
});
ctrack.init();

/**
 * Loads a the camera to be used in the demo
 * @param videoElementId
 */
export async function setupCamera(videoElementId: any): Promise<HTMLElement> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById(videoElementId);

  if (isVideo(video)) {
    video.width = videoWidth;
    video.height = videoHeight;

    video.srcObject = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
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
export async function loadVideo(videoElementId: any): Promise<HTMLElement> {
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
 * @param canvasElementIdTFJS
 * @param canvasElementIdCLM
 */
export async function bindPage(videoElementId: string, canvasElementIdTFJS: string, canvasElementIdCLM: string): Promise<void> {
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
  const canvasTFJS = document.getElementById(canvasElementIdTFJS);
  const canvasCLM = document.getElementById(canvasElementIdCLM);

  try {
    video = await loadVideo(videoElementId);
  } catch (e) {
    // FIXME general
    const info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  guiState.net = net;
  detect(video, canvasTFJS, canvasCLM);
}

/**
 *
 * @param video
 * @param canvasTFJS
 * @param canvasCLM
 */
export function detect(video: any, canvasTFJS: any, canvasCLM: any): void {
  if (isCanvas(canvasTFJS) && isCanvas(canvasCLM)) {
    const ctxTFJS = canvasTFJS.getContext('2d');
    const ctxCLM = canvasCLM.getContext('2d');

    canvasTFJS.width = videoWidth;
    canvasTFJS.height = videoHeight;

    canvasCLM.width = videoWidth;
    canvasCLM.height = videoHeight;

    // since images are being fed from a webcam, we want to feed in the
    // original image and then just flip the keypoints' x coordinates. If instead
    // we flip the image, then correcting left-right keypoint pairs requires a
    // permutation on all the keypoints.

    ctrack.start(video);

    async function poseDetectionFrame(): Promise<void> {
      // Begin monitoring code for frames per second
      stats.begin();

      // TFJS Start
      // FIXME fps is stuck this line
      const pose = await guiState.net.estimatePoses(video, {
        flipHorizontal: flipPoseHorizontal,
        decodingMethod: 'single-person'
      });

      const minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      const minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

      // ctxTFJS.clearRect(0, 0, videoWidth, videoHeight);
      //
      // if (guiState.output.showVideo) {
      //   ctxTFJS.save();
      //   ctxTFJS.scale(-1, 1);
      //   ctxTFJS.translate(-videoWidth, 0);
      //   // ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      //   ctxTFJS.restore();
      // }

      pose.forEach(({score, keypoints}) => {
        if (score >= minPoseConfidence) {
          // TODO draw position here
          // drawKeypoints(keypoints, minPartConfidence, ctxTFJS);
          // TODO use position info here
          parsePosition(keypoints);
        }
      });
      // TFJS End

      // CLM Start
      ctxCLM.clearRect(0, 0, videoWidth, videoHeight);
      // psrElement.innerHTML = "score :" + ctrack.getScore().toFixed(4);
      // ctrack.getCurrentPosition((e) => {
      //   console.log(e);
      //   this.draw(canvasCLM);
      // });
      if (ctrack.getCurrentPosition()) {
        console.log(ctrack);
        ctrack.draw(canvasCLM);
      }
      // CLM End

      // End monitoring code for frames per second
      stats.end();

      requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
  }
}

/**
 * Parse position data
 * @param positions
 */
function parsePosition(positions: any): void {
  positions = {
    nose: positions[0],
    leftEye: positions[1],
    rightEye: positions[2],
    leftEar: positions[3],
    rightEar: positions[4],
  };

  // console.log(positions);
}
