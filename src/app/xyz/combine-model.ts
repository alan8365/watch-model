import {isCanvas, isVideo} from './tfjs-model';
import {
  InputResolution,
  MobileNetMultiplier,
  PoseNetArchitecture,
  PoseNetOutputStride,
  PoseNetQuantBytes
} from '@tensorflow-models/posenet/dist/types';
import {drawPoint, toggleLoadingUI} from './util';
import Stats from 'stats.js';
import * as posenet from '@tensorflow-models/posenet';
import clm from 'clmtrackr';

// Setup start
const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();
const flipPoseHorizontal = false;

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
    minPoseConfidence: 0.05,
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

var TFJSPositions;
var nonCheatingFlag = true;

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
export async function bindPage(videoElementId: string, canvasElementIdTFJS: string, canvasElementIdCLM: string, minPoseConfidence?: number): Promise<void> {
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

  TFJSDetect(video, canvasTFJS, minPoseConfidence);
  CLMDetect(video, canvasTFJS, canvasCLM);
}

/**
 * detect point of face.
 * @param video
 * @constructor
 */
async function TFJSDetect(video: any, canvasTFJS: any, minPoseConfidence: number): Promise<void> {
  if (isCanvas(canvasTFJS)) {
    const ctxTFJS = canvasTFJS.getContext('2d');

    canvasTFJS.width = videoWidth;
    canvasTFJS.height = videoHeight;

    // since images are being fed from a webcam, we want to feed in the
    // original image and then just flip the keypoints' x coordinates. If instead
    // we flip the image, then correcting left-right keypoint pairs requires a
    // permutation on all the keypoints.

    minPoseConfidence = minPoseConfidence || +guiState.singlePoseDetection.minPoseConfidence;
    async function poseDetectionFrame(): Promise<void> {
      // Begin monitoring code for frames per second
      stats.begin();
      // TFJS Start
      // FIXME fps is stuck this line
      const pose = await guiState.net.estimatePoses(video, {
        flipHorizontal: flipPoseHorizontal,
        decodingMethod: 'single-person'
      });

      // const minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      const minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

      ctxTFJS.clearRect(0, 0, videoWidth, videoHeight);


      pose.forEach(({score, keypoints}) => {
        if (score >= minPoseConfidence) {
          TFJSPositions = parseTFJSPosition(keypoints);
        }
      });
      // TFJS End
      requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
  }
}

/**
 *
 * @param video
 * @param canvasTFJS
 * @param canvasCLM
 */
export function CLMDetect(video: any, canvasTFJS: any, canvasCLM: any): void {
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

      // CLM Start
      ctxCLM.clearRect(0, 0, videoWidth, videoHeight);
      // psrElement.innerHTML = "score :" + ctrack.getScore().toFixed(4);
      // ctrack.getCurrentPosition((e) => {
      //   console.log(e);
      //   this.draw(canvasCLM);
      // });

      if (TFJSPositions){
        for(const [key, value] of Object.entries(TFJSPositions)){
          drawPoint(ctxCLM, value['y'], value['x'], 3, 'aqua');
        }

        if (ctrack.getCurrentPosition()) {
          let positions = parseCLMPosition(ctrack.getCurrentPosition());
          let distance = checkAllDistance(positions, TFJSPositions);
          let drawFlag = true;

          for(const [key, value] of Object.entries(distance)){
            if (value > 30){
              drawFlag = false;
            }
          }

          for(const [key, value] of Object.entries(positions)){
            drawPoint(ctxCLM, value['y'], value['x'], 3, 'pink');
          }

          cheatDetect(positions, TFJSPositions, drawFlag);

          if (drawFlag){
            ctrack.draw(canvasCLM);
          }
          // console.log(TFJSPositions);
        }
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
function parseTFJSPosition(positions: any): void {
  positions = {
    nose: positions[0]["position"],
    leftEye: positions[1]["position"],
    rightEye: positions[2]["position"],
    leftEar: positions[3]["position"],
    rightEar: positions[4]["position"],
  };

  return positions;
  // console.log(CLMPositions);
  // console.log(positions);
}

/**
 * Parse position data
 * @param positions
 */
function parseCLMPosition(positions: any): Object {
  positions = {
    nose: {x: positions[62][0], y:positions[62][1]},
    rightEye: {x: positions[27][0], y:positions[27][1]},
    leftEye: {x: positions[32][0], y:positions[32][1]},
    rightEar: {x: positions[1][0], y:positions[1][1]},
    leftEar: {x: positions[13][0], y:positions[13][1]},
  };

  return positions;
}

/**
 * 
 * @param position1 
 * @param position2 
 */
function distanceTwoDim(position1: Object, position2: Object): number{
  let a1 = Math.pow(position1['x'] - position2['x'], 2);
  let a2 = Math.pow(position1['y'] - position2['y'], 2);

  return Math.sqrt(a1 + a2);
}

/**
 * 
 * @param positions1 
 * @param positions2 
 */
function checkAllDistance(positions1: Object, positions2: Object): Object{
  let result = {};

  for(const [key, value] of Object.entries(positions1)){
    result[key] = distanceTwoDim(positions1[key], positions2[key]);
  }

  return result;
}

/**
 * 
 * @param CLMPositions 
 * @param TFJSPositions 
 * @param drawFlag 
 */
function cheatDetect(CLMPositions: Object, TFJSPositions: Object, drawFlag: boolean): void{
  const sensitivity = 25;

  const headTurnLeftFlag = distanceTwoDim(TFJSPositions['leftEye'], TFJSPositions['leftEar']) < sensitivity;
  const headTurnRightFlag = distanceTwoDim(TFJSPositions['rightEye'], TFJSPositions['rightEar']) < sensitivity;

  console.log(nonCheatingFlag);
  if(nonCheatingFlag){
    if(drawFlag){
      // TODO TFJS position combine postions
      const eyeTurnLeftFlag = false;
      const eyeTurnRightFlag = false;

      if(headTurnLeftFlag || headTurnRightFlag || eyeTurnLeftFlag || eyeTurnRightFlag){
        callBackend();
      }
    }else{
      if(headTurnLeftFlag || headTurnRightFlag){
        callBackend();
      }
    }
  }

}

// TODO temp function
async function callBackend(){
  const canvas = document.getElementById('test1');
  const video = document.getElementById("video");
  const bububu = document.getElementById("bububu");
  const timeInterval = 200;
  const snapNumber = 3;

  nonCheatingFlag = false;

  if (isCanvas(canvas) && isVideo(video)){
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext('2d');

    for (let counter = 0; counter < snapNumber; counter++) {
      setTimeout(() => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        let dataURI = canvas.toDataURL('image/jpeg');
        console.log(dataURI);

        let img = document.createElement("img");
        img.setAttribute("src", dataURI);

        bububu.appendChild(img);
      }, timeInterval * counter)
    }

    setTimeout(() => {
      nonCheatingFlag = true;
    }, timeInterval * snapNumber)
  }
  console.log("someone cheat!!!!!!")
}
