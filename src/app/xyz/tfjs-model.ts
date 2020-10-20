import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';
import {drawBoundingBox, drawKeypoints, toggleLoadingUI,} from './util';

import {
  InputResolution,
  MobileNetMultiplier,
  PoseNetArchitecture,
  PoseNetOutputStride,
  PoseNetQuantBytes
} from '@tensorflow-models/posenet/dist/types';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

/**
 * camera load start
 */

/**
 * Loads a the camera to be used in the demo
 */
export async function setupCamera(videoElementId) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // throw new Error(
    //   'Browser API navigator.mediaDevices.getUserMedia not available');
    insertAltVideo(vid);
    document.getElementById('gum').className = 'hide';
    document.getElementById('nogum').className = 'nohide';
    alert('Your browser does not seem to support getUserMedia, using a fallback video instead.');
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

export async function loadVideo(videoElementId) {
  const video = await setupCamera(videoElementId);

  if (isVideo(video)) {
    await video.play();
  }

  return video;
}

function insertAltVideo(video) {
  // insert alternate video if getUserMedia not available
  video.src = './media/cap12_edit.webm';
  // video.src = './media/cap12_edit.mp4';
}

/**
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

function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  let architectureController = null;
  // architectureController.setValue('ResNet50')
  // gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  // updateTryResNetButtonDatGuiCss();

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
    gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  const input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  architectureController =
    input.add(guiState.input, 'architecture', ['MobileNetV1', 'ResNet50']);
  guiState.architecture = guiState.input.architecture;
  // Input resolution:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The higher the value of the input
  // resolution the better the accuracy but slower the speed.
  let inputResolutionController = null;

  function updateGuiInputResolution(
    inputResolution,
    inputResolutionArray,
  ) {
    if (inputResolutionController) {
      inputResolutionController.remove();
    }
    guiState.inputResolution = inputResolution;
    guiState.input.inputResolution = inputResolution;
    inputResolutionController =
      input.add(guiState.input, 'inputResolution', inputResolutionArray);
    inputResolutionController.onChange(function(inputResolution) {
      guiState.changeToInputResolution = inputResolution;
    });
  }

  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  let outputStrideController = null;

  function updateGuiOutputStride(outputStride, outputStrideArray) {
    if (outputStrideController) {
      outputStrideController.remove();
    }
    guiState.outputStride = outputStride;
    guiState.input.outputStride = outputStride;
    outputStrideController =
      input.add(guiState.input, 'outputStride', outputStrideArray);
    outputStrideController.onChange(function(outputStride) {
      guiState.changeToOutputStride = outputStride;
    });
  }

  // Multiplier: this parameter affects the number of feature map channels in
  // the MobileNet. The higher the value, the higher the accuracy but slower the
  // speed, the lower the value the faster the speed but lower the accuracy.
  let multiplierController = null;

  function updateGuiMultiplier(multiplier, multiplierArray) {
    if (multiplierController) {
      multiplierController.remove();
    }
    guiState.multiplier = multiplier;
    guiState.input.multiplier = multiplier;
    multiplierController =
      input.add(guiState.input, 'multiplier', multiplierArray);
    multiplierController.onChange(function(multiplier) {
      guiState.changeToMultiplier = multiplier;
    });
  }

  // QuantBytes: this parameter affects weight quantization in the ResNet50
  // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
  // the value, the larger the model size and thus the longer the loading time,
  // the lower the value, the shorter the loading time but lower the accuracy.
  let quantBytesController = null;

  function updateGuiQuantBytes(quantBytes: PoseNetQuantBytes, quantBytesArray) {
    if (quantBytesController) {
      quantBytesController.remove();
    }
    guiState.quantBytes = +quantBytes;
    // guiState.input.quantBytes = +quantBytes;
    quantBytesController =
      input.add(guiState.input, 'quantBytes', quantBytesArray);
    quantBytesController.onChange(function(quantBytes) {
      guiState.changeToQuantBytes = +quantBytes;
    });
  }

  function updateGui() {
    if (guiState.input.architecture === 'MobileNetV1') {
      updateGuiInputResolution(
        defaultMobileNetInputResolution,
        [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
      updateGuiOutputStride(defaultMobileNetStride, [8, 16]);
      updateGuiMultiplier(defaultMobileNetMultiplier, [0.50, 0.75, 1.0]);
    } else {  // guiState.input.architecture === "ResNet50"
      updateGuiInputResolution(
        defaultResNetInputResolution,
        [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]);
      updateGuiOutputStride(defaultResNetStride, [32, 16]);
      updateGuiMultiplier(defaultResNetMultiplier, [1.0]);
    }
    updateGuiQuantBytes(defaultQuantBytes, [1, 2, 4]);
  }

  updateGui();
  input.open();
  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  const single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  const multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
    .min(1)
    .max(20)
    .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  const output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function(architecture) {
    // if architecture is ResNet50, then show ResNet50 options
    updateGui();
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });
}

/**
 *
 * @param videoElementId
 * @param canvasElementIdTFJS
 * @param canvasElementIdCLM
 */
export async function bindPage(videoElementId: string, canvasElementIdTFJS: string, canvasElementIdCLM: string) {
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
 * detect two model.
 * @param video
 * @param canvasTFJS
 * @param canvasCLM
 */
export function detect(video, canvasTFJS, canvasCLM) {
  if (isCanvas(canvasTFJS) && isCanvas(canvasCLM)) {
    const ctxTFJS = canvasTFJS.getContext('2d');
    const ctxCLM = canvasTFJS.getContext('2d');

    canvasTFJS.width = videoWidth;
    canvasTFJS.height = videoHeight;

    canvasCLM.width = videoWidth;
    canvasCLM.height = videoHeight;


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

      const minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      const minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

      ctxTFJS.clearRect(0, 0, videoWidth, videoHeight);

      if (guiState.output.showVideo) {
        ctxTFJS.save();
        ctxTFJS.scale(-1, 1);
        ctxTFJS.translate(-videoWidth, 0);
        // ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        ctxTFJS.restore();
      }

      pose.forEach(({score, keypoints}) => {
        if (score >= minPoseConfidence) {
          // TODO draw position here
          drawKeypoints(keypoints, minPartConfidence, ctxTFJS);
          // TODO use position info here
          parsePosition(keypoints);
        }
      });

      // End monitoring code for frames per second
      stats.end();

      requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
  }
}

/**
 * Check obj is HTMLVideoElement.
 * @param obj
 */
export function isVideo(obj: HTMLVideoElement | HTMLElement): obj is HTMLVideoElement {
  return obj.tagName === 'VIDEO';
}

/**
 * Check obj is HTMLCanvasElement.
 * @param obj
 */
export function isCanvas(obj: HTMLCanvasElement | HTMLElement): obj is HTMLCanvasElement {
  return obj.tagName === 'CANVAS';
}

/**
 * Parse position data
 * @param positions
 */
function parsePosition(positions) {
  positions = {
    nose: positions[0],
    leftEye: positions[1],
    rightEye: positions[2],
    leftEar: positions[3],
    rightEar: positions[4],
  };

  // console.log(positions);
}
