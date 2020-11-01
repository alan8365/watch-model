import {
  InputResolution,
  MobileNetMultiplier,
  PoseNetArchitecture,
  PoseNetOutputStride,
  PoseNetQuantBytes
} from '@tensorflow-models/posenet/dist/types';
import { isCanvas, isVideo, drawPoint, addData } from './util';
// import Stats from 'stats.js';
import * as posenet from '@tensorflow-models/posenet';
import clm from 'clmtrackr';
import Chart from 'chart.js';
import { ThrowStmt } from '@angular/compiler';

export class CheatDetectModel {
  guiState: Object;
  // TODO find datatype
  ctrack: any;
  TFJSPositions: Object;
  nonCheatingFlag: boolean;

  videoWidth: number;
  videoHeight: number;
  flipPoseHorizontal: boolean;

  videoElementId: string;
  canvasElementIdTFJS: string;
  canvasElementIdCLM: string;

  cheatLog: Function;

  chart: Chart;

  constructor(videoElementId: string, canvasElementIdTFJS: string, canvasElementIdCLM: string, cheatLog: Function) {
    // Setup start
    this.videoWidth = 640;
    this.videoHeight = 480;
    // const stats = new Stats();
    this.flipPoseHorizontal = false;

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

    this.guiState = {
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
    this.ctrack = new clm.tracker({
      faceDetection: {
        useWebWorkers: false,
      },
    });
    this.ctrack.init();
    this.nonCheatingFlag = true;

    this.videoElementId = videoElementId;
    this.canvasElementIdTFJS = canvasElementIdTFJS;
    this.canvasElementIdCLM = canvasElementIdCLM;

    this.cheatLog = cheatLog;

    //TODO delete test chart
    const chartCanvas = document.getElementById("chart");

    if (isCanvas(chartCanvas)) {
      const chartCtx = chartCanvas.getContext('2d');
      this.chart = new Chart(chartCanvas, {
        type: 'line',
        data: {
          datasets: [{
            label: 'leftEar',
            data: [],
            borderWidth: 1,
            borderColor: 'aqua'
          }, {
            label: 'rightEar',
            data: [],
            borderWidth: 1,
            borderColor: 'pink'
          }]
        },
        options: {
          responsive: true
        }
      });
    }
  }

  /**
   * Loads a the camera to be used in the demo
   * @param videoElementId
   */
  async setupCamera(videoElementId: any): Promise<HTMLElement> {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById(videoElementId);

    if (isVideo(video)) {
      video.width = this.videoWidth;
      video.height = this.videoHeight;

      video.srcObject = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          // width: videoWidth,
          // height: videoHeight,
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
  async loadVideo(videoElementId: any): Promise<HTMLElement> {
    const video = await this.setupCamera(videoElementId);

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
  async bindPage(minPoseConfidence?: number): Promise<void> {
    // toggleLoadingUI(true);
    const input = this.guiState['input'];

    const net = await posenet.load({
      architecture: input.architecture,
      outputStride: input.outputStride,
      inputResolution: input.inputResolution,
      multiplier: input.multiplier,
      quantBytes: input.quantBytes
    });

    // toggleLoadingUI(false);

    let video;
    const canvasTFJS = document.getElementById(this.canvasElementIdTFJS);
    const canvasCLM = document.getElementById(this.canvasElementIdCLM);

    try {
      video = await this.loadVideo(this.videoElementId);
    } catch (e) {
      // FIXME general
      const info = document.getElementById('info');
      info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
      info.style.display = 'block';
      throw e;
    }

    this.guiState['net'] = net;

    this.TFJSDetect(video, canvasTFJS, minPoseConfidence);
    this.CLMDetect(video, canvasTFJS, canvasCLM);
  }

  /**
   * detect point of face.
   * @param video
   * @constructor
   */
  async TFJSDetect(video: any, canvasTFJS: any, minPoseConfidence: number): Promise<void> {
    if (isCanvas(canvasTFJS)) {
      const ctxTFJS = canvasTFJS.getContext('2d');

      canvasTFJS.width = this.videoWidth;
      canvasTFJS.height = this.videoHeight;

      // since images are being fed from a webcam, we want to feed in the
      // original image and then just flip the keypoints' x coordinates. If instead
      // we flip the image, then correcting left-right keypoint pairs requires a
      // permutation on all the keypoints.

      minPoseConfidence = minPoseConfidence || +this.guiState['singlePoseDetection']['minPoseConfidence'];
      let guiState = this.guiState;
      let flipPoseHorizontal = this.flipPoseHorizontal;

      let cheatDetectModel = this;
      async function poseDetectionFrame(): Promise<void> {
        // Begin monitoring code for frames per second
        // stats.begin();
        // TFJS Start
        // FIXME fps is stuck this line
        const pose = await guiState['net'].estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: 'single-person'
        });

        // const minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        const minPartConfidence = +guiState['singlePoseDetection']['minPartConfidence'];

        ctxTFJS.clearRect(0, 0, cheatDetectModel.videoWidth, cheatDetectModel.videoHeight);


        pose.forEach(({ score, keypoints }) => {
          if (score >= minPoseConfidence) {
            cheatDetectModel.TFJSPositions = cheatDetectModel.parseTFJSPosition(keypoints);
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
  CLMDetect(video: any, canvasTFJS: any, canvasCLM: any): void {
    if (isCanvas(canvasTFJS) && isCanvas(canvasCLM)) {
      const ctxTFJS = canvasTFJS.getContext('2d');
      const ctxCLM = canvasCLM.getContext('2d');

      canvasTFJS.width = this.videoWidth;
      canvasTFJS.height = this.videoHeight;

      canvasCLM.width = this.videoWidth;
      canvasCLM.height = this.videoHeight;

      // since images are being fed from a webcam, we want to feed in the
      // original image and then just flip the keypoints' x coordinates. If instead
      // we flip the image, then correcting left-right keypoint pairs requires a
      // permutation on all the keypoints.

      this.ctrack.start(video);
      let cheatDetectModel = this;

      async function poseDetectionFrame(): Promise<void> {
        // Begin monitoring code for frames per second
        // stats.begin();

        // CLM Start
        ctxCLM.clearRect(0, 0, cheatDetectModel.videoWidth, cheatDetectModel.videoHeight);
        // psrElement.innerHTML = "score :" + ctrack.getScore().toFixed(4);
        // ctrack.getCurrentPosition((e) => {
        //   console.log(e);
        //   this.draw(canvasCLM);
        // });

        if (cheatDetectModel.TFJSPositions) {
          for (const [key, value] of Object.entries(cheatDetectModel.TFJSPositions)) {
            drawPoint(ctxCLM, value['y'], value['x'], 3, 'aqua');
          }

          let positions = cheatDetectModel.parseCLMPosition(cheatDetectModel.ctrack.getCurrentPosition());
          let drawFlag = true;

          if (cheatDetectModel.ctrack.getCurrentPosition()) {
            let distance = cheatDetectModel.checkAllDistance(positions, cheatDetectModel.TFJSPositions);

            for (const [key, value] of Object.entries(distance)) {
              if (value > 30) {
                drawFlag = false;
              }
            }

            for (const [key, value] of Object.entries(positions)) {
              drawPoint(ctxCLM, value['y'], value['x'], 3, 'pink');
            }

            if (drawFlag) {
              cheatDetectModel.ctrack.draw(canvasCLM);
            }
            // console.log(TFJSPositions);
          }

          cheatDetectModel.cheatDetect(positions, cheatDetectModel.TFJSPositions, drawFlag);
        }
        // CLM End

        // End monitoring code for frames per second
        // stats.end();

        requestAnimationFrame(poseDetectionFrame);
      }

      poseDetectionFrame();
    }
  }

  /**
   * Parse position data
   * @param positions
   */
  parseTFJSPosition(positions: any): Object {
    positions = {
      nose: positions[0]["position"],
      leftEye: positions[1]["position"],
      rightEye: positions[2]["position"],
      leftEar: positions[3]["position"],
      rightEar: positions[4]["position"],
    };

    return positions;
  }

  measureRelativityPositions(positions: Object): Object {
    const widthOfFase = this.distanceTwoDim(positions["leftEar"], positions["rightEar"]);
    const sensitivity = 0.3;

    let result = {
      positions: {
        nose: 0,
        leftEye: -this.distanceTwoDim(positions["leftEye"], positions["nose"]) / widthOfFase,
        rightEye: this.distanceTwoDim(positions["rightEye"], positions["nose"]) / widthOfFase,
        leftEar: -this.distanceTwoDim(positions["leftEar"], positions["nose"]) / widthOfFase,
        rightEar: this.distanceTwoDim(positions["rightEar"], positions["nose"]) / widthOfFase,
      },
    };

    result['headTurnFlag'] = Math.abs(result.positions.rightEar - 0.5) > sensitivity && Math.abs(result.positions.rightEar - 0.5) > sensitivity;

    addData(this.chart, 'test', result)
    return result;
  }

  /**
   * Parse position data
   * @param positions
   */
  parseCLMPosition(positions: any): Object {
    if (positions) {
      positions = {
        nose: { x: positions[62][0], y: positions[62][1] },
        rightEye: { x: positions[27][0], y: positions[27][1] },
        leftEye: { x: positions[32][0], y: positions[32][1] },
        rightEar: { x: positions[1][0], y: positions[1][1] },
        leftEar: { x: positions[13][0], y: positions[13][1] },
      };
    }

    return positions;
  }

  /**
   * Calculate two point's distance
   * @param position1
   * @param position2
   */
  distanceTwoDim(position1: Object, position2: Object): number {
    let a1 = Math.pow(position1['x'] - position2['x'], 2);
    let a2 = Math.pow(position1['y'] - position2['y'], 2);

    return Math.sqrt(a1 + a2);
  }

  /**
   * Calculate all point's distance
   * @param positions1
   * @param positions2
   */
  checkAllDistance(positions1: Object, positions2: Object): Object {
    let result = {};

    for (const [key, value] of Object.entries(positions1)) {
      result[key] = this.distanceTwoDim(positions1[key], positions2[key]);
    }

    return result;
  }

  /**
   *
   * @param CLMPositions
   * @param TFJSPositions
   * @param drawFlag
   */
  cheatDetect(CLMPositions: Object, TFJSPositions: Object, drawFlag: boolean): void {
    const sensitivity = 0.3;

    const headTurnLeftFlag = this.distanceTwoDim(TFJSPositions['leftEye'], TFJSPositions['leftEar']) < sensitivity;
    const headTurnRightFlag = this.distanceTwoDim(TFJSPositions['rightEye'], TFJSPositions['rightEar']) < sensitivity;

    const measureResult = this.measureRelativityPositions(TFJSPositions);
    if (this.nonCheatingFlag) {
      if (drawFlag) {
        // TODO TFJS position combine postions
        const eyeTurnLeftFlag = false;
        const eyeTurnRightFlag = false;

        if (headTurnLeftFlag || headTurnRightFlag || eyeTurnLeftFlag || eyeTurnRightFlag) {
          this.callBackend();
        }
      } else {
        if (headTurnLeftFlag || headTurnRightFlag) {
          this.callBackend();
        }
      }
    }

  }

  // TODO temp
  async callBackend() {
    const canvas = document.getElementById(this.canvasElementIdCLM);
    const video = document.getElementById(this.videoElementId);
    const timeInterval = 200;
    const snapNumber = 3;
    let data = [];

    this.nonCheatingFlag = false;

    if (isCanvas(canvas) && isVideo(video)) {
      canvas.width = this.videoWidth;
      canvas.height = this.videoHeight;
      const ctx = canvas.getContext('2d');

      for (let counter = 0; counter < snapNumber; counter++) {
        setTimeout(() => {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          let dataURI = canvas.toDataURL('image/jpeg');
          data.push(dataURI);
        }, timeInterval * counter)
      }

      setTimeout(() => {
        // TODO post to backend
        let payload = {
          studentId: localStorage.getItem('studentId'),
          studentName: localStorage.getItem('studentName'),
          cheatTime: Date.now(),
          cheatProbability: 0.9,
          cheatImages: data
        }
        console.log(payload);
        // 學生傳送作弊機率
        this.cheatLog(localStorage.getItem('teacherIp'), payload);
        this.nonCheatingFlag = true;
      }, timeInterval * snapNumber)
    }
    console.log("someone cheat!!!!!!")
  }
}