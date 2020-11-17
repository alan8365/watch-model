import {
  InputResolution,
  MobileNetMultiplier,
  PoseNetArchitecture,
  PoseNetOutputStride,
  PoseNetQuantBytes
} from '@tensorflow-models/posenet/dist/types';
import {isCanvas, isVideo, drawPoint, addData} from './util';
// import Stats from 'stats.js';
import * as posenet from '@tensorflow-models/posenet';
import clm from 'clmtrackr';
import Chart from 'chart.js';
import {FacePosition, FaceScale, ModelType, Vector2D} from './types';
import {sigmoid} from './util';

export class CheatDetectModel {
  guiState: any;
  ctrack: any;
  TFJSPositions: FacePosition;
  nonCheatingFlag: boolean;

  videoWidth: number;
  videoHeight: number;
  flipPoseHorizontal: boolean;

  videoElementId: string;
  canvasElementIdTFJS: string;
  canvasElementIdCLM: string;

  cheatLog: (...args: any[]) => void;

  chart: Chart;

  pupilTurnFlame = 0;

  constructor(videoElementId: string, canvasElementIdTFJS: string, canvasElementIdCLM: string, cheatLog: (...args: any[]) => void) {
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

    // TODO delete test chart
    const chartCanvas = document.getElementById('chart');

    if (isCanvas(chartCanvas)) {
      const chartCtx = chartCanvas.getContext('2d');
      this.chart = new Chart(chartCanvas, {
        type: 'line',
        data: {
          datasets: [{
            label: 'leftEye',
            data: [],
            borderWidth: 1,
            borderColor: 'aqua'
          }, {
            label: 'nose',
            data: [],
            borderWidth: 1,
            borderColor: 'pink'
          }, {
            label: 'leftEar',
            data: [],
            borderWidth: 1,
            borderColor: 'purple'
          }]
        },
        options: {
          responsive: true,
          scales: {
            yAxes: [{
              ticks: {
                suggestedMax: 1,
                suggestedMin: 0
              }
            }]
          }
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
   * @param minPoseConfidence
   */
  async bindPage(minPoseConfidence?: number): Promise<void> {
    // toggleLoadingUI(true);
    const input = this.guiState.input;

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

    this.guiState.net = net;

    this.TFJSDetect(video, canvasTFJS, minPoseConfidence);
    this.CLMDetect(video, canvasTFJS, canvasCLM);
  }

  /**
   * detect point of face.
   * @param video
   * @param canvasTFJS
   * @param minPoseConfidence
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

      minPoseConfidence = minPoseConfidence || +this.guiState.singlePoseDetection.minPoseConfidence;
      const guiState = this.guiState;
      const flipPoseHorizontal = this.flipPoseHorizontal;

      const cheatDetectModel = this;

      async function poseDetectionFrame(): Promise<void> {
        // Begin monitoring code for frames per second
        // stats.begin();
        // TFJS Start
        // FIXME fps is stuck this line
        const pose = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: 'single-person'
        });

        // const minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        const minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;

        ctxTFJS.clearRect(0, 0, cheatDetectModel.videoWidth, cheatDetectModel.videoHeight);


        pose.forEach(({score, keypoints}) => {
          if (score >= minPoseConfidence) {
            cheatDetectModel.TFJSPositions = cheatDetectModel.parsePosition(keypoints, 'TFJS');
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
      const cheatDetectModel = this;

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
            drawPoint(ctxCLM, value.y, value.x, 3, 'aqua');
          }

          const positions = cheatDetectModel.parsePosition(cheatDetectModel.ctrack.getCurrentPosition(), 'CLM');
          let drawFlag = false;

          if (cheatDetectModel.ctrack.getCurrentPosition()) {
            const distance = cheatDetectModel.checkAllDistance(
              positions,
              cheatDetectModel.TFJSPositions
            );

            console.log(distance);
            addData(cheatDetectModel.chart, new Date().getMilliseconds().toString(), distance);

            if (Object.values(distance).every((e) => e < 0.1)) {
              drawFlag = true;
            }

            // FIXME delete this in prod
            for (const [key, value] of Object.entries(positions)) {
              drawPoint(ctxCLM, value.y, value.x, 3, 'pink');
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
   * @param position
   * @param positionsType
   */
  parsePosition(position: FacePosition, positionsType: ModelType): FacePosition {
    if (position) {
      if (positionsType === 'CLM') {
        position = {
          nose: {x: position[62][0], y: position[62][1]},
          rightEye: {x: position[27][0], y: position[27][1]},
          leftEye: {x: position[32][0], y: position[32][1]},
          rightEar: {x: position[1][0], y: position[1][1]},
          leftEar: {x: position[13][0], y: position[13][1]},
        };
      } else if (positionsType === 'TFJS') {
        position = {
          nose: position[0].position,
          leftEye: position[1].position,
          rightEye: position[2].position,
          leftEar: position[3].position,
          rightEar: position[4].position,
        };
      }
    }

    return position;
  }


  /**
   * measure the relativity position and get detect signal.
   * @param positions
   */
  measureRelativityPositions(positions: FacePosition): {
    nose: number;
    leftEar: number;
    rightEye: number;
    leftEye: number;
    rightEar: number
  } {
    const widthOfFace = this.calculateTwoPointDistance(positions.leftEar, positions.rightEar);

    return {
      nose: 0,
      leftEye: -this.calculateTwoPointDistance(positions.leftEye, positions.nose) / widthOfFace,
      rightEye: this.calculateTwoPointDistance(positions.rightEye, positions.nose) / widthOfFace,
      leftEar: -this.calculateTwoPointDistance(positions.leftEar, positions.nose) / widthOfFace,
      rightEar: this.calculateTwoPointDistance(positions.rightEar, positions.nose) / widthOfFace,
    };
  }

  /**
   * Get detect degree.
   * @param TFJSPositions
   * @param CLMPositions
   * @param drawFlag
   * @param sensitivity
   */
  measureCheatDegree(
    TFJSPositions: FacePosition,
    CLMPositions: FacePosition,
    drawFlag: boolean,
    sensitivity: number = 1
  ): any {
    const widthOfFace = this.calculateTwoPointDistance(TFJSPositions.leftEar, TFJSPositions.rightEar);
    const pupilBaseLine = 0.03;
    const headBaseLine = 0.15;


    let relativityPosition: any;
    let headTurnDegree: number;
    let pupilTurnDegree: number;

    relativityPosition = this.measureRelativityPositions(TFJSPositions);

    if (drawFlag) {
      const leftPupilTurn = this.calculateTwoPointDistance(CLMPositions.leftEye, TFJSPositions.leftEye) / widthOfFace;
      const rightPupilTurn = this.calculateTwoPointDistance(CLMPositions.rightEye, TFJSPositions.rightEye) / widthOfFace;

      relativityPosition.leftPupilTurn = leftPupilTurn;
      relativityPosition.rightPupilTurn = rightPupilTurn;

      pupilTurnDegree = Math.max(
        Math.abs(rightPupilTurn) - pupilBaseLine * sensitivity,
        Math.abs(leftPupilTurn) - pupilBaseLine * sensitivity
      );

      if (pupilTurnDegree > 0) {
        this.pupilTurnFlame += 1;
        pupilTurnDegree *= this.pupilTurnFlame * 0.1;
        console.log(this.pupilTurnFlame);
      } else {
        this.pupilTurnFlame = 0;
      }
    }

    const leftHeadTurn = this.calculateTwoPointDistance(TFJSPositions.leftEye, TFJSPositions.leftEar) / widthOfFace;
    const rightHeadTurn = this.calculateTwoPointDistance(TFJSPositions.rightEye, TFJSPositions.rightEar) / widthOfFace;

    headTurnDegree = Math.abs(leftHeadTurn - rightHeadTurn) - headBaseLine;

    const result = {
      relativityPosition,
      headTurnDegree,
      pupilTurnDegree,
      leftHeadTurn,
      rightHeadTurn
    };

    return result;
  }

  /**
   * Calculate two point's distance
   * @param point1
   * @param point2
   */
  calculateTwoPointDistance(point1: Vector2D, point2: Vector2D): number {
    const a1 = Math.pow(point1.x - point2.x, 2);
    const a2 = Math.pow(point1.y - point2.y, 2);

    return Math.sqrt(a1 + a2);
  }

  /**
   * Calculate all point's distance
   * @param TFJSPositions
   * @param CLMPositions
   */
  checkAllDistance(TFJSPositions: FacePosition, CLMPositions: FacePosition): object {
    const widthOfFace = this.calculateTwoPointDistance(TFJSPositions.leftEar, TFJSPositions.rightEar);

    const TFJSRelativityPositions = this.measureRelativityPositions(TFJSPositions);
    const CLMRelativityPositions = {
      nose: this.calculateTwoPointDistance(CLMPositions.nose, TFJSPositions.nose) / widthOfFace,
      leftEye: -this.calculateTwoPointDistance(CLMPositions.leftEye, TFJSPositions.nose) / widthOfFace,
      rightEye: this.calculateTwoPointDistance(CLMPositions.rightEye, TFJSPositions.nose) / widthOfFace,
      leftEar: -this.calculateTwoPointDistance(CLMPositions.leftEar, TFJSPositions.nose) / widthOfFace,
      rightEar: this.calculateTwoPointDistance(CLMPositions.rightEar, TFJSPositions.nose) / widthOfFace,
    };

    const result = {};
    for (const [key, value] of Object.entries(TFJSPositions)) {
      result[key] = Math.abs(TFJSRelativityPositions[key] - CLMRelativityPositions[key]);
    }

    return result;
  }

  /**
   *
   * @param CLMPositions
   * @param TFJSPositions
   * @param drawFlag
   */
  cheatDetect(CLMPositions: FacePosition, TFJSPositions: FacePosition, drawFlag: boolean): void {
    const sensitivity = 0.8;
    const threshold = 0.8;
    const measureResult = this.measureCheatDegree(TFJSPositions, CLMPositions, drawFlag, sensitivity);

    let possibility = 0;
    let {headTurnDegree, pupilTurnDegree} = measureResult;

    headTurnDegree = headTurnDegree * 20 - 2;
    pupilTurnDegree = pupilTurnDegree * 20 - 2;

    if (this.nonCheatingFlag) {
      if (drawFlag) {
        possibility = sigmoid(headTurnDegree * 0.9 + pupilTurnDegree * 0.7);
      } else {
        possibility = sigmoid(headTurnDegree);
      }

      if (possibility > threshold) {
        this.callBackend(possibility);
      }
    }

  }

  // TODO temp
  async callBackend(possibility: number): Promise<void> {
    const canvas = document.getElementById(this.canvasElementIdCLM);
    const video = document.getElementById(this.videoElementId);
    const timeInterval = 200;
    const snapNumber = 3;
    const data = [];

    this.nonCheatingFlag = false;

    if (isCanvas(canvas) && isVideo(video)) {
      canvas.width = this.videoWidth;
      canvas.height = this.videoHeight;
      const ctx = canvas.getContext('2d');

      for (let counter = 0; counter < snapNumber; counter++) {
        setTimeout(() => {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataURI = canvas.toDataURL('image/jpeg');
          data.push(dataURI);
        }, timeInterval * counter);
      }

      setTimeout(() => {
        // TODO post to backend
        const payload = {
          studentId: localStorage.getItem('studentId'),
          studentName: localStorage.getItem('studentName'),
          cheatTime: Date.now(),
          cheatProbability: possibility,
          cheatImages: data
        };
        console.log(payload);
        // 學生傳送作弊機率
        this.cheatLog(localStorage.getItem('teacherIp'), payload);
        this.nonCheatingFlag = true;
      }, timeInterval * snapNumber);
    }
    console.log('someone cheat!!!!!!');
  }
}
