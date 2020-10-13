import clm from 'clmtrackr';
import {loadVideo, isCanvas, isVideo} from './tfjs-model';

let canvas;
let ctx;

let trackingStarted = false;
let ctrack = new clm.tracker({
  faceDetection: {
    useWebWorkers: false,
  },
});
ctrack.init();

const videoWidth = 600;
const videoHeight = 500;

export async function startVideo(videoElementId: string, canvasElementId: string) {
  canvas = document.getElementById(canvasElementId);
  if (isCanvas(canvas)) {
    ctx = canvas.getContext('2d');
  }

  let video = await loadVideo(videoElementId);
  // start video
  if (isVideo(video)) {
    await video.play();
  }

  // start tracking
  ctrack.start(video);
  trackingStarted = true;
  // start loop to draw face
  drawLoop();
}

function drawLoop() {
  requestAnimFrame(drawLoop);
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  //psrElement.innerHTML = "score :" + ctrack.getScore().toFixed(4);
  if (ctrack.getCurrentPosition()) {
    console.log(ctrack);
    ctrack.draw(canvas);
  }
}

// helper functions

/**
 * Provides requestAnimationFrame in a cross browser way.
 */
let requestAnimFrame = (function() {
  return window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    function(/* function FrameRequestCallback */ callback) {
      return window.setTimeout(callback, 1000/60);
    };
})();

/**
 * Provides cancelRequestAnimationFrame in a cross browser way.
 */
let cancelRequestAnimFrame = (function() {
  return window.cancelAnimationFrame ||
    window.clearTimeout;
})();
