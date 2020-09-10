import clm from 'clmtrackr';
import {loadVideo, isCanvas, isVideo} from './camera';

let overlay;
let overlayCC;

let trackingStarted = false;
let ctrack = new clm.tracker({
  faceDetection: {
    useWebWorkers: false,
  },
});
ctrack.init();

const videoWidth = 600;
const videoHeight = 500;

export async function startVideo() {
  overlay = document.getElementById('overlay');
  if (isCanvas(overlay)) {
    overlayCC = overlay.getContext('2d');
  }

  let vid = await loadVideo();
  // start video
  if (isVideo(vid)) {
    await vid.play();
  }

  // start tracking
  ctrack.start(vid);
  trackingStarted = true;
  // start loop to draw face
  drawLoop();
}

function drawLoop() {
  console.log('bububu');
  requestAnimFrame(drawLoop);
  overlayCC.clearRect(0, 0, videoWidth, videoHeight);
  //psrElement.innerHTML = "score :" + ctrack.getScore().toFixed(4);
  if (ctrack.getCurrentPosition()) {
    console.log(ctrack);
    ctrack.draw(overlay);
  }
}

// helper functions

/**
 * Provides requestAnimationFrame in a cross browser way.
 */
let requestAnimFrame = (function() {
  return window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    function(/* function FrameRequestCallback */ callback, /* DOMElement Element */ element) {
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

// video support utility functions
function supports_video() {
  return !!document.createElement('video').canPlayType;
}

function supports_h264_baseline_video() {
  if (!supports_video()) { return false; }
  var v = document.createElement("video");
  return v.canPlayType('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
}

function supports_webm_video() {
  if (!supports_video()) { return false; }
  var v = document.createElement("video");
  return v.canPlayType('video/webm; codecs="vp8"');
}
