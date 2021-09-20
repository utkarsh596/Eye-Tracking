/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

import {TRIANGULATION} from './triangulation';


import { checkConversionForErrors } from '@tensorflow/tfjs-core/dist/util';
import { isCapableOfRenderingToFloatTexture } from '@tensorflow/tfjs-backend-webgl/dist/webgl_util';
import { eye } from '@tensorflow/tfjs-core';
import { collectGatherOpShapeInfo } from '@tensorflow/tfjs-core/dist/ops/segment_util';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;

const GREEN = '#32EEDB';
const RED = "#FF2C35";
const BLUE = "#157AB3";


function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

let model, ctx, videoWidth, videoHeight, video, canvas,
    scatterGLHasInitialized = false, scatterGL, rafID;

const VIDEO_SIZE = 500;

const state = {
  backend: 'webgl',
  maxFaces: 1,
  triangulateMesh: true,
  predictIrises: true
};


tf.env().set('WEBGL_CPU_FORWARD', false); //to be used when importing cpu backend

async function setupCamera() {  
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',

      width: VIDEO_SIZE,   
      height: VIDEO_SIZE        
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const predictDirection = (eyeRatio,keypoints,upDist) =>{

  if(calculateFaceAngle(keypoints).pitch <-0.1)
  {
   // console.log("Looking Up");
   lookingDirection.faceUp++;
   constantlyLooking.faceUp++;
   onScreen=0; 
  }
  else if(calculateFaceAngle(keypoints).yaw > 0.17)
  {
   // console.log("Looking Left");
   lookingDirection.faceLeft++;
   constantlyLooking.faceLeft++;
   onScreen=0; 
  }
  else if(calculateFaceAngle(keypoints).yaw < -0.18)
  {
   // console.log("Looking Right");
   lookingDirection.faceRight++;
   constantlyLooking.faceRight++;
   onScreen=0; 
  }
  else if(upDist>14)
  {
   // console.log("Up");
   lookingDirection.up++;
   constantlyLooking.up++;
   onScreen=0;
  }
  else if(eyeRatio<0.992)
  {
   // console.log("Right");
   lookingDirection.right++;
   constantlyLooking.right++;
   onScreen=0;
  }
  else if(eyeRatio>1.0125)
  {
   // console.log("Left");
   lookingDirection.left++;
   constantlyLooking.left++;
   onScreen=0; 
  }
  else if(distance(keypoints[66],keypoints[296])<43)
  {
   console.log("Return to Screen");
   onScreen=0;
  }
  else
   onScreen++;
};

const resetValues = () =>{

  for(const values of Object.keys(lookingDirection))
  {
    lookingDirection[values]=0;
  }
  
  for(const values of Object.keys(constantlyLooking))
  {
    constantlyLooking[values]=0;
  }
}

const resetObject = () => {
  for(const values of Object.keys(directionCount))
  {
    directionCount[values]=0;
  }
};
 
 const calculateFaceAngle = (mesh) => {
   if (!mesh) return {};
   const radians = (a1, a2, b1, b2) => Math.atan2(b2 - a2, b1 - a1);
   const angle = {
     roll: radians(mesh[33][0], mesh[33][1], mesh[263][0], mesh[263][1]),
     yaw: radians(mesh[33][0], mesh[33][2], mesh[263][0], mesh[263][2]),
     pitch: radians(mesh[10][1], mesh[10][2], mesh[152][1], mesh[152][2]),
   };
   return angle;
 }
 
 let onScreen=0;

 const constantlyLooking ={
  left:0,
  right:0,
  up:0,
  faceLeft:0,
  faceRight:0,
  faceUp:0
 };

 const lookingDirection = {
   left:0,
   right:0,
   up:0,
   faceLeft:0,
   faceRight:0,
   faceUp:0
 };

 const directionCount = {
   left:0,
   right:0,
   up:0,
   faceLeft:0,
   faceRight:0,
   faceUp:0
 };
 
 function generateWarning(){
  
   if(onScreen>30)
    {
      resetValues();

      if(onScreen>300)
       resetObject();
    }
    
  for(const values of Object.keys(directionCount))
  {
    if(lookingDirection[values] > 50)
     {
       directionCount[values]++;
       lookingDirection[values]=0;
     }
  };


  for(const values of Object.keys(constantlyLooking))
    {
      if(constantlyLooking[values] > 150)
       {
        //WARNING
        console.log('Warning:-Looking away for a long time');
        resetValues();
        resetObject();
       }
    }   

  
    for(const values of Object.keys(directionCount))
    {
      if(directionCount[values] === 4)
       {
        //WARNING
        console.log('Warning:-Looking away constantly at a particular direction');
        resetObject();
       }
    }   
}

async function renderPrediction() {


  const predictions = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: state.predictIrises
  });
  ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
  
  
  if (predictions.length > 0) {
    predictions.forEach(prediction => {
      const keypoints = prediction.scaledMesh;

      if(keypoints.length > NUM_KEYPOINTS) {

        const leftCenter = keypoints[NUM_KEYPOINTS];
        
        const leftDiameterY = distance(
          keypoints[NUM_KEYPOINTS + 4],
          keypoints[NUM_KEYPOINTS + 2]);
        const leftDiameterX = distance(
          keypoints[NUM_KEYPOINTS + 3],
          keypoints[NUM_KEYPOINTS + 1]);

        if(keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          const rightDiameterY = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
          const rightDiameterX = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);


          const centreR = (keypoints[160][0]+keypoints[159][0]+keypoints[158][0]+keypoints[144][0]+keypoints[145][0]+keypoints[153][0])/6;
          const centreL = (keypoints[385][0]+keypoints[386][0]+keypoints[387][0]+keypoints[380][0]+keypoints[374][0]+keypoints[373][0])/6;
          const ratio = (leftCenter[0]+rightCenter[0])/(centreL+centreR);
          
          const top0 = keypoints[23];
          const top1 = keypoints[253];
          const upDist = (top0[1]-rightCenter[1])/2+(top1[1]-leftCenter[1])/2;
          predictDirection(ratio,keypoints,upDist);
          generateWarning();
        }
      }
    });

  }

  requestAnimationFrame(renderPrediction);
};

async function startEyeTracking() {
  await tf.setBackend(state.backend);

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = GREEN;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 0.5;

  model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    {maxFaces: state.maxFaces});
  renderPrediction();

};

startEyeTracking();
