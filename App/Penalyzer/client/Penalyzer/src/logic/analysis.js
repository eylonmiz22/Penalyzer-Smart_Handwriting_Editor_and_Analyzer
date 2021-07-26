import React from 'react';
import Toast from 'react-native-simple-toast';
import { Alert } from 'react-native';

import { BASE_URL } from '@env';



const serverErrMsg = "Analysis failed. The document may be invalid";

function doneAnalyzingMessage() {
  Toast.show("Done Analyzing", Toast.SHORT);
}

function pleaseUploadMessage() {
  Toast.show("Please upload a new and a non-empty document", Toast.SHORT);
}

function networkFailureMessage() {
  Toast.show("Network request failed", Toast.SHORT);
}


async function detectAuthenticityRequest(image) {
  try {
    return await fetch(`${BASE_URL}/analysis/predict-authenticity`, {
      method: 'post',
      body: JSON.stringify({ "image": image }),
      headers: {
        'Content-Type': 'application/json',
      }
    });
  } catch (err) {
    console.error("logic-->analysis-->detectAuthenticityRequest:\nError occurred: " + err);
  }
  return null;
};


async function fitStyleRequest(image) {
  try {
    return await fetch(`${BASE_URL}/analysis/fit-writer`, {
      method: 'post',
      body: JSON.stringify({ "image": image }),
      headers: {
        'Content-Type': 'application/json',
      }
    });
  } catch (err) {
    console.error("logic-->analysis-->fitStyleRequest:\nError occurred: " + err);
  }
  return null;
};


async function identifyWriterRequest(image) {
  try {
    return await fetch(`${BASE_URL}/analysis/predict-writer`, {
      method: 'post',
      body: JSON.stringify({ "image": image }),
      headers: {
        'Content-Type': 'application/json',
      }
    });
  } catch (err) {
    console.error("logic-->analysis-->detectAuthenticityRequest:\nError occurred: " + err);
  }
  return null;
};


export async function detectAuthenticity(imageState, setImageState, setBusy) {
  try {
    if (imageState != null && imageState['analyzeFlag']) {
      setBusy(true);
      await detectAuthenticityRequest(imageState["data"])
        .then((response) => response.json())
        .then((analyzed) => {
          if (analyzed["data"] === "null") {
            Toast.show(serverErrMsg, Toast.SHORT);
            setBusy(false);
            throw errorMsg;
          }
          setImageState({
            data: analyzed["data"],
            analyzeFlag: false,
            isEdited: false,
            isBlankInitialization: false,
            editingFlag: false
          });
          setBusy(false);
        });
    } else {
      pleaseUploadMessage();
    }
  } catch (err) {
    console.log(`components-->screens-->analyzeScreen-->detectAuthenticity:\nError occurred: ${err}`);
    networkFailureMessage();
    setBusy(false);
  }
}


// export async function identifyWriter(imageState, setBusy, isFitted) {
//   try {
//     if (imageState != null && imageState['analyzeFlag']) {
//       if (isFitted) {
//         setBusy(true);
//         await identifyWriterRequest(imageState["data"])
//           .then((response) => response.json())
//           .then((data) => {
//             if (data["confidence"] === "null") {
//               Toast.show(serverErrMsg, Toast.SHORT);
//               setBusy(false);
//               throw errorMsg;
//             }
//             setBusy(false);
//             Alert.alert(`The given style is ${data['confidence']*100}% similar to the fitted one`);
//           });
//       } else {
//         Toast.show("Please upload a style calibration document first");
//       }
//     } else {
//       pleaseUploadMessage();
//     }
//   } catch (err) {
//     console.log(`components-->screens-->analyzeScreen-->identifyWriter:\nError occurred: ${err}`);
//     networkFailureMessage();
//     setBusy(false);
//   }
// }

// NO FITTING for testing purposes
export async function identifyWriter(imageState, setBusy, isFitted) {
  try {
    if (imageState != null && imageState['analyzeFlag']) {
      setBusy(true);
      await identifyWriterRequest(imageState["data"])
        .then((response) => response.json())
        .then((data) => {
          if (data["confidence"] === "null") {
            Toast.show(serverErrMsg, Toast.SHORT);
            setBusy(false);
            throw errorMsg;
          }
          setBusy(false);
          Alert.alert(`The given style is ${data['confidence']*100}% similar to the fitted one`);
        });
    } else {
      pleaseUploadMessage();
    }
  } catch (err) {
    console.log(`components-->screens-->analyzeScreen-->identifyWriter:\nError occurred: ${err}`);
    networkFailureMessage();
    setBusy(false);
  }
}

export async function fitStyle(imageState, setBusy, setIsFitted) {
  try {
    if (imageState != null && imageState['analyzeFlag']) {
      setBusy(true);
      await fitStyleRequest(imageState["data"])
        .then((response) => response.json())
        .then((data) => {
          if (data["fitted"] === "null" || data["fitted"] === false) {
            Toast.show(serverErrMsg, Toast.SHORT);
            setBusy(false);
            throw errorMsg;
          }
          setBusy(false);
          setIsFitted(true);
        });
    } else {
      pleaseUploadMessage();
    }
  } catch (err) {
    console.log(`components-->screens-->analyzeScreen-->fitStyle:\nError occurred: ${err}`);
    networkFailureMessage();
    setBusy(false);
  }
}