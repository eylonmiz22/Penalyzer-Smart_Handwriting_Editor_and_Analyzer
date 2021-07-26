import React from 'react';
import Toast from 'react-native-simple-toast';

import { BASE_URL } from '@env';


const NUM_GENERAL_STYLES = 5;


function checkImageState(imageState) {
  let imageStateFlag = false;
  if (imageState == null) {
    imageStateFlag = true;
  } else if (!imageState['editingFlag']) {
    imageStateFlag = true;
  } 
  if (imageStateFlag) {
    throw "Please upload a new document";
  }
}


function checkInsert(imageState, wordIndex, styleIndex, text) {
  checkImageState(imageState);

  let errorFlag = false;
  let errorMsg = "Errors occurred:\n";
  if (!Number.isInteger(wordIndex)) {
    if (wordIndex < 0 && wordIndex != -1) {
      errorMsg += "- Please insert a positive integer word index or -1 to append the text at the end\n";
      errorFlag = true;
    }
  }

  let styleFlag = false;
  if (!Number.isInteger(styleIndex)) {
    styleFlag = true;
  } else if ((styleIndex < 0 && styleIndex != -1) || styleIndex > NUM_GENERAL_STYLES) {
    styleFlag = true;
  }
  if (styleFlag) {
    errorMsg += `- Please insert an integer style index (-1 to ${NUM_GENERAL_STYLES})\n`;
    errorFlag = true;
  }

  if (text.length === 0) {
    errorMsg += "- Please insert a non empty text";
    errorFlag = true;
  }

  if (imageState["isBlankInitialization"] && styleIndex === -1) {
    errorMsg += "- Cannot extract style from a blank document";
    errorFlag = true; 
  }

  if (errorFlag) {
    throw errorMsg;
  }
}


function checkRemove(imageState, wordIndex) {
  checkImageState(imageState);
  if (!Number.isInteger(wordIndex) || wordIndex == null) {
    throw "Please insert an integer word index";
  }
}


export async function insert(imageState, wordIndex, styleIndex, text, setImageState, setInsertDialogVisible, setBusy) {
  try {
    setBusy(true);
    checkInsert(imageState, wordIndex, styleIndex, text);

    let request_body = {
      "wordIndex": wordIndex,
      "styleIndex": styleIndex,
      "text": text,
      "isBlankInitialization": imageState["isBlankInitialization"]
    };
    if (!imageState["isEdited"]) {
      request_body.image = imageState["data"];
    }

    await fetch(`${BASE_URL}/editing/insert`, {
      method: 'post',
      body: JSON.stringify(request_body),
      headers: { 'Content-Type': 'application/json' }
    })
    .then((response) => response.json())
    .then((edited) => {
      setImageState({
        data: edited["data"],
        analyzeFlag: true,
        isEdited: true,
        isBlankInitialization: false,
        editingFlag: imageState['editingFlag']
      });
      setInsertDialogVisible(false);
      setBusy(false);
    });

  } catch (err) {
    Toast.show(err, Toast.LONG);
    console.error(`logic-->editing-->insert:\n"${err}`);
    setBusy(false);
  }
};


export async function remove(imageState, wordIndex, setImageState, setRemoveDialogVisible, setBusy) {
  try {
    setBusy(true);
    checkRemove(imageState, wordIndex)

    let request_body = {
      "wordIndex": wordIndex,
    };
    if (!imageState["isEdited"]) {
      request_body.image = imageState["data"];
    }

    await fetch(`${BASE_URL}/editing/delete`, {
      method: 'post',
      body: JSON.stringify(request_body),
      headers: { 'Content-Type': 'application/json' }
    })
    .then((response) => response.json())
    .then((edited) => {
      setImageState({ data: edited["data"],
        analyzeFlag: true,
        isEdited: true,
        isBlankInitialization: false,
        editingFlag: imageState['editingFlag'] 
      });
      setRemoveDialogVisible(false);
      setBusy(false);
    });

  } catch (err) {
    Toast.show("Network Request Failed", Toast.SHORT);
    console.error("logic-->editing-->remove:\nError occurred: " + err);
    setBusy(false);
  }
};
