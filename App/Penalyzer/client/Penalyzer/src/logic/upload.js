
import React from 'react';
import Toast from 'react-native-simple-toast';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { BASE_URL } from '@env';


const uploadToast = () => {
  Toast.show("File Uploaded Successfully!", Toast.SHORT)
};


async function selectFile() {
  try { // Opening Document Picker to select one file
    let res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      base64: true
    });
    if (!res.cancelled) {
      return res.base64;
    }
  } catch (err) {
    console.log(`logic-->upload-->selectFile:\nError occurred: ${err}`);
  }
  return null;
};


async function getBlankPage() {
  try {
    return await fetch(`${BASE_URL}/blank_page`, { method: 'get' }, bodyParser = {
      json: {limit: '50mb', extended: true},
      urlencoded: {limit: '50mb', extended: true}
    });
  } catch (err) {
    console.log(`logic-->upload-->getBlankPage:\nError occurred: ${err}`);
  }
  return null;
};


export async function uploadDoc(setImageState) {
  let pic = await selectFile();
  if (pic != null) {
    setImageState({ 
      data: pic,
      analyzeFlag: true,
      isEdited: false,
      isBlankInitialization: false,
      editingFlag: true
    });
    uploadToast();
  }
}


export async function blankPage(setImageState) {
  await getBlankPage().then((response) => response.json()).then((responseJson) => {
    setImageState({
      data: responseJson['data'],
      analyzeFlag: false,
      isEdited: false,
      isBlankInitialization: true,
      editingFlag: true
    });
  }).then(uploadToast);
}


export async function takePicture(cameraRef) {
  try {
    if (cameraRef) {
      let pic = await cameraRef.current.takePictureAsync({ base64: false, skipProcessing: false });
      let aspect_ratio = pic.width / pic.height;
      let resolution = null;
      if (aspect_ratio < 1) {
        resolution = aspect_ratio === 3/4 ? { width: 1080, height: 1440 } : { width: 1080, height: 1920 };
      } else {
        resolution = aspect_ratio === 4/3 ? { width: 1440, height: 1080 } : { width: 1920, height: 1080 };
      }
      pic = await ImageManipulator.manipulateAsync(
        pic.localUri || pic.uri,
        [{ resize: resolution }],
        { base64: true }    
      );
      return pic.base64;
    }
  } catch (err) {
    console.log(`logic-->upload-->takePicture:\nError occurred: ${err}`);

  }
  return null;
};
