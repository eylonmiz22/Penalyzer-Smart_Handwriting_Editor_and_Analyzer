import { Camera } from 'expo-camera';

import { takePicture } from './upload';


export function toggleFlash(flash, setFlash, setFlashIconName) {
    if (flash === Camera.Constants.FlashMode.off) {
        setFlash(Camera.Constants.FlashMode.on);
        setFlashIconName("flash");
    } else {
        setFlash(Camera.Constants.FlashMode.off);
        setFlashIconName("flash-off");
    }
}


export async function captureDoc(cameraRef, setImageState, setBusy) {
    setBusy(true);
    let pic = await takePicture(cameraRef);
    if (pic != null) {
        setImageState({
            data: pic,
            analyzeFlag: true,
            isEdited: false,
            isBlankInitialization: false,
            editingFlag: true
        });
        setBusy(false);
    }
}