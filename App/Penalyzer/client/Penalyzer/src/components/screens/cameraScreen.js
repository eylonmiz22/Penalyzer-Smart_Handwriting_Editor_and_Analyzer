import React, { useState, useEffect, useRef, useContext } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import Toast from 'react-native-simple-toast';

import { ImageStateContext } from '../contexts/imageStateContext';
import { BusyContext } from '../contexts/busyContext';
import { toggleFlash, captureDoc } from '../../logic/camera';


export default function CameraScreen() {
    const [hasPermission, setHasPermission] = useState(null);
    const [flash, setFlash] = useState(Camera.Constants.FlashMode.off);
    const [flashIconName, setFlashIconName] = useState("flash-off");
    const [imageState, setImageState] = useContext(ImageStateContext);
    const [busy, setBusy] = useContext(BusyContext);
    const cameraRef = useRef(Camera | null);

    useEffect(() => {
        (async () => {
        const { status } = await Camera.requestPermissionsAsync();
        setHasPermission(status === 'granted');
        })();
    }, []);

    if (hasPermission === null) {
        return <View />;
    }
    if (hasPermission === false) {
        return <Text>No access to camera</Text>;
    }
    return (
        <View style={styles.container}>
            <Camera style={styles.camera} type={Camera.Constants.Type.back}
                flashMode={flash} autoFocus={Camera.Constants.AutoFocus.on}
                ref = {cameraRef} useCamera2Api={true}>
                <View style={styles.buttonContainer}>
                    <TouchableOpacity
                        style={styles.flashButton}
                        onPress={() => { toggleFlash(flash, setFlash, setFlashIconName); }}>
                        <Ionicons name={flashIconName} size={60} color="black" />
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.cameraButton}
                        onPress={() => { captureDoc(cameraRef, setImageState, setBusy).then(Toast.show("Image Captured!", Toast.SHORT)); }}>
                        <Ionicons name="camera" size={65} color="black" />
                    </TouchableOpacity>
                </View>
            </Camera>
        </View>
    );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 0,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    alignSelf: 'center',
    position: 'absolute',
    bottom: 15,
    width: '90%',
    justifyContent: 'space-between'
  },
  flashButton: {
    position: 'relative',
    alignSelf: 'flex-start',
  },
  cameraButton: {
    position: 'relative',
  }
});