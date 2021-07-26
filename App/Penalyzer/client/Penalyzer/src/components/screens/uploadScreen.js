import React, { useContext } from 'react';
import { ActivityIndicator, StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import { Entypo } from '@expo/vector-icons'; 
import { MaterialCommunityIcons } from '@expo/vector-icons'; 

import { uploadDoc, blankPage } from '../../logic/upload';
import { ImageStateContext } from '../contexts/imageStateContext';
import { BusyContext } from '../contexts/busyContext';


export default function UploadScreen({ navigation }) {
  const [imageState, setImageState] = useContext(ImageStateContext);
  const [busy, setBusy] = useContext(BusyContext);

  return (
    busy ? <View style={styles.activityIndicatorView}><ActivityIndicator size="large" color='#1E90DD' /></View> :
    <View style={styles.container}>
      <TouchableOpacity style={styles.button} onPress={() => { blankPage(setImageState); }} >
        <Text style={styles.text}>Blank Page</Text>
        <Entypo name="document" style={styles.entypo} />
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={ () => { uploadDoc(setImageState); }}>
        <Text style={styles.text}>Upload</Text>
        <Entypo name="text-document" style={styles.entypo} />
      </TouchableOpacity>
      <TouchableOpacity style={styles.button}>
        <Text style={styles.text}>Camera</Text>
        <MaterialCommunityIcons name="cellphone-screenshot"
          style={styles.MaterialCommunityIcons} onPress={() => { navigation.navigate('Camera'); }} />
      </TouchableOpacity>
      <ActivityIndicator />
    </View>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    marginTop: 10,
    alignItems: 'center',
    padding: 20,
    justifyContent: 'space-between',
  },

  activityIndicatorView : {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },

  button: {
    flex: 1,
    alignItems: 'center',
  },

  text: {
    fontWeight: 'bold',
    fontSize: 20,
    color: "#1E90DD",
  },

  entypo: {
    color: "#1E90DD",
    fontSize: 160
  },

  MaterialCommunityIcons: {
    color: "#1E90DD",
    fontSize: 180
  }
});