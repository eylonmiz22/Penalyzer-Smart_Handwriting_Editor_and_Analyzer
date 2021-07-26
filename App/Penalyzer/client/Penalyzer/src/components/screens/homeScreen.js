import React from 'react';
import { StyleSheet, View, Image, Dimensions } from 'react-native';
import { Button } from 'react-native-elements';
import { Feather } from '@expo/vector-icons';
import { AntDesign } from '@expo/vector-icons';
import { FontAwesome5 } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

import Spacer from '../utils/spacer';
import welcomeImage from '../../../assets/welcome.png';



const welcomeImageUri = Image.resolveAssetSource(welcomeImage).uri;
const imgRatio = 0.75;
const dimensions = Dimensions.get('window');
const imageWidth = dimensions.width * imgRatio;
const imageHeight = Math.round(dimensions.width * 9 / 16);

const HomeScreen = () => {
  const navigation = useNavigation();  

  return (
  <View style={styles.container}>
    <View style={styles.imageView}>
      <Image source={{ uri: welcomeImageUri }} style={styles.image} />
    </View>
    <View style={styles.buttonsView}>
      <Button title={"Analyze Document"} type="outline" icon={
        <AntDesign name="linechart" size={24} style={styles.analyzeIcon} />} 
        onPress = {() => { navigation.navigate('Analyze'); }} />
      <Spacer />
      <Button title={"Edit Document"} type="outline" icon={
        <Feather name="edit" size={25} style={styles.editIcon} /> }
        onPress = { () => { navigation.navigate('Edit'); }} />
      <Spacer />
      <Button title={"Upload Document"} type="outline" icon={
        <FontAwesome5 name="file-upload" size={24} style={styles.uploadIcon} /> }
        onPress = { () => { navigation.navigate('Upload'); }} />
    </View>
  </View>
  );
}

export default HomeScreen;
// export default withNavigation(HomeScreen);

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    marginTop: 30,
    // flexDirection: 'row',
    justifyContent: 'space-between',
    borderBottomColor: '#333',
  },

  buttonsView: {
    position: 'absolute',
    bottom: '15%',
    left: '15%',
    right: '15%',
  },

  image: {
    resizeMode: 'contain',
    width: imageWidth,
    height: imageHeight,
  },

  imageView: {
    justifyContent: 'center',
    alignItems: 'center',
    top: '15%',
  },

  editIcon: {
    paddingRight: 100,
    color: "#1E90DD",
  },

  analyzeIcon: {
    paddingRight: 70,
    color: "#1E90DD",
  },

  uploadIcon: {
    paddingRight: 80,
    color: "#1E90DD",
  }
});