import React, { useContext } from 'react';
import { StyleSheet, View, Dimensions, Image, ActivityIndicator } from 'react-native';
import ImageZoom from 'react-native-image-pan-zoom';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { Button } from 'react-native-elements';

import { detectAuthenticity, identifyWriter, fitStyle } from '../../logic/analysis';
import { ImageStateContext } from '../contexts/imageStateContext';
import { BusyContext } from '../contexts/busyContext';
import { FittedContext } from '../contexts/fittedContext';


const AnalyzeScreen = () => {

  let dimensions = Dimensions.get('window');
  let imageStyle = {
    flex: 1,
    flexDirection: 'row',
    resizeMode: 'contain',
    width: dimensions.width,
    height: dimensions.height,
  }

  const [imageState, setImageState] = useContext(ImageStateContext);
  const [busy, setBusy] = useContext(BusyContext);
  const [isFitted, setIsFitted] = useContext(FittedContext);

  return (
    busy ? <View style={styles.activityIndicatorView}><ActivityIndicator size="large" color='#1E90DD' /></View> :
    <View style={styles.container}>
      <View style={styles.imageView}>
      {
        imageState != null ?
        <ImageZoom  cropWidth={dimensions.width-60}
                    cropHeight={dimensions.height-80}
                    imageWidth={dimensions.width}
                    imageHeight={dimensions.height}>
          <Image source={{uri: `data:image/gif;base64,${imageState['data']}`}} style={imageStyle} />
        </ImageZoom> : <View />
      }
      </View>
      <View style={styles.buttonsView}>
        <View style={styles.deepfakeDetectionButton}>
          <Button
            title={"Detect Forgery"}
            type="outline"
            onPress={ () => detectAuthenticity(imageState, setImageState, setBusy) }
            icon={<Ionicons name="locate" size={24} style={styles.locateIcon} />} />
        </View>
        <View style={styles.identificationButtonsView}>
          <View style={styles.identificationButton}>
            <Button 
              title={"Identify Writer"}
              type="outline"
              onPress={ () => identifyWriter(imageState, setBusy, isFitted) }
              icon={<Ionicons name="locate" size={24} style={styles.locateIcon} />} />
          </View>
          <View style={styles.identificationButton}>
            <Button 
              title={"Fit Style"}
              type="outline"
              onPress={ () => fitStyle(imageState, setBusy, setIsFitted) }
              icon={<MaterialCommunityIcons name="format-font" size={24} style={styles.locateIcon} />} />
          </View>
        </View>
      </View>
    </View>
  );
}


export default AnalyzeScreen;


const styles = StyleSheet.create({
  container: {
    flex: 1,
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    marginTop: 30,
    justifyContent: 'space-between',
  },

  activityIndicatorView : {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },

  imageView: {
    flex: 1,
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },

  buttonsView: {
    position: 'absolute',
    bottom: '10%',
    left: '5%',
    right: '5%',
  },

  identificationButtonsView: {
    justifyContent: 'space-between',
    flexDirection: 'row',
    marginTop: "5%",
  },

  identificationButton: {
    width: "47%",
  },

  deepfakeDetectionButton: {
    marginHorizontal: "20%"
  },
  
  locateIcon: {
    paddingRight: 10,
    color: "#1E90DD",
  }
});