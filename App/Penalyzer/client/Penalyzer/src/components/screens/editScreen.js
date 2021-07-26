import React, { useContext } from 'react';
import { StyleSheet, View, Dimensions, Image, ActivityIndicator } from 'react-native';
import ImageZoom from 'react-native-image-pan-zoom';
import { Button } from 'react-native-elements';
import { FontAwesome5 } from '@expo/vector-icons';
import { Entypo } from '@expo/vector-icons';

import Spacer from '../utils/spacer';
import { ImageStateContext } from '../contexts/imageStateContext';
import { BusyContext } from '../contexts/busyContext';
import InsertDialog from '../dialogs/insertDialog';
import RemoveDialog from '../dialogs/removeDialog';
import { InsertDialogVisibleContext } from '../contexts/insertDialogContext';
import { RemoveDialogVisibleContext } from '../contexts/removeDialogContext';


const EditScreen = () => {
  const [imageState, setImageState] = useContext(ImageStateContext);
  const [busy, setBusy] = useContext(BusyContext);
  const [insertDialogVisible, setInsertDialogVisible] = useContext(InsertDialogVisibleContext);
  const [removeDialogVisible, setRemoveDialogVisible] = useContext(RemoveDialogVisibleContext);
  
  let dimensions = Dimensions.get('window');
  let imageStyle = {
    flex: 1,
    flexDirection: 'row',
    resizeMode: 'contain',
    width: dimensions.width,
    height: dimensions.height,
  }

  return (
    busy ? <View style={styles.activityIndicatorView}><ActivityIndicator size="large" color='#1E90DD' /></View> :
    <View style={styles.container}>
      <InsertDialog />
      <RemoveDialog />
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
        <Button 
          title={"Add"}
          type="outline"
          onPress={ () => setInsertDialogVisible(true) }
          icon={ <Entypo name="pencil" size={24} style={styles.addIcon} /> } />
        <Spacer />
        <Button
          title={"Erase"}
          type="outline"
          onPress={ () => setRemoveDialogVisible(true) }
          icon={ <FontAwesome5 name="eraser" size={24} style={styles.eraseIcon} /> } />
      </View>
    </View>
  );
}


export default EditScreen;


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
    // borderBottomColor: '#333',
    // borderWidth: 5,
  },

  imageView: {
    // marginTop: 100,
    flex: 1,
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },

  buttonsView: {
    justifyContent: 'center',
    flexDirection: 'row',
    position: 'absolute',
    bottom: '10%',
    left: '25%',
    right: '25%',
  },

  activityIndicatorView : {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },

  addIcon: {
    paddingRight: 30,
    color: "#1E90DD",
  },

  eraseIcon: {
    paddingRight: 28,
    color: "#1E90DD",
  }
});