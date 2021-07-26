import React, { useContext, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import Dialog from "react-native-dialog";

import { BusyContext } from '../contexts/busyContext';
import { RemoveDialogVisibleContext } from '../contexts/removeDialogContext';
import { ImageStateContext } from '../contexts/imageStateContext';
import { remove } from '../../logic/editing';


const RemoveDialog = () => {
    const [busy, setBusy] = useContext(BusyContext);
    const [removeDialogVisible, setRemoveDialogVisible] = useContext(RemoveDialogVisibleContext);
    const [imageState, setImageState] = useContext(ImageStateContext);
    const [text, setText] = useState(null);
    const [wordIndex, setWordIndex] = useState(null);
    const [styleIndex, setStyleIndex] = useState(null);

    return (
        <View>
        <Dialog.Container
            visible={removeDialogVisible}
            headerStyle={styles.header}>

            <Dialog.Title>Remove Handwriting</Dialog.Title>
            <Dialog.Input 
            placeholder={"Word position index within the document"}
            style={styles.input}
            onChangeText={ (inputWordIndex) => setWordIndex(inputWordIndex) }
            />
            <Dialog.Button
            label="Erase"
            onPress={ () => remove(imageState, parseInt(wordIndex), setImageState, setRemoveDialogVisible, setBusy) }
            />
            <Dialog.Button 
            label="Cancel"
            onPress={ () => setRemoveDialogVisible(false) }
            />
        </Dialog.Container>
        </View>
    );
}


export default RemoveDialog;


const styles = StyleSheet.create({
  header: {
    paddingBottom: 0
  },

  input: {
    marginBottom: '-8%'
  }
});