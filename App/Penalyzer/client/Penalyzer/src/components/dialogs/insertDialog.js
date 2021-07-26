import React, { useContext, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import Dialog from "react-native-dialog";

import { InsertDialogVisibleContext } from '../contexts/insertDialogContext';
import { BusyContext } from '../contexts/busyContext';
import { ImageStateContext } from '../contexts/imageStateContext';
import { insert } from '../../logic/editing';


const InsertDialog = () => {
  const [busy, setBusy] = useContext(BusyContext);
  const [insertDialogVisible, setInsertDialogVisible] = useContext(InsertDialogVisibleContext);
  const [imageState, setImageState] = useContext(ImageStateContext);
  const [text, setText] = useState(null);
  const [wordIndex, setWordIndex] = useState(null);
  const [styleIndex, setStyleIndex] = useState(null);

  return (
    <View>
      <Dialog.Container
        visible={insertDialogVisible}
        headerStyle={styles.header}>

        <Dialog.Title>Insert Handwriting</Dialog.Title>
        <Dialog.Input 
          placeholder={"Input text"}
          style={styles.input}
          onChangeText={ (inputText) => setText(inputText) }
        />
        <Dialog.Input 
          placeholder={"Word position index within the document"}
          style={styles.input}
          onChangeText={ (inputWordIndex) => setWordIndex(inputWordIndex) }
        />
        <Dialog.Input 
          placeholder={"Style index (-1 for your font style)"}
          style={styles.input} 
          onChangeText={ (inputStyleIndex) => setStyleIndex(inputStyleIndex) }
        />
        <Dialog.Button
          label="Add"
          onPress={
            () => insert(imageState, parseInt(wordIndex), parseInt(styleIndex), text, setImageState, setInsertDialogVisible, setBusy)
          }
        />
        <Dialog.Button 
          label="Cancel"
          onPress={() => setInsertDialogVisible(false) }
        />
      </Dialog.Container>
    </View>
  );
}


export default InsertDialog;


const styles = StyleSheet.create({
  header: {
    paddingBottom: 0
  },

  input: {
    marginBottom: '-8%'
  }
});