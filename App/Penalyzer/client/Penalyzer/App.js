import React, { useState } from 'react';
import { StyleSheet } from 'react-native';
import HomeScreen from './src/components/screens/homeScreen';
import AnalyzeScreen from './src/components/screens/analyzeScreen';
import { createStackNavigator } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';
import EditScreen from './src/components/screens/editScreen';
import UploadScreen from './src/components/screens/uploadScreen';
import CameraScreen from './src/components/screens/cameraScreen';
import { ImageStateContext } from './src/components/contexts/imageStateContext';
import { BusyContext } from './src/components/contexts/busyContext';
import { InsertDialogVisibleContext } from './src/components/contexts/insertDialogContext';
import { RemoveDialogVisibleContext } from './src/components/contexts/removeDialogContext';
import { FittedContext } from './src/components/contexts/fittedContext';


const MainNavigator = createStackNavigator();


export default function App() {
  const [imageState, setImageState] = useState(null);
  const [busy, setBusy] = useState(false);
  const [insertDialogVisible, setInsertDialogVisible] = useState(false);
  const [removeDialogVisible, setRemoveDialogVisible] = useState(false);
  const [isFitted, setIsFitted] = useState(false);
  
  return (
    <FittedContext.Provider value={[isFitted, setIsFitted]}>
      <RemoveDialogVisibleContext.Provider value={[removeDialogVisible, setRemoveDialogVisible]}>
        <InsertDialogVisibleContext.Provider value={[insertDialogVisible, setInsertDialogVisible]}>
          <BusyContext.Provider value={[busy, setBusy]}>
            <ImageStateContext.Provider value={[imageState, setImageState]}>
              <NavigationContainer>
                <MainNavigator.Navigator initialRouteName="Home" headerMode="screen">
                  <MainNavigator.Screen name="Home" component={HomeScreen} options={{
                    title: 'Penalyzer',
                    headerStyle: { backgroundColor: '#1E90DD' },
                    headerTitleAlign: "center",
                    headerTintColor: '#1E90DD',
                    headerTitleStyle: styles.headerTitleStyle 
                    }} />
                  <MainNavigator.Screen name="Analyze" component={AnalyzeScreen} options={{
                    title: 'Analyze',
                    headerStyle: { backgroundColor: '#1E90DD' },
                    headerTitleAlign: "center",
                    headerTintColor: 'white',
                    headerTitleStyle: styles.headerTitleStyle,
                  }} />
                  <MainNavigator.Screen name="Edit" component={EditScreen} options={{
                    title: 'Edit',
                    headerStyle: { backgroundColor: '#1E90DD' },
                    headerTitleAlign: "center",
                    headerTintColor: 'white',
                    headerTitleStyle: styles.headerTitleStyle,
                  }} />
                  <MainNavigator.Screen name="Upload" component={UploadScreen} options={{
                    title: 'Upload',
                    headerStyle: { backgroundColor: '#1E90DD' },
                    headerTitleAlign: "center",
                    headerTintColor: 'white',
                    headerTitleStyle: styles.headerTitleStyle
                  }} />
                  <MainNavigator.Screen name="Camera" component={CameraScreen} options={{
                    title: 'Camera',
                    headerStyle: { backgroundColor: '#1E90DD' },
                    headerTitleAlign: "center",
                    headerTintColor: 'white',
                    headerTitleStyle: styles.headerTitleStyle
                  }} />
                </MainNavigator.Navigator>
              </NavigationContainer>
            </ImageStateContext.Provider>
          </BusyContext.Provider>
        </InsertDialogVisibleContext.Provider>
      </RemoveDialogVisibleContext.Provider>
    </FittedContext.Provider>
  );
}


const styles = StyleSheet.create({
  headerTitleStyle: {
    fontWeight: 'bold',
    color: "white",
    fontSize: 20,
    letterSpacing: 1,
  },

  iconLeft: {
    left: 6,
    color: "white",
    fontSize: 24,
  }, 
  
  iconRight: {
    right: 6,
    fontSize: 24,
    color: "white",
  },
});