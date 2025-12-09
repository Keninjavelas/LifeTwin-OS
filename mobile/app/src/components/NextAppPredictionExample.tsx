import React, { useState } from 'react';
import { View, Text, Button } from 'react-native';
import { NativeCollectors } from '@services/nativeCollectors';

const NextAppPredictionExample: React.FC = () => {
  const [prediction, setPrediction] = useState<string | null>(null);

  const runPrediction = async () => {
    // Example history: last few package names
    const history = ['com.example.chat', 'com.example.news', 'com.example.mail'];
    try {
      const p = await NativeCollectors.predictNextApp(history);
      setPrediction(p);
    } catch (e) {
      setPrediction(null);
      console.warn('prediction failed', e);
    }
  }

  return (
    <View style={{ padding: 12 }}>
      <Button title="Predict next app" onPress={runPrediction} />
      <Text style={{ marginTop: 8 }}>Prediction: {prediction ?? 'none'}</Text>
    </View>
  )
}

export default NextAppPredictionExample;
