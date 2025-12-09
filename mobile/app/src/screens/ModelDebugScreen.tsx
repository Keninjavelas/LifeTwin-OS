import React, { useState } from 'react'
import { View, Text, Button, StyleSheet, ScrollView } from 'react-native'
import NativeInferenceService from '../services/NativeInference'

const ModelDebugScreen: React.FC = () => {
  const [status, setStatus] = useState<any>(null)
  const [log, setLog] = useState<string[]>([])

  const appendLog = (s: string) => setLog(prev => [s, ...prev].slice(0, 50))

  const onReload = async () => {
    appendLog('Reloading model...')
    try {
      const res = await NativeInferenceService.reloadModel()
      setStatus(res)
      appendLog('Reload returned: ' + JSON.stringify(res))
    } catch (e: any) {
      appendLog('Reload failed: ' + String(e.message || e))
    }
  }

  const onStatus = async () => {
    appendLog('Fetching status...')
    try {
      const res = await NativeInferenceService.getModelStatus()
      setStatus(res)
      appendLog('Status: ' + JSON.stringify(res))
    } catch (e: any) {
      appendLog('Status failed: ' + String(e.message || e))
    }
  }

  const onUnload = async () => {
    appendLog('Unloading model...')
    try {
      const res = await NativeInferenceService.unloadModel()
      appendLog('Unload returned: ' + JSON.stringify(res))
      setStatus(await NativeInferenceService.getSessionInfo())
    } catch (e: any) {
      appendLog('Unload failed: ' + String(e.message || e))
    }
  }

  const onRun = async () => {
    appendLog('Running inference (heuristic)...')
    try {
      const res = await NativeInferenceService.runInference(['com.example.app', 'com.example.other'])
      appendLog('Run result: ' + String(res))
    } catch (e: any) {
      appendLog('Run failed: ' + String(e.message || e))
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Model Debug</Text>

      <View style={styles.row}>
        <Button title="Reload Model" onPress={onReload} />
        <View style={{ width: 12 }} />
        <Button title="Get Status" onPress={onStatus} />
        <View style={{ width: 12 }} />
        <Button title="Unload" onPress={onUnload} />
        <View style={{ width: 12 }} />
        <Button title="Run" onPress={onRun} />
      </View>

      <View style={styles.statusBox}>
        <Text style={styles.sub}>Current Status</Text>
        <Text>{status ? JSON.stringify(status, null, 2) : 'No status yet'}</Text>
      </View>

      <View style={styles.logBox}>
        <Text style={styles.sub}>Recent Log</Text>
        {log.map((l, idx) => (
          <Text key={idx} style={styles.logLine}>{l}</Text>
        ))}
      </View>
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
  },
  row: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  statusBox: {
    padding: 12,
    backgroundColor: '#f6f6f6',
    borderRadius: 8,
    marginBottom: 12,
  },
  sub: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  logBox: {
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 8,
  },
  logLine: {
    fontSize: 12,
    color: '#333',
    marginBottom: 6,
  }
})

export default ModelDebugScreen
