import React, { useState } from 'react'
import { View, Text, Button, StyleSheet } from 'react-native'
import AutomationService from '../services/Automation'

const AutomationScreen: React.FC = () => {
  const [status, setStatus] = useState<string | null>(null)

  const start = async () => {
    try {
      const res = await AutomationService.start()
      setStatus(res?.status || 'started')
    } catch (e: any) {
      setStatus('error: ' + String(e.message || e))
    }
  }

  const stop = async () => {
    try {
      const res = await AutomationService.stop()
      setStatus(res?.status || 'stopped')
    } catch (e: any) {
      setStatus('error: ' + String(e.message || e))
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Automation</Text>
      <View style={styles.row}>
        <Button title="Start" onPress={start} />
        <View style={{ width: 12 }} />
        <Button title="Stop" onPress={stop} />
      </View>
      <Text style={styles.status}>Status: {status ?? 'idle'}</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  container: { padding: 16, flex: 1 },
  title: { fontSize: 20, fontWeight: '600', marginBottom: 12 },
  row: { flexDirection: 'row', marginBottom: 12 },
  status: { marginTop: 12 }
})

export default AutomationScreen
