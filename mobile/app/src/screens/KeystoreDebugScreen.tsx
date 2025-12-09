import React, { useState } from 'react'
import { View, Text, Button, StyleSheet } from 'react-native'
import KeystoreService from '../services/Keystore'

const KeystoreDebugScreen: React.FC = () => {
  const [log, setLog] = useState<string[]>([])

  const append = (s: string) => setLog(l => [s, ...l].slice(0, 50))

  const gen = async () => {
    append('Generating keypair...')
    try {
      const res = await KeystoreService.generateKeyPair('mlp-demo-key')
      append(JSON.stringify(res))
    } catch (e: any) {
      append('Error: ' + String(e.message || e))
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Keystore Debug</Text>
      <Button title="Generate Key Pair" onPress={gen} />
      {log.map((l, i) => (
        <Text key={i} style={styles.line}>{l}</Text>
      ))}
    </View>
  )
}

const styles = StyleSheet.create({ container: { padding: 16, flex: 1 }, title: { fontSize: 18, fontWeight: '600', marginBottom: 12 }, line: { marginTop: 8 } })

export default KeystoreDebugScreen
