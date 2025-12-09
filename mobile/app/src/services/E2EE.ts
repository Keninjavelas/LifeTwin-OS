import { NativeModules } from 'react-native'

const { KeystoreModule } = NativeModules as any

const E2EE = {
  // Generate a random DEK and return a wrapped (RSA-encrypted) base64 string
  generateWrappedDataKey: (): Promise<{ wrapped: string }> => {
    return new Promise((resolve, reject) => {
      KeystoreModule.generateWrappedDataKey((res: any) => resolve(res), (err: any) => reject(err))
    })
  },

  // Unwrap a wrapped DEK (base64) and return the raw DEK as base64
  unwrapDataKey: (wrappedB64: string): Promise<{ dek: string }> => {
    return new Promise((resolve, reject) => {
      KeystoreModule.unwrapDataKey(wrappedB64, (res: any) => resolve(res), (err: any) => reject(err))
    })
  }
}

export default E2EE
