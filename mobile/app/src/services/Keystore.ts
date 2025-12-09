import { NativeModules } from 'react-native'

const { KeystoreModule } = NativeModules as any

const KeystoreService = {
  generateKeyPair: async (alias: string): Promise<any> => {
    return new Promise((resolve, reject) => {
      try {
        KeystoreModule.generateKeyPair(alias, (res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  }
}

export default KeystoreService
