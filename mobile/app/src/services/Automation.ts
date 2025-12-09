import { NativeModules } from 'react-native'

const { NativeAutomation } = NativeModules as any

const AutomationService = {
  start: async (): Promise<any> => {
    return new Promise((resolve, reject) => {
      try {
        NativeAutomation.startAutomation((res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  },

  stop: async (): Promise<any> => {
    return new Promise((resolve, reject) => {
      try {
        NativeAutomation.stopAutomation((res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  }
}

export default AutomationService
