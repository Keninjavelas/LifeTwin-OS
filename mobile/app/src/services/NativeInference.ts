import { NativeModules } from 'react-native'

const { NativeInference } = NativeModules as any

type ModelStatus = {
  modelPresent: boolean
  onnxRuntimeAvailable: boolean
  modelSessionLoaded: boolean
  loaderReturned?: boolean
}

const NativeInferenceService = {
  reloadModel: async (): Promise<ModelStatus> => {
    return new Promise((resolve, reject) => {
      try {
        NativeInference.reloadModel((res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  },

  getModelStatus: async (): Promise<ModelStatus> => {
    return new Promise((resolve, reject) => {
      try {
        NativeInference.getModelStatus((res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  },

  predictNextApp: async (history: string[]): Promise<string | null> => {
    return new Promise((resolve, reject) => {
      try {
        NativeInference.predictNextApp(history, (res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  }

  , unloadModel: async (): Promise<{ status: string, modelSessionLoaded: boolean }> => {
    return new Promise((resolve, reject) => {
      try {
        NativeInference.unloadModel((res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  },

  getSessionInfo: async (): Promise<any> => {
    return new Promise((resolve, reject) => {
      try {
        NativeInference.getSessionInfo((res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  },

  runInference: async (history: string[]): Promise<string | null> => {
    return new Promise((resolve, reject) => {
      try {
        NativeInference.runInference(history, (res: any) => resolve(res), (err: any) => reject(err))
      } catch (e) {
        reject(e)
      }
    })
  }
}

export default NativeInferenceService
