declare module 'react-native' {
  interface ModelStatus {
    modelPresent: boolean;
    onnxRuntimeAvailable: boolean;
    modelSessionLoaded: boolean;
    loaderReturned?: boolean;
  }

  interface NativeModulesStatic {
    NativeInference?: {
      reloadModel(success: (res: ModelStatus) => void, error: (err: any) => void): void;
      getModelStatus(success: (res: ModelStatus) => void, error: (err: any) => void): void;
      predictNextApp(history: string[], success: (res: string | null) => void, error: (err: any) => void): void;
    }
  }
}
