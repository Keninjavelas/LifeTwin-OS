describe('NativeInference extended wrapper (JS smoke test)', () => {
  beforeAll(() => {
    const { NativeModules } = require('react-native')
    NativeModules.NativeInference = {
      reloadModel: (cb) => cb({ modelPresent: false, onnxRuntimeAvailable: false, modelSessionLoaded: false, loaderReturned: false }),
      getModelStatus: (cb) => cb({ modelPresent: false, onnxRuntimeAvailable: false, modelSessionLoaded: false }),
      predictNextApp: (history, cb) => cb('com.example.app'),
      unloadModel: (cb) => cb({ status: 'ok', modelSessionLoaded: false }),
      getSessionInfo: (cb) => cb({ modelPresent: false, onnxRuntimeAvailable: false, modelSessionLoaded: false }),
      runInference: (history, cb) => cb('com.example.app')
    }
  })

  test('unloadModel and getSessionInfo exist', async () => {
    const NativeInference = require('../src/services/NativeInference').default
    const u = await NativeInference.unloadModel()
    expect(u).toHaveProperty('status')
    const s = await NativeInference.getSessionInfo()
    expect(s).toHaveProperty('modelPresent')
  })

  test('runInference returns a package', async () => {
    const NativeInference = require('../src/services/NativeInference').default
    const r = await NativeInference.runInference(['a','b'])
    expect(typeof r).toBe('string')
  })
})
