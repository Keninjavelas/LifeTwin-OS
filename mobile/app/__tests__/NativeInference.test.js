const { NativeModules } = require('react-native')

describe('NativeInference wrapper (JS smoke test)', () => {
  beforeEach(() => {
    // reset mock
    NativeModules.NativeInference = {
      reloadModel: jest.fn((s, e) => s({ modelPresent: false, onnxRuntimeAvailable: false, modelSessionLoaded: false })),
      getModelStatus: jest.fn((s, e) => s({ modelPresent: false, onnxRuntimeAvailable: false, modelSessionLoaded: false })),
      predictNextApp: jest.fn((history, s, e) => s('com.example.app')),
    }
  })

  test('native module mock exists and functions are callable', async () => {
    const { reloadModel, getModelStatus, predictNextApp } = NativeModules.NativeInference
    expect(typeof reloadModel).toBe('function')
    expect(typeof getModelStatus).toBe('function')
    expect(typeof predictNextApp).toBe('function')

    const reloadRes = await new Promise((res, rej) => reloadModel(res, rej))
    expect(reloadRes).toHaveProperty('modelPresent')

    const statusRes = await new Promise((res, rej) => getModelStatus(res, rej))
    expect(statusRes).toHaveProperty('onnxRuntimeAvailable')

    const pred = await new Promise((res, rej) => predictNextApp(['a','b'], res, rej))
    expect(typeof pred).toBe('string')
  })
})
