const { NativeModules } = require('react-native')

describe('Keystore wrapper (JS smoke test)', () => {
  beforeEach(() => {
    NativeModules.KeystoreModule = {
      generateKeyPair: jest.fn((alias, success, error) => success({ status: 'ok', alias })),
    }
  })

  test('keystore mock generates keys', async () => {
    const { generateKeyPair } = NativeModules.KeystoreModule
    expect(typeof generateKeyPair).toBe('function')

    const res = await new Promise((resv, rej) => generateKeyPair('mlp-demo-key', resv, rej))
    expect(res).toHaveProperty('status', 'ok')
    expect(res).toHaveProperty('alias', 'mlp-demo-key')
  })
})
