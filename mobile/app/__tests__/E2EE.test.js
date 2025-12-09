describe('E2EE wrapper (JS smoke test)', () => {
  beforeAll(() => {
    const { NativeModules } = require('react-native')
    NativeModules.KeystoreModule = {
      generateWrappedDataKey: (cb) => cb({ wrapped: 'dGVzdF93cmFwcGVk' }),
      unwrapDataKey: (wrapped, cb) => cb({ dek: 'cmF3X2Rlay==' })
    }
  })

  test('generateWrappedDataKey returns wrapped string', async () => {
    const E2EE = require('../src/services/E2EE').default
    const res = await E2EE.generateWrappedDataKey()
    expect(res).toHaveProperty('wrapped')
  })

  test('unwrapDataKey returns dek', async () => {
    const E2EE = require('../src/services/E2EE').default
    const res = await E2EE.unwrapDataKey('dummy')
    expect(res).toHaveProperty('dek')
  })
})
