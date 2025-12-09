const { NativeModules } = require('react-native')

describe('NativeAutomation wrapper (JS smoke test)', () => {
  beforeEach(() => {
    NativeModules.NativeAutomation = {
      startAutomation: jest.fn((s, e) => s({ status: 'started' })),
      stopAutomation: jest.fn((s, e) => s({ status: 'stopped' })),
    }
  })

  test('automation module mock exists and is callable', async () => {
    const { startAutomation, stopAutomation } = NativeModules.NativeAutomation
    expect(typeof startAutomation).toBe('function')
    expect(typeof stopAutomation).toBe('function')

    const started = await new Promise((res, rej) => startAutomation(res, rej))
    expect(started).toHaveProperty('status', 'started')

    const stopped = await new Promise((res, rej) => stopAutomation(res, rej))
    expect(stopped).toHaveProperty('status', 'stopped')
  })
})
