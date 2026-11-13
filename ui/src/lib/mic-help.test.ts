import { describe, expect, it } from 'vitest'
import { detectEnvironment, diagnoseMic, micHelp, osSteps, siteSteps, type MicFailure } from './mic-help'

const UA = {
  iphoneSafari:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1',
  iphoneChrome:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/140.0.7339.122 Mobile/15E148 Safari/604.1',
  iphoneWebview:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
  iphoneInstagram:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22F76 Instagram 385.0.0.26.82 (iPhone15,2; iOS 18_5; en_US; en; scale=3.00; 1179x2556; 742961871)',
  iphoneFacebook:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/22F76 [FBAN/FBIOS;FBAV/520.0.0.38.101;FBBV/750000000;FBDV/iPhone15,2;FBMD/iPhone;FBSN/iOS;FBSV/18.5;FBSS/3;FBID/phone;FBLC/en_GB;FBOP/5]',
  iphoneGoogleApp:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) GSA/380.0.788317806 Mobile/15E148 Safari/604.1',
  iphoneLinkedIn:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 [LinkedInApp]/9.31.1234',
  ipadSafari:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15',
  androidChrome:
    'Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36',
  androidWebview:
    'Mozilla/5.0 (Linux; Android 15; Pixel 9 Build/AP4A.250105.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/140.0.0.0 Mobile Safari/537.36',
  androidLine:
    'Mozilla/5.0 (Linux; Android 15; SM-S921B Build/AP3A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/140.0.0.0 Mobile Safari/537.36 Line/15.2.1/IAB',
  androidSlack:
    'Mozilla/5.0 (Linux; Android 15; Pixel 9 Build/AP4A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/140.0.0.0 Mobile Safari/537.36 Slack/25.03.10.0',
  androidSamsung:
    'Mozilla/5.0 (Linux; Android 15; SM-S921B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/28.0 Chrome/130.0.0.0 Mobile Safari/537.36',
  androidFirefox: 'Mozilla/5.0 (Android 15; Mobile; rv:143.0) Gecko/143.0 Firefox/143.0',
  macSafari:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15',
  macChrome:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
  winEdge:
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0',
  linuxFirefox: 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
}

const env = (ua: string, touch = 0) => detectEnvironment(ua, { maxTouchPoints: touch })

describe('detectEnvironment', () => {
  it('names the platform and browser of ordinary browsers and does not call them in-app', () => {
    expect(env(UA.iphoneSafari, 5)).toEqual({ platform: 'ios', browser: 'safari', inApp: null })
    expect(env(UA.iphoneChrome, 5)).toEqual({ platform: 'ios', browser: 'chrome', inApp: null })
    expect(env(UA.androidChrome, 5)).toEqual({ platform: 'android', browser: 'chrome', inApp: null })
    expect(env(UA.androidSamsung, 5)).toEqual({ platform: 'android', browser: 'samsung', inApp: null })
    expect(env(UA.androidFirefox, 5)).toEqual({ platform: 'android', browser: 'firefox', inApp: null })
    expect(env(UA.macSafari)).toEqual({ platform: 'macos', browser: 'safari', inApp: null })
    expect(env(UA.macChrome)).toEqual({ platform: 'macos', browser: 'chrome', inApp: null })
    expect(env(UA.winEdge)).toEqual({ platform: 'windows', browser: 'edge', inApp: null })
    expect(env(UA.linuxFirefox)).toEqual({ platform: 'linux', browser: 'firefox', inApp: null })
  })

  it('treats a touch-screen "Mac" as an iPad', () => {
    expect(env(UA.ipadSafari, 5).platform).toBe('ios')
    expect(env(UA.ipadSafari, 0).platform).toBe('macos')
  })

  it('recognises in-app browsers by name where it can', () => {
    expect(env(UA.iphoneInstagram, 5).inApp).toBe('Instagram')
    expect(env(UA.iphoneFacebook, 5).inApp).toBe('Facebook')
    expect(env(UA.iphoneGoogleApp, 5).inApp).toBe('the Google app')
    expect(env(UA.iphoneLinkedIn, 5).inApp).toBe('LinkedIn')
    expect(env(UA.androidLine, 5).inApp).toBe('LINE')
    expect(env(UA.androidSlack, 5).inApp).toBe('Slack')
  })

  it('recognises anonymous web views: Android "; wv)" and iOS WebKit without "Safari/"', () => {
    expect(env(UA.androidWebview, 5).inApp).toBe('an app')
    expect(env(UA.iphoneWebview, 5).inApp).toBe('an app')
  })
})

const failure = (over: Partial<MicFailure> = {}): MicFailure => ({
  kind: 'denied',
  name: 'NotAllowedError',
  detail: 'Permission denied',
  elapsedMs: 10,
  permission: 'unknown',
  ...over,
})

describe('diagnoseMic', () => {
  const safari = env(UA.iphoneSafari, 5)
  it('blames an in-app browser first when one is detected', () => {
    expect(diagnoseMic(failure(), env(UA.iphoneInstagram, 5))).toBe('in-app')
    expect(diagnoseMic(failure({ kind: 'failed' }), env(UA.androidWebview, 5))).toBe('in-app')
  })
  it('reads a saved block as a site-level block', () => {
    expect(diagnoseMic(failure({ permission: 'denied' }), safari)).toBe('site')
  })
  it('reads a slow refusal with nothing saved as a dismissed prompt', () => {
    expect(diagnoseMic(failure({ elapsedMs: 4200, permission: 'prompt' }), safari)).toBe('dismissed')
    expect(diagnoseMic(failure({ kind: 'dismissed', detail: 'Permission dismissed' }), safari)).toBe('dismissed')
  })
  it('reads an instant refusal with nothing saved as an operating-system block', () => {
    expect(diagnoseMic(failure({ elapsedMs: 8, permission: 'prompt' }), safari)).toBe('os')
    expect(diagnoseMic(failure({ detail: 'Permission denied by system', permission: 'denied' }), safari)).toBe('os')
  })
  it('admits when it cannot tell (Safari reports no permission state)', () => {
    expect(diagnoseMic(failure({ elapsedMs: 8, permission: 'unknown' }), safari)).toBe('unknown')
    const help = micHelp(failure({ elapsedMs: 8, permission: 'unknown' }), safari)
    expect(help.sections).toHaveLength(2)
  })
  it('passes hardware and context problems through', () => {
    expect(diagnoseMic(failure({ kind: 'missing' }), safari)).toBe('missing')
    expect(diagnoseMic(failure({ kind: 'busy' }), safari)).toBe('busy')
    expect(diagnoseMic(failure({ kind: 'insecure' }), safari)).toBe('insecure')
  })
})

describe('steps', () => {
  it('are specific to the platform and browser', () => {
    expect(siteSteps(env(UA.iphoneSafari, 5)).join(' ')).toContain('Website Settings')
    expect(siteSteps(env(UA.iphoneChrome, 5)).join(' ')).toContain('Settings app, then Apps, then Chrome')
    expect(siteSteps(env(UA.androidChrome, 5)).join(' ')).toContain('Permissions')
    expect(siteSteps(env(UA.macSafari)).join(' ')).toContain('Settings for This Website')
    expect(siteSteps(env(UA.linuxFirefox)).join(' ')).toContain('Remove “Blocked”')
    expect(osSteps(env(UA.iphoneSafari, 5)).join(' ')).toContain('Settings app, then Apps, then Safari')
    expect(osSteps(env(UA.androidChrome, 5)).join(' ')).toContain('Mic access')
    expect(osSteps(env(UA.macChrome)).join(' ')).toContain('Privacy & Security')
    expect(osSteps(env(UA.winEdge)).join(' ')).toContain('Privacy & security')
  })
  it('offer a link to copy only when another browser is the answer', () => {
    expect(micHelp(failure(), env(UA.iphoneInstagram, 5)).offerCopyLink).toBe(true)
    expect(micHelp(failure(), env(UA.iphoneInstagram, 5)).explanation).toContain('Safari')
    expect(micHelp(failure(), env(UA.androidLine, 5)).explanation).toContain('Chrome')
    expect(micHelp(failure({ permission: 'denied' }), env(UA.macChrome)).offerCopyLink).toBe(false)
  })
})
