/**
 * Working out why the microphone did not open, and what the person can do
 * about it on the device in front of them. Pure functions: the user agent and
 * the facts about the failure go in, an explanation comes out.
 */

export type Platform = 'ios' | 'android' | 'macos' | 'windows' | 'linux' | 'other'
export type BrowserName = 'safari' | 'chrome' | 'firefox' | 'edge' | 'samsung' | 'other'

export interface Environment {
  platform: Platform
  browser: BrowserName
  /** Name of the app whose built-in browser this looks like, or null. Deliberately conservative. */
  inApp: string | null
}

const IN_APP: [RegExp, string][] = [
  [/\bFBAN\/|\bFBAV\/|\bFB_IAB\/|\bFBIOS\b/, 'Facebook'],
  [/\bInstagram\b/, 'Instagram'],
  [/\bLine\//, 'LINE'],
  [/\bLinkedInApp\b/, 'LinkedIn'],
  [/\bSlack(?:_SSB)?\//i, 'Slack'],
  [/\bGSA\//, 'the Google app'],
  [/\bTwitter(?:Android|\s+for\s+iPhone)?\b/, 'X (Twitter)'],
  [/\bMicroMessenger\//, 'WeChat'],
  [/\bSnapchat\//, 'Snapchat'],
  [/\bPinterest\//, 'Pinterest'],
  [/\b(?:musical_ly|BytedanceWebview|TikTok)\b/i, 'TikTok'],
  [/\bTelegram(?:Bot)?\b/, 'Telegram'],
  [/\bWhatsApp\//, 'WhatsApp'],
  [/\bDiscord\//i, 'Discord'],
]

export function detectEnvironment(
  ua: string = typeof navigator === 'undefined' ? '' : navigator.userAgent,
  hints: { maxTouchPoints?: number; platform?: string } = typeof navigator === 'undefined'
    ? {}
    : { maxTouchPoints: navigator.maxTouchPoints, platform: navigator.platform },
): Environment {
  // iPadOS reports itself as a Mac; a Mac with a touch screen is an iPad.
  const ipad = /Macintosh/.test(ua) && (hints.maxTouchPoints ?? 0) > 1
  const ios = /iPhone|iPad|iPod/.test(ua) || ipad
  const android = /Android/.test(ua)
  const platform: Platform = ios
    ? 'ios'
    : android
      ? 'android'
      : /Macintosh|Mac OS X/.test(ua)
        ? 'macos'
        : /Windows/.test(ua)
          ? 'windows'
          : /Linux|X11|CrOS/.test(ua)
            ? 'linux'
            : 'other'

  const browser: BrowserName = /SamsungBrowser\//.test(ua)
    ? 'samsung'
    : /Edg(?:e|A|iOS)?\//.test(ua)
      ? 'edge'
      : /Firefox\/|FxiOS\//.test(ua)
        ? 'firefox'
        : /Chrome\/|CriOS\//.test(ua)
          ? 'chrome'
          : /Safari\//.test(ua)
            ? 'safari'
            : 'other'

  let inApp: string | null = null
  for (const [pattern, name] of IN_APP) {
    if (pattern.test(ua)) {
      inApp = name
      break
    }
  }
  if (!inApp) {
    // Android System WebView marks itself with "; wv)".
    if (android && /;\s*wv\)/.test(ua)) inApp = 'an app'
    // Every real iOS browser carries "Safari/"; a bare WebKit user agent is a web view inside an app.
    else if (ios && /AppleWebKit\//.test(ua) && !/Safari\//.test(ua)) inApp = 'an app'
  }
  return { platform, browser, inApp }
}

export type MicFailureKind = 'denied' | 'dismissed' | 'missing' | 'busy' | 'unsupported' | 'insecure' | 'failed'

export interface MicFailure {
  kind: MicFailureKind
  /** The DOMException name, when there was one. */
  name: string
  /** The browser's own message. */
  detail: string
  /** How long getUserMedia took to fail. A person cannot answer a prompt in a fraction of a second. */
  elapsedMs: number
  /** What the Permissions API said afterwards. */
  permission: 'granted' | 'denied' | 'prompt' | 'unknown'
}

export type MicCause =
  'in-app' | 'site' | 'os' | 'dismissed' | 'missing' | 'busy' | 'unsupported' | 'insecure' | 'unknown'

/** Faster than a person can read and answer a permission prompt. */
const NO_PROMPT_MS = 400

/** The most likely reason for a failure in this environment. */
export function diagnoseMic(failure: MicFailure, env: Environment): MicCause {
  switch (failure.kind) {
    case 'missing':
    case 'busy':
    case 'unsupported':
    case 'insecure':
      return env.inApp && failure.kind === 'unsupported' ? 'in-app' : failure.kind
    case 'failed':
      return env.inApp ? 'in-app' : 'unknown'
    default:
      break
  }
  if (env.inApp) return 'in-app'
  if (/system|by the os|operating system/i.test(failure.detail)) return 'os'
  if (failure.kind === 'dismissed' || /dismiss/i.test(failure.detail)) return 'dismissed'
  if (failure.permission === 'denied') {
    // The site is on the browser's block list. (Chrome on Android also reports this when the OS switch is off.)
    return 'site'
  }
  if (failure.elapsedMs >= NO_PROMPT_MS) {
    // There was time for a prompt, and the browser has not recorded a block: it was closed or declined once.
    return failure.permission === 'prompt' ? 'dismissed' : 'site'
  }
  // Refused instantly without a recorded site block: the browser itself has no access, or the answer is remembered.
  return failure.permission === 'prompt' ? 'os' : 'unknown'
}

export interface MicHelp {
  title: string
  /** One or two plain sentences. */
  explanation: string
  /** Sections of numbered steps, most likely first. */
  sections: { heading: string; steps: string[] }[]
  /** Offer "Copy link" so the page can be opened in a real browser. */
  offerCopyLink: boolean
}

function browserLabel(env: Environment): string {
  switch (env.browser) {
    case 'safari':
      return 'Safari'
    case 'chrome':
      return 'Chrome'
    case 'firefox':
      return 'Firefox'
    case 'edge':
      return 'Edge'
    case 'samsung':
      return 'Samsung Internet'
    default:
      return 'your browser'
  }
}

/** Steps to allow the microphone for this site in the browser's own settings. */
export function siteSteps(env: Environment): string[] {
  const b = browserLabel(env)
  if (env.platform === 'ios') {
    if (env.browser === 'safari')
      return [
        'Tap the page menu at the left of the address bar (the icon with lines under a rectangle, “aA” on older versions).',
        'Tap “Website Settings”.',
        'Set Microphone to “Ask” or “Allow”, then tap Done.',
        'Come back here and press “Try again”.',
      ]
    return [
      `Open the Settings app, then Apps, then ${b}.`,
      'Turn Microphone on.',
      'Come back here, reload the page and press “Try again”. Tap Allow when asked.',
    ]
  }
  if (env.platform === 'android') {
    if (env.browser === 'firefox')
      return [
        'Tap the padlock at the left of the address bar.',
        'Tap the Microphone permission and allow it (or clear the blocked permission).',
        'Press “Try again”.',
      ]
    return [
      'Tap the icon at the left of the address bar (a padlock or a tune icon).',
      'Tap “Permissions”, then “Microphone”, and choose Allow.',
      'Press “Try again”.',
    ]
  }
  if (env.browser === 'safari')
    return [
      'In the menu bar choose Safari, then “Settings for This Website…”.',
      'Set Microphone to “Ask” or “Allow”.',
      'Press “Try again”.',
    ]
  if (env.browser === 'firefox')
    return [
      'Click the permissions icon at the left of the address bar.',
      'Remove “Blocked” next to “Use the microphone”.',
      'Press “Try again” and choose Allow when asked.',
    ]
  return [
    'Click the icon at the left of the address bar (a tune icon or a padlock).',
    'Set Microphone to “Allow” (or open “Site settings” and allow it there).',
    'Press “Try again”. Reload the page if the browser asks you to.',
  ]
}

/** Steps to give the browser itself access to the microphone in the operating system. */
export function osSteps(env: Environment): string[] {
  const b = browserLabel(env)
  switch (env.platform) {
    case 'ios':
      return [
        `Open the Settings app, then Apps, then ${b} (on iOS 17 and earlier: Settings, then ${b}).`,
        env.browser === 'safari'
          ? 'Under “Settings for Websites”, tap Microphone and choose “Ask” or “Allow”.'
          : 'Turn Microphone on.',
        'If Screen Time is on: Settings, Screen Time, Content & Privacy Restrictions, Microphone must allow changes.',
        'Return here and press “Try again”.',
      ]
    case 'android':
      return [
        `Open Settings, then Apps, then ${b}, then Permissions.`,
        'Tap Microphone and choose “Allow only while using the app”.',
        'Also check the quick-settings panel: the “Mic access” switch must be on.',
        'Return here and press “Try again”.',
      ]
    case 'macos':
      return [
        'Open System Settings, then Privacy & Security, then Microphone.',
        `Turn on the switch next to ${b}.`,
        `Quit and reopen ${b} if macOS asks, then press “Try again”.`,
      ]
    case 'windows':
      return [
        'Open Settings, then Privacy & security, then Microphone.',
        'Turn on “Microphone access” and “Let desktop apps access your microphone”.',
        'Press “Try again”.',
      ]
    default:
      return [
        'Check that your desktop’s privacy or sound settings allow applications to use the microphone.',
        'Check that an input device is selected and not muted.',
        'Press “Try again”.',
      ]
  }
}

function realBrowserName(env: Environment): string {
  return env.platform === 'ios' ? 'Safari' : env.platform === 'android' ? 'Chrome' : 'your usual browser'
}

/** The explanation and steps shown in the microphone help panel. */
export function micHelp(failure: MicFailure, env: Environment): MicHelp {
  const cause = diagnoseMic(failure, env)
  const b = browserLabel(env)
  switch (cause) {
    case 'in-app': {
      const app = env.inApp ?? 'an app'
      const real = realBrowserName(env)
      return {
        title: 'This looks like a browser inside another app',
        explanation: `You seem to have opened this page inside ${app}. Built-in browsers like that often cannot use the microphone at all, and they do not show a permission prompt. The page works in ${real}.`,
        sections: [
          {
            heading: `Open this page in ${real}`,
            steps: [
              env.platform === 'ios'
                ? 'Tap the share or “…” button of the in-app browser and choose “Open in Safari” (or “Open in browser”).'
                : 'Tap the “⋮” menu of the in-app browser and choose “Open in Chrome” (or “Open in browser”).',
              `If there is no such button, press “Copy link” below, open ${real} and paste it into the address bar.`,
            ],
          },
          { heading: 'If you are already in a normal browser', steps: siteSteps(env) },
        ],
        offerCopyLink: true,
      }
    }
    case 'site':
      return {
        title: 'The microphone is blocked for this site',
        explanation: `${b} has a saved “block” for this site, so it refused without asking. Nothing is wrong with your microphone. Change the saved setting and try again.`,
        sections: [
          { heading: `Allow the microphone for this site in ${b}`, steps: siteSteps(env) },
          { heading: 'If that setting already says Allow', steps: osSteps(env) },
        ],
        offerCopyLink: false,
      }
    case 'os':
      return {
        title: `${b === 'your browser' ? 'Your browser' : b} is not allowed to use the microphone`,
        explanation: `The request was refused immediately, before any prompt could appear. That usually means the device’s own settings do not let ${b} use the microphone.`,
        sections: [
          { heading: `Give ${b} access to the microphone`, steps: osSteps(env) },
          { heading: 'Then check the setting for this site', steps: siteSteps(env) },
        ],
        offerCopyLink: false,
      }
    case 'dismissed':
      return {
        title: 'The microphone prompt was closed',
        explanation:
          'The browser asked for permission and the prompt was closed or declined, so nothing was recorded. Press “Try again” and choose Allow. The audio stays on this device unless you are using a server.',
        sections: [{ heading: 'If no prompt appears next time', steps: siteSteps(env) }],
        offerCopyLink: false,
      }
    case 'missing':
      return {
        title: 'No microphone was found',
        explanation:
          'The browser is allowed to record, but it could not find an input device. Plug in or switch on a microphone or headset, then try again.',
        sections: [{ heading: 'Things to check', steps: osSteps(env).slice(0, -1).concat('Press “Try again”.') }],
        offerCopyLink: false,
      }
    case 'busy':
      return {
        title: 'The microphone is in use or unavailable',
        explanation:
          'Permission is fine, but the device could not be opened. Another app (a call, a recorder) may be holding it, or it was unplugged.',
        sections: [
          {
            heading: 'Things to try',
            steps: [
              'End any call or recording in another app or tab.',
              'Unplug and reconnect the headset or microphone.',
              'Press “Try again”.',
            ],
          },
        ],
        offerCopyLink: false,
      }
    case 'insecure':
      return {
        title: 'Microphones only work on secure pages',
        explanation:
          'Browsers allow microphone access on HTTPS and on localhost only. This page was opened over plain HTTP from another machine.',
        sections: [
          {
            heading: 'What to do',
            steps: [
              'Open the console through https://, or on the machine itself as http://localhost.',
              'Or put the server behind a reverse proxy that provides HTTPS.',
              'Until then, upload a file or play a sample.',
            ],
          },
        ],
        offerCopyLink: false,
      }
    case 'unsupported':
      return {
        title: 'This browser cannot capture audio',
        explanation:
          'The page needs getUserMedia and AudioWorklet, which this browser does not offer. A current version of Safari, Chrome, Edge or Firefox will work.',
        sections: [],
        offerCopyLink: true,
      }
    default:
      return {
        title: 'The microphone could not be opened',
        explanation: `The browser refused the request${failure.detail ? ` (“${failure.detail}”)` : ''}. The two usual reasons are a saved block for this site and the device not allowing ${b} to use the microphone.`,
        sections: [
          { heading: `Check the setting for this site in ${b}`, steps: siteSteps(env) },
          { heading: `Check that ${b} may use the microphone`, steps: osSteps(env) },
        ],
        offerCopyLink: Boolean(env.inApp),
      }
  }
}

export const MIC_FAILURE_KINDS: readonly string[] = [
  'denied',
  'dismissed',
  'missing',
  'busy',
  'unsupported',
  'insecure',
  'failed',
]
