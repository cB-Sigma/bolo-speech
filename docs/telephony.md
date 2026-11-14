# Phone calls

bolo can answer and place phone calls and connect them to a voice agent. A
telephony provider (Twilio, Plivo, Telnyx or Exotel) or a SIP trunk carries
the call; bolo is the agent on the line: it hears the caller, asks your
language model what to say and speaks the answer, with the same Realtime
session that backs `WS /v1/realtime` ([voice-agents.md](voice-agents.md)):
turn detection, barge-in, the three pipelines, tools and traces.

* [What you need](#what-you-need)
* [Configuration](#configuration)
* [Security](#security)
* [Agents](#agents) and [numbers](#numbers)
* [What happens on a call](#what-happens-on-a-call)
* [Setting up a provider](#setting-up-a-provider): [Twilio](#twilio), [Plivo](#plivo), [Telnyx](#telnyx), [Exotel](#exotel)
* [Outbound calls](#outbound-calls) and [voicemail](#voicemail)
* [Call records](#call-records)
* [The call simulator](#the-call-simulator)
* [SIP](#sip)
* [API](#api)
* [What was verified, and what was not](#what-was-verified-and-what-was-not)

**No real call has been made with this code.** It was written against the
providers' published protocols and is tested against fake providers that
speak those protocols (see [the last section](#what-was-verified-and-what-was-not)
for exactly what was read and what was assumed). Expect to adjust details the
first time you connect a real account, and please report them.

## What you need

* a bolo server reachable from the internet over **HTTPS** (providers call
  webhooks and open a `wss://` WebSocket to it; plain `http://` is refused by
  all of them). A reverse proxy with a certificate, or a tunnel while you
  try it out;
* a language model (`llm_base_url`), as for any voice agent;
* an account and a phone number at one of the providers, or a SIP trunk;
* low latency between the provider's media servers and bolo: host bolo in the
  provider's region.

## Configuration

Everything is under one setting, `telephony`: a `[telephony]` table in the
configuration file (see [configuration.md](configuration.md)).

```toml
[telephony]
public_base_url = "https://voice.example.org"   # how the providers reach this server
default_agent = "front-desk"                    # answers numbers that are not mapped below
# validate_signatures = false                   # explicit opt-out, see Security

[telephony.providers.twilio]
account_sid = "AC..."
auth_token = "..."                              # or BOLO_TELEPHONY_TWILIO_AUTH_TOKEN
from_number = "+14155550100"                    # caller ID of outbound calls

[telephony.agents.front-desk]
instructions = "You answer the phone for Example Clinic. Be brief and friendly."
greeting = "Example Clinic, how can I help?"
voice = "amy"
language = "en"
transfer_number = "+14155550199"
recording_consent = "This call is transcribed for quality purposes."

[telephony.numbers]
"+14155550100" = "front-desk"
```

In the environment: `BOLO_TELEPHONY` holds the whole table as JSON;
`BOLO_TELEPHONY_PUBLIC_BASE_URL`, `BOLO_TELEPHONY_DEFAULT_AGENT`,
`BOLO_TELEPHONY_VALIDATE_SIGNATURES` and
`BOLO_TELEPHONY_<PROVIDER>_<FIELD>` (for example
`BOLO_TELEPHONY_TELNYX_API_KEY`) add to it, which keeps credentials out of
the file.

| Key | Meaning |
| --- | --- |
| `public_base_url` | the public `https://` address of this server. Used to build the webhook and stream URLs and to check signatures (a signature covers the exact URL the provider called) |
| `validate_signatures` | `true` (default). `false` accepts requests nobody signed; see [Security](#security) |
| `default_agent` | the agent for calls to numbers that are not in `numbers`. Unset: such calls are refused |
| `call_buffer` | call records kept in memory (default 200) |
| `providers.<name>` | credentials of `twilio`, `plivo`, `telnyx`, `exotel`; fields below |
| `agents.<name>` | voice agents; fields in [Agents](#agents) |
| `numbers` | `"<called number>" = "<agent>"` |

Provider fields:

| Provider | Fields | Needed for |
| --- | --- | --- |
| `twilio` | `auth_token` | checking signatures (inbound calls need nothing else) |
| | `account_sid`, `auth_token` (or `api_key_sid` + `api_key_secret` for the REST calls), `from_number` | placing calls |
| `plivo` | `auth_token` | checking signatures |
| | `auth_id`, `auth_token`, `from_number` | placing calls |
| `telnyx` | `public_key` (Mission Control Portal > Keys & Credentials > Public Key) | checking signatures |
| | `api_key` | answering calls (Telnyx is driven by commands), hanging up, transfer |
| | `connection_id` (the Voice API application), `from_number` | placing calls |
| `exotel` | `stream_username`, `stream_password` (you choose them; they go into the applet's URL) | admitting the media stream |
| | `account_sid`, `api_key`, `api_token`, `app_id` (the flow to connect), `from_number` (an ExoPhone), `subdomain` (default `api.exotel.com`) | placing calls |
| | `sample_rate` | `8000` (default), `16000` or `24000`, matching `?sample-rate=` in the applet URL |

Credentials are never logged, never returned by any API (`GET /api/v1/telephony`
shows `***` for a secret that is set) and are removed from provider error
messages before those are shown.

`bolo telephony status` prints what is configured, what is missing and the
URLs to paste into each dashboard; the console's **Phone** page shows the
same.

## Security

The provider endpoints (`/telephony/{provider}/...`) cannot sit behind the
server's API key: a provider has no way to send it. Instead **every request
is checked against the provider's own proof of origin**, and the endpoints of
a provider stay closed until that check is possible:

```
POST /telephony/twilio/voice
503 {"error": {"code": "telephony_not_configured", "message": "the Twilio endpoints are closed: bolo cannot
check request signatures because telephony.providers.twilio.auth_token is not set; telephony.public_base_url
is not set (the address the provider reaches this server at). Configure them (docs/telephony.md), or set
telephony.validate_signatures = false to accept unauthenticated requests (not recommended)."}}
```

| Provider | What is checked |
| --- | --- |
| Twilio | `X-Twilio-Signature`: HMAC-SHA1 of the URL and the sorted POST fields with the auth token |
| Plivo | `X-Plivo-Signature-V3` (or `-Ma-V3`) with `X-Plivo-Signature-V3-Nonce`: HMAC-SHA256 with the auth token |
| Telnyx | `telnyx-signature-ed25519` over `"<telnyx-timestamp>|<body>"` with the account's public key; requests more than five minutes old are refused. The same headers on the media WebSocket (signed text `"<timestamp>|<stream_url>"`) are checked when present |
| Exotel | Exotel signs nothing. Its applet sends the HTTP Basic credentials you put in the URL; bolo compares them with `stream_username` / `stream_password`. Ask Exotel for IP whitelisting as a second layer |

The **media WebSocket** is protected by a one-time token: the answer webhook
(which was just verified) hands out a stream URL ending in a random token;
a stream without a valid, unused token is refused before the WebSocket is
accepted. That is why nothing has to be configured for the stream at Twilio,
Plivo or Telnyx. (Twilio's documentation says the `X-Twilio-Signature` of the
WebSocket handshake must be validated but does not say what is signed, so
bolo does not rely on it.)

A wrong signature answers 403 (401 for Exotel's credentials) and a line in
the log that names the usual cause: `public_base_url` differs from the URL in
the provider's dashboard (scheme, host, a trailing slash).

`validate_signatures = false` is the explicit opt-out: unsigned requests and
streams without a token are accepted, and when `public_base_url` is unset the
address is taken from the request's `Host` / `X-Forwarded-*` headers. It is
meant for a first experiment behind a tunnel, not for a server anyone can
reach: whoever can reach it can start a session with your language model.

The operator's API (`/api/v1/telephony/...`) is behind the server's API key
like every other native route. Agents and numbers can be changed through it
only on a loopback bind or with an API key.

## Agents

An agent is what answers: instructions, a voice and how it behaves on the
line. Agents live in the configuration (`[telephony.agents.<name>]`,
read-only at run time) or are added through the API or the console (stored
in `<data_dir>/telephony.json`, mode 0600).

| Field | Default | Meaning |
| --- | --- | --- |
| `instructions` | the server's `llm_instructions` | the system prompt |
| `greeting` | none | said when the call is answered; without one the caller speaks first |
| `voice`, `language` | server defaults | the voice; a language hint for speech recognition |
| `stt_model`, `tts_model`, `llm_model` | `default_realtime_stt`, `default_realtime_tts`, `llm_model` | models for this agent |
| `pipeline` | `realtime_pipeline` | `cascade`, `audio-llm` or `speech-to-speech` |
| `romanised` | the voice's setting | `off`, `auto` or `on`: romanised text in what the agent says (the greeting, prompts, the language model's answers) is written in the voice's script first; never an error on a call, see api.md, "Romanised text" |
| `turn_detection` | server VAD | passed to the Realtime session, e.g. `{ type = "semantic_vad" }` or `{ silence_duration_ms = 400 }` |
| `tools` | `[]` | function tools, see below |
| `max_duration_s` | `600` | longest call (`0`: no limit); `max_duration_message` is said before hanging up |
| `silence_timeout_s` | `10` | seconds of silence from the caller before `silence_prompt` ("Are you still there?") is said; `0`: never |
| `silence_prompts` | `2` | prompts before the agent says `goodbye` (if set) and hangs up |
| `transfer_number` | none | where `transfer_call` sends the caller; `transfer_message` is said first |
| `recording_consent` | none | an announcement said first on every call and never interrupted |
| `voicemail` | `hangup` | outbound calls answered by a machine: `hangup`, `leave_message` (says `voicemail_message`) or `ignore` (no detection) |
| `dtmf` | `message` | key presses are given to the agent as a user message; `ignore` only records them |
| `summary` | `true` | write a summary with the language model when the call ends |

**Tools.** Two are built in and offered to the language model on every call:
`hang_up` (ends the call once what the agent said has been played) and, when
the agent has a `transfer_number`, `transfer_call`. The model cannot dial a
number of its own choosing: a transfer always goes to the configured number.
Your own tools are function definitions with a `url`:

```toml
[[telephony.agents.front-desk.tools]]
name = "find_appointment"
description = "Look up a caller's next appointment by phone number or name."
url = "https://internal.example.org/tools/find_appointment"
[telephony.agents.front-desk.tools.parameters]
type = "object"
properties = { name = { type = "string" } }
```

When the model calls it, bolo POSTs
`{"name", "arguments": {...}, "call": {"id", "provider", "direction", "agent", "from", "to", "dtmf"}}`
to the URL (10 s timeout, `timeout_s` to change it) and gives the response
body to the model as the tool's output; the model then answers the caller.
The URL is never shown to the model. A tool without a `url` answers "not
available".

## Numbers

`[telephony.numbers]` maps a called number to an agent. Numbers are compared
by their digits, and by their last ten digits when the full forms differ, so
`+91 80 4567 8900` and `08045678900` meet. Calls to other numbers go to
`default_agent`, or are refused when there is none (Twilio `<Reject>`, Plivo
`<Hangup reason="rejected">`, a Telnyx hangup, SIP `404`).

## What happens on a call

1. The provider announces the call (webhook, or the stream itself for
   Exotel). bolo picks the agent, creates the call record and answers with
   the provider's document or command pointing at the stream URL.
2. The stream connects. Audio is 8 kHz μ-law for Twilio, Plivo and Telnyx
   (A-law and linear PCM are understood when the provider announces them),
   16-bit linear PCM for Exotel. bolo resamples it to 16 kHz for recognition
   and sends the agent's speech back in the provider's format.
3. The **recording-consent announcement**, then the **greeting**, are spoken
   with the agent's voice. The greeting is added to the conversation so the
   model knows what was said.
4. Turns: the Realtime session detects the end of the caller's speech,
   transcribes, asks the model, speaks. Each turn is timed (see
   [Call records](#call-records)).
5. **Barge-in.** Providers play what they receive in real time while bolo
   synthesises faster than that, so several seconds of speech can be queued
   at the provider. When the caller starts speaking, bolo stops generating,
   tells the provider to drop its queue (Twilio / Telnyx / Exotel `clear`,
   Plivo `clearAudio`; over SIP bolo's own RTP queue), and truncates the
   assistant's message to what was actually played, so the model's history
   says what the caller heard. The consent announcement and a voicemail
   message are never interrupted.
6. **Key presses** (DTMF) arrive as events. They interrupt the agent like
   speech does; after 1.5 s without another key (or at once on `#`) the
   agent receives `[The caller pressed 4 2 # on the phone keypad]` as a user
   message and answers. Tools receive all digits pressed so far in
   `call.dtmf`.
7. **Silence.** When the agent has finished and the caller says nothing for
   `silence_timeout_s`, the agent asks `silence_prompt`; after
   `silence_prompts` unanswered prompts it says `goodbye` and hangs up
   (outcome `no_response`).
8. The call ends when the caller hangs up, the agent calls `hang_up`, the
   call is transferred, `max_duration_s` is reached, or the provider reports
   the end.

How bolo **hangs up and transfers** depends on the provider:

| Provider | Hang up | Transfer to a person |
| --- | --- | --- |
| Twilio | bolo closes the stream; Twilio then runs the TwiML after `<Connect>`, a `<Redirect>` to `/telephony/twilio/after/<token>`, which answers `<Hangup/>` | the same redirect answers `<Dial>number</Dial>` |
| Plivo | the same with `<Stream keepCallAlive="true">` and `<Redirect>`; `/after` answers `<Hangup/>` | `/after` answers `<Dial><Number>number</Number></Dial>` |
| Telnyx | the `hangup` command | the `transfer` command with `to` |
| Exotel | bolo closes the stream and the flow moves to the next applet | put a **Passthru** applet after the Voicebot applet pointing at `/telephony/exotel/after`: it answers `200` when a transfer was requested (wire that branch to a **Connect** applet that dials your number) and `302` otherwise (wire it to Hangup). The number dialled is the one in the Connect applet, not `transfer_number` |
| SIP | `BYE` | not implemented (the tool is not offered) |

## Setting up a provider

The dashboard paths below are how the providers' consoles were organised
when this was written; the labels move. What matters is the URL and the
method. `bolo telephony status` and the console's Phone page print the URLs
for your `public_base_url`.

### Twilio

1. Configuration: `[telephony.providers.twilio]` with `auth_token` (and
   `account_sid`, `from_number` to place calls).
2. Twilio Console > Phone Numbers > Manage > Active numbers > your number >
   Voice Configuration: **A call comes in** = Webhook,
   `https://<your host>/telephony/twilio/voice`, HTTP POST.
3. Optional: **Call status changes** =
   `https://<your host>/telephony/twilio/status`, HTTP POST.

bolo answers with:

```xml
<Response>
  <Connect><Stream url="wss://<your host>/telephony/twilio/stream/<token>"/></Connect>
  <Redirect method="POST">https://<your host>/telephony/twilio/after/<token></Redirect>
</Response>
```

`<Connect><Stream>` makes the stream bidirectional; Twilio runs the
`<Redirect>` only once bolo closes the WebSocket. Stream URLs cannot carry a
query string at Twilio, which is why the token is part of the path.

### Plivo

1. Configuration: `[telephony.providers.plivo]` with `auth_token` (and
   `auth_id`, `from_number` to place calls).
2. Plivo console > Voice > Applications > XML > New Application: **Primary
   Answer URL** = `https://<your host>/telephony/plivo/voice`, method POST;
   optionally **Hangup URL** = `https://<your host>/telephony/plivo/status`.
3. Phone Numbers > your number > assign the application.

bolo answers with:

```xml
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://<your host>/telephony/plivo/stream/<token></Stream>
  <Redirect method="POST">https://<your host>/telephony/plivo/after/<token></Redirect>
</Response>
```

### Telnyx

1. Configuration: `[telephony.providers.telnyx]` with `public_key` and
   `api_key` (and `connection_id`, `from_number` to place calls).
2. Mission Control Portal > Voice > Programmable Voice > create a Voice API
   application: **Send a webhook to the URL** =
   `https://<your host>/telephony/telnyx/voice`, API v2. Assign your number
   to the application. `connection_id` is the application's ID.

Telnyx sends `call.initiated`; bolo answers the call with the `answer`
command carrying `stream_url`, `stream_track: inbound_track`,
`stream_bidirectional_mode: rtp` and `stream_bidirectional_codec: PCMU`.
All later events (`call.hangup`, machine detection) arrive at the same URL.

### Exotel

Exotel has no answer document: the call flow is built in its dashboard.

1. Configuration: `[telephony.providers.exotel]` with a `stream_username`
   and `stream_password` of your choosing.
2. App Bazaar > create a flow > add the **Voicebot** applet (the
   bidirectional one; the Stream applet is one-way) with the URL
   `wss://<stream_username>:<stream_password>@<your host>/telephony/exotel/stream`.
   Exotel sends those credentials as HTTP Basic authentication. The agent is
   chosen by the called number, or by a custom parameter `agent=<name>`.
   For wideband audio add `?sample-rate=16000` and set `sample_rate = 16000`.
3. For transfers: a **Passthru** applet after it with
   `https://<stream_username>:<stream_password>@<your host>/telephony/exotel/after`,
   its 200 branch to a Connect applet, its 302 branch to Hangup. bolo
   expects the same Basic credentials on this request; whether Exotel's
   Passthru applet sends credentials placed in the URL is not documented, so
   test a transfer before relying on it.
4. Assign the flow to your ExoPhone.

The applet can also ask an HTTPS endpoint for the WebSocket address:
`GET /telephony/exotel/voice` answers `{"url": "wss://.../telephony/exotel/stream/<token>"}`.
What Exotel sends to such an endpoint is not documented, so the static URL
above is the path to prefer.

bolo sends Exotel audio in 100 ms chunks (3,200 bytes at 8 kHz), which is the
smallest chunk Exotel documents, in multiples of 320 bytes.

## Outbound calls

```bash
curl -X POST https://voice.example.org/api/v1/telephony/calls \
  -H "Authorization: Bearer $BOLO_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider": "twilio", "to": "+14155550123", "agent": "front-desk"}'
```

bolo asks the provider to place the call (`from` defaults to the provider's
`from_number`) and to come back to it for the audio:

| Provider | Request |
| --- | --- |
| Twilio | `POST /2010-04-01/Accounts/{sid}/Calls.json` with `Url` (the answer webhook, with a token), `StatusCallback`, `MachineDetection` |
| Plivo | `POST /v1/Account/{auth_id}/Call/` with `answer_url`, `hangup_url`, `machine_detection` |
| Telnyx | `POST /v2/calls` with `connection_id`, `webhook_url`, the `stream_*` fields, `answering_machine_detection` |
| Exotel | `POST /v1/Accounts/{sid}/Calls/connect` with `From` (the number to ring), `CallerId`, `Url` = `http://my.exotel.com/{sid}/exoml/start_voice/{app_id}`: the callee is connected to your flow, whose Voicebot applet reaches bolo |

The answer (202) is the call record; poll `GET /api/v1/telephony/calls/{id}`.
A call that never connects ends with the outcome `no_answer`, `busy`,
`failed` or `canceled`. `POST /api/v1/telephony/calls/{id}/hangup` ends a
call in progress.

> **Compliance is yours.** Outbound calling is regulated almost everywhere:
> consent, calling hours, caller identification, do-not-call registers,
> disclosure that a machine is speaking, recording rules. In India that
> includes TRAI's commercial-communication regulations (TCCCPR) and the
> do-not-disturb (DND) registry; in the United States the TCPA and state
> laws; elsewhere their equivalents. bolo checks none of this and keeps no
> do-not-call list. Do not place calls with it until you know the rules that
> apply to you.

### Voicemail

Answering-machine detection is the provider's; bolo acts on the result
according to the agent's `voicemail` setting.

| Provider | Detection used | `hangup` | `leave_message` |
| --- | --- | --- | --- |
| Twilio | `MachineDetection=Enable`: `AnsweredBy` arrives with the answer webhook | bolo answers `<Hangup/>` and never opens a stream | `MachineDetection=DetectMessageEnd`: when Twilio reports `machine_end_*`, the stream opens, bolo says `voicemail_message` and hangs up |
| Plivo | `machine_detection=hangup` | Plivo hangs up itself | `machine_detection=true` with `machine_detection_url`: on `Machine=true` bolo waits until the far end has been quiet for 1.2 s (at most 30 s), says the message and hangs up |
| Telnyx | `answering_machine_detection=detect`, webhook `call.machine.detection.ended` | `hangup` command on `result: machine` | `greeting_end`: bolo speaks after `call.machine.greeting.ended` |
| Exotel | none documented | not available | not available |

The outcome of such a call is `voicemail`, and `answered_by` on the record
says `human`, `machine` or `unknown`. `not_sure` / `unknown` are treated as a
person, as Telnyx recommends. Detection is a guess by the provider: expect
both kinds of error.

## Call records

Every call has a record, kept in memory like request traces (the last
`call_buffer` calls): `GET /api/v1/telephony/calls` lists them,
`GET /api/v1/telephony/calls/{id}` returns one.

```json
{
  "id": "call_000004_31337", "provider": "twilio", "direction": "inbound", "agent": "front-desk",
  "from": "+14155550123", "to": "+14155550100", "status": "ended", "outcome": "caller_hangup",
  "duration_s": 18.2, "pipeline": "cascade", "interruptions": 1, "dtmf": "",
  "summary": "A caller asked about opening hours and was told the desk is open until six.",
  "latency": {"turns": 2, "response_ms": {"p50": 1061.0, "max": 1180.0, "mean": 1120.5}, "...": "..."},
  "timeline": [
    {"t_ms": 0.0, "type": "stream_started", "encoding": "mulaw", "sample_rate": 8000},
    {"t_ms": 4.1, "type": "greeting", "text": "Example Clinic, how can I help?"},
    {"t_ms": 4900.2, "type": "user_speech_stopped", "audio_ms": 4560},
    {"t_ms": 5461.0, "type": "agent_audio_started", "turn": 0, "response_ms": 1061.0},
    {"t_ms": 7900.3, "type": "barge_in", "cause": "speech", "heard_ms": 2400, "cleared_ms": 1800}
  ],
  "turns": [{
    "index": 0, "kind": "speech", "user_text": "What time do you close today?",
    "assistant_text": "We are open until six this evening.", "interrupted": true, "tool_calls": [],
    "latency": {"endpoint_ms": 500.0, "stt_ms": 90.0, "llm_first_token_ms": 310.0,
                "tts_first_audio_ms": 160.0, "send_ms": 1.0, "total_ms": 561.0, "response_ms": 1061.0}
  }],
  "transcript": [
    {"t_ms": 4.1, "speaker": "agent", "text": "Example Clinic, how can I help?", "kind": "greeting"},
    {"t_ms": 4990.0, "speaker": "caller", "text": "What time do you close today?"},
    {"t_ms": 5600.0, "speaker": "agent", "text": "We are open until six this evening.", "interrupted": true,
     "full_text": "We are open until six this evening. Is there anything else I can help you with?"}
  ]
}
```

**Latency of a turn**, in the order things happen:

| Field | From → to |
| --- | --- |
| `endpoint_ms` | the end of the caller's speech → the turn detector deciding it has ended (its silence window; `silence_duration_ms`, 500 ms by default) |
| `stt_ms` | → the transcript |
| `llm_first_token_ms` | → the first token of the language model |
| `tts_first_audio_ms` | → the first synthesised audio |
| `send_ms` | → the first audio frame handed to the provider's socket |
| `total_ms` | the four parts after the detector: what the server spent |
| `response_ms` | `endpoint_ms + total_ms`: what the caller waits, up to the provider. The network and the provider's own buffering come on top and are not visible to bolo |

Turns that did not start with speech (`kind`: `dtmf`, `tool`, `prompt`) have
no `endpoint_ms` or `stt_ms`. For the `audio-llm` and `speech-to-speech`
pipelines the transcript runs beside the answer, so `stt_ms` is not part of
the path.

**Outcomes:** `caller_hangup`, `agent_hangup`, `transferred`, `max_duration`,
`no_response`, `voicemail`, `no_answer`, `busy`, `failed`, `canceled`,
`error`.

**Summary:** when a language model is configured and the agent's `summary`
is on, the transcript is summarised once the call ends (`summary`, or
`summary_error`).

Each turn is also an ordinary request trace of kind `agent` with transport
`telephony`, `call_id`, `agent` and `telephony_provider`
(`GET /api/v1/requests?kind=agent`). `GET /api/v1/stats` has a `telephony`
block, `/metrics` has `bolo_telephony_active_calls` and
`bolo_telephony_calls_total{provider,outcome}`, and the event stream
(`GET /api/v1/events`) sends `call` events when a call starts, ends and gets
its summary.

Records hold transcripts of what callers said. They live in memory only and
are gone after a restart; bolo does not record audio.

## The call simulator

A scripted caller rings your agent through a fake provider, so you can test
an agent, compare models and measure latency without an account or a phone.

```bash
bolo telephony simulate                          # the "basic" scenario through a fake Twilio
bolo telephony simulate --all --provider exotel  # every built-in scenario, Exotel's protocol
bolo telephony simulate --scenario my-call.json --agent front-desk --noise-db 15 --json
```

`bolo telephony simulate` starts a private copy of the server on a loopback
port with **your** settings (models, language model, agents) and made-up
provider credentials, then takes the whole path a provider takes: the signed
answer webhook, the document bolo returns, the WebSocket media stream and
the `after` webhook. `--direct` skips HTTP and feeds the call session
directly; `POST /api/v1/telephony/simulate` and the console's **Simulate a
call** button do that inside a running server.

The fake provider behaves like a phone line: it sends 8 kHz audio in 20 ms
frames in real time (100 ms for Exotel), plays what bolo sends at real
speed, drops its queue on `clear`, and returns marks when playback reaches
them. That is what makes barge-in and latency measurable.

```
barge-in: PASS  (plivo, agent front-desk, 13.11 s, outcome caller_hangup)
  caller heard the agent after: p50 792 ms, max 802 ms (end of the caller's speech to the first audio at the provider)
  server breakdown (p50): endpoint 465 ms  stt 44 ms  llm_first_token 78 ms  tts_first_audio 82 ms  send 1 ms
  agent audio heard 7.2 s; barge-in clears 1 (3.28 s dropped)
  [ok] barge_in: 1 barge-in(s) recorded, 1 clear(s) reached the provider, 3.3 s dropped
  [ok] turns_min: 2 answered turn(s)
```

(These numbers are from the test-suite's fake engines and a fake language
model; they show the report, not the speed of any real model.)

Built-in scenarios: `basic`, `barge-in`, `silence`, `dtmf`, `noisy`. A
scenario file:

```json
{
  "description": "Ask for the opening hours and interrupt the answer",
  "noise_db": 20,
  "steps": [
    {"wait_for_agent": true},
    {"say": "Hello, what time do you close today?"},
    {"interrupt": {"after_s": 0.8, "say": "Sorry, I meant tomorrow."}},
    {"wait_for_agent": true},
    {"dtmf": "1#"},
    {"silence": 2.0},
    {"hangup": true}
  ],
  "expect": [
    {"type": "agent_says", "contains": "tomorrow"},
    {"type": "barge_in"},
    {"type": "max_response_ms", "value": 1500},
    {"type": "outcome", "is": "caller_hangup"}
  ]
}
```

Steps: `say` (spoken with the server's text-to-speech; `voice` and `model`
choose which), `audio` (a file), `speech_s` (a synthetic speech-like signal
for model-free tests), `silence`, `dtmf`, `interrupt` (start speaking
`after_s` seconds into the agent's reply), `wait_for_agent`, `wait_for_end`
(until bolo hangs up), `hangup`. `noise_db` adds white noise at that
signal-to-noise ratio. Expectations: `agent_spoke`, `agent_says`,
`caller_heard_as` (what recognition made of the caller), `barge_in`,
`tool_called`, `timeline` (an event occurred), `turns_min`, `outcome`,
`max_response_ms`, `after` (what the provider was told at the end:
`hangup` or `dial:<number>`).

The report's `caller.response_ms` is measured at the fake provider: from the
end of the caller's utterance to the first agent audio arriving. It includes
the detector's silence window, like `response_ms` on the record, plus
loopback transport. A real call adds the network and the provider.

Limits: the caller's voice is a text-to-speech voice or your recordings, not
a person on a handset; the noise is white noise, not a street; there is no
packet loss, jitter or echo. The simulator tells you whether the agent
behaves and how fast the server is, not how a real line sounds.

## SIP

For a SIP trunk or a PBX, bolo has a minimal inbound SIP endpoint.

```bash
pip install 'bolo-speech[sip]'
```

```toml
sip = true
sip_port = 5060
sip_public_ip = "203.0.113.20"        # the address written into SDP answers
sip_allowed_ips = ["198.51.100.0/24"] # your trunk's signalling addresses
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `sip` | `false` | listen for SIP |
| `sip_host`, `sip_port` | `0.0.0.0`, `5060` | UDP listening address |
| `sip_public_ip` | the local address the INVITE reached | the media address in SDP answers; set it behind NAT |
| `sip_allowed_ips` | `[]`: loopback only | source addresses and networks allowed to send requests; everything else gets `403` |
| `sip_rtp_port_min`, `sip_rtp_port_max` | `40000`, `40100` | UDP ports for RTP, one per call |

Point the trunk at `sip:<number>@<your host>:5060` over UDP. The called
number (the user part of the request URI) picks the agent through
`[telephony.numbers]`, as for the providers; calls appear among the call
records with provider `sip`.

**What it does:** accepts `INVITE` with an SDP offer, answers with PCMU or
PCMA (the caller's first choice of the two) at 8 kHz over RTP and
`telephone-event` when offered; reads key presses sent as RTP telephone
events (RFC 4733, formerly RFC 2833), one per event however often it is
repeated; `ACK`; `BYE` from either side; `CANCEL`; `OPTIONS` (keep-alive
pings); answers retransmitted requests and repeats its `200 OK` until the
`ACK` arrives; sends RTP every 20 ms (silence between utterances, which
keeps NAT bindings open) and answers RTP where it comes from (symmetric
RTP). Barge-in drops bolo's own RTP queue, so it takes effect within one
packet. A call without RTP for 60 s is ended.

**What it does not do:** `REGISTER` (bolo is a trunk endpoint, not a phone),
digest authentication (callers are admitted by source address only), TCP,
TLS and SRTP (signalling and audio are **not encrypted**: use it on a
private network or a VPN to the trunk), an `INVITE` without an SDP offer
(`488`), re-INVITE (hold, codec change: `488`), `REFER` (so no transfer),
in-band DTMF tones, `SIP INFO` DTMF, G.722 / G.729 / Opus, outbound calls,
RTCP, a jitter buffer (packets are used in arrival order).

**Design.** Signalling is one asyncio UDP socket. SIP messages are parsed
and written with [sipmessage](https://github.com/spacinov/sipmessage)
(BSD-2-Clause, pure Python, typed); the dialog and transaction handling the
list above needs, SDP offer/answer and RTP are about 550 lines in
`bolo/telephony/sip.py`. A call is the same `CallSession` the provider
streams use, fed with neutral stream events instead of a provider's JSON.

Libraries considered (licences and last releases as listed on PyPI on
2026-11-14):

| Library | Licence | Last release | Why not (or why) |
| --- | --- | --- | --- |
| sipmessage 0.7.1 | BSD-2-Clause | 2026-02 | **used**: message parsing and serialising only, no dependencies |
| aiortc 1.16 | BSD-3-Clause | 2026-11 | already bolo's WebRTC transport; has RTP and codecs but no SIP, and its RTP is tied to ICE/DTLS |
| voip-utils 0.5.1 (Home Assistant) | Apache-2.0 | 2026-10 | a SIP/RTP server for Home Assistant's phones; depends on a system Opus library and is built around Opus |
| sippy 2.5.1 (B2BUA) | BSD | 2026-11 | a full back-to-back user agent with compiled dependencies; far more than an endpoint needs |
| aiovoip 0.3.2 | not declared on PyPI | 2025-12 | a fork of aiosip; no licence metadata |
| aiosip 0.2.0 | Apache-2.0 | 2017-09 | unmaintained |
| PySIPio 1.8.0 | MIT | 2024-11 | a client that registers with a server; brings a cloud TTS package and pydub |
| pyVoIP 1.6.8 | GPL-3.0 | 2024-01 | copyleft; needs the `audioop` module removed in Python 3.13 |
| pjsua2 (PJSIP) | GPL-2.0 or commercial | 2023-07 on PyPI | copyleft; a compiled stack |

**Tested** against the test-suite's own SIP user agent over loopback UDP
(INVITE, ACK, RTP both ways, RFC 4733 key presses, BYE from either side,
PCMU and PCMA, the refusals). **Not tested** against any trunk, PBX or
softphone: no independent SIP client could be installed here without system
packages (the pure-Python candidate, pyVoIP, needs `audioop` and
registration). Treat it as experimental until it has met a real trunk.

## API

Provider endpoints (no API key; signature or credentials instead):

| Route | Called by |
| --- | --- |
| `POST /telephony/{provider}/voice` | the answer webhook (Twilio, Plivo; Telnyx sends every event here; `GET` for Exotel's dynamic URL) |
| `POST /telephony/{provider}/voice/{token}` | the same for a call bolo placed |
| `WS /telephony/{provider}/stream/{token}` | the media stream (`/stream` without a token for Exotel) |
| `POST /telephony/{provider}/after/{token}` | Twilio / Plivo `<Redirect>` once the stream is over; `GET /telephony/exotel/after` for the Passthru applet |
| `POST /telephony/{provider}/status[/{token}]` | call-status / hangup callbacks |
| `POST /telephony/plivo/amd/{token}` | Plivo's machine-detection callback |

Operator API (behind the API key):

| Route | Purpose |
| --- | --- |
| `GET /api/v1/telephony` | providers (state, what is missing, webhook URLs), agents, numbers, statistics; no secrets |
| `GET /api/v1/telephony/agents`, `GET` / `PUT` / `DELETE /api/v1/telephony/agents/{name}` | agents |
| `GET /api/v1/telephony/numbers`, `PUT` / `DELETE /api/v1/telephony/numbers/{number}` | `PUT` body `{"agent": "<name>"}` |
| `GET /api/v1/telephony/calls?limit=&provider=&agent=&status=`, `GET /api/v1/telephony/calls/{id}` | call records |
| `POST /api/v1/telephony/calls` | place a call: `{"provider", "to", "agent", "from", "metadata"}` |
| `POST /api/v1/telephony/calls/{id}/hangup` | end a call |
| `GET /api/v1/telephony/scenarios`, `POST /api/v1/telephony/simulate` | the simulator: `{"agent", "provider", "scenario", "noise_db"}` |

`GET /api/v1/info` reports `capabilities.telephony` (providers, which are
exposed, the SIP endpoint).

## What was verified, and what was not

The provider protocols were read from the providers' own documentation on
**2026-11-14**. "Verified" means: stated on the page named. Nothing was
verified against a live account.

### Twilio

Verified ([WebSocket messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages),
[`<Stream>`](https://www.twilio.com/docs/voice/twiml/stream),
[Media Streams](https://www.twilio.com/docs/voice/media-streams),
[Call resource](https://www.twilio.com/docs/voice/api/call-resource),
[AMD](https://www.twilio.com/docs/voice/answering-machine-detection),
[webhook security](https://www.twilio.com/docs/usage/webhooks/webhooks-security)):

* messages from Twilio `connected`, `start` (`streamSid`, `callSid`, `mediaFormat` `audio/x-mulaw` 8000, `customParameters`), `media` (`track`, `chunk`, `timestamp`, base64 `payload`), `dtmf` and `mark` (bidirectional streams only), `stop`;
* messages to Twilio `media`, `mark`, `clear`, each with `streamSid`; the payload must be base64 μ-law at 8000 Hz without file headers;
* `<Connect><Stream>` is bidirectional; `wss` only; no query string in the URL; `<Parameter>` for custom values; TwiML after `<Connect>` runs only once the server closes the WebSocket; one bidirectional stream per call;
* creating a call (`To`, `From`, `Url`, `StatusCallback`, `MachineDetection` = `Enable` / `DetectMessageEnd`), updating one (`Url`, `Twiml`, `Status`);
* `AnsweredBy` values and that synchronous detection delivers it with the webhook to `Url`;
* the signature algorithm; bolo's implementation reproduces the worked example of that page bit for bit.

Not verified: how the WebSocket handshake's `X-Twilio-Signature` is computed
(the page says to validate it and does not say what is signed; bolo uses the
token instead); the `<Reject>`, `<Dial>`, `<Redirect>` and `<Hangup>` verbs
and the field names of the voice and status webhooks (`CallSid`, `From`,
`To`, `CallStatus`) were written from long-standing knowledge of TwiML and
not re-read; the dashboard labels.

### Plivo

Verified ([Stream XML](https://www.plivo.com/docs/voice/xml/audio-streaming),
[audio streaming reference](https://docs.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference),
[signature validation](https://www.plivo.com/docs/voice/concepts/signature-validation),
[make a call](https://www.plivo.com/docs/voice/api/call/make-a-call),
[machine detection](https://www.plivo.com/docs/voice/concepts/machine-detection)):

* `<Stream>` attributes `bidirectional`, `keepCallAlive` ("subsequent XML executes only after stream disconnects"), `contentType` (`audio/x-l16;rate=8000|16000|24000`, `audio/x-mulaw;rate=8000`), `extraHeaders`, `streamTimeout`;
* events from Plivo `start` (`callId`, `streamId`, `mediaFormat`), `media`, `dtmf`, `playedStream`, `clearedAudio`, and `extra_headers` on them; to Plivo `playAudio` (`contentType`, `sampleRate`, `payload`), `checkpoint`, `clearAudio`;
* signature V3: HMAC-SHA256 with the auth token, Base64, a nonce header, sorted POST fields appended to the URL, the `-Ma-V3` variant, several signatures separated by commas;
* the call API fields, `request_uuid`, `machine_detection` = `true` / `hangup`, `machine_detection_url` with `Machine=true`.

Not verified: the exact separators of the signed string. Plivo's page gives
two examples that disagree with each other; bolo builds it the way Plivo's
SDKs do (`<url>.<name1><value1>...<nameN><valueN>.<nonce>` for POST). **This
is the detail most likely to need a correction**: check it with a captured
request (`BOLO_TEST_TELEPHONY_CAPTURE`, below). Also not re-read: the
`<Redirect>`, `<Dial><Number>` and `<Hangup>` elements, the answer webhook's
field names (`CallUUID`, `From`, `To`), `Machine=true` on the hangup callback
(from a legacy support article), whether the stream WebSocket is signed. The
default `contentType` differs between Plivo's pages; bolo always sets it. No
WebSocket `stop` event is documented: bolo treats the socket closing as the
end.

### Telnyx

Verified ([media streaming](https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming),
[answer](https://developers.telnyx.com/api-reference/call-commands/answer-call),
[dial](https://developers.telnyx.com/api-reference/call-commands/dial),
[transfer](https://developers.telnyx.com/api-reference/call-commands/transfer-call),
[AMD](https://developers.telnyx.com/docs/voice/programmable-voice/answering-machine-detection),
[webhooks](https://developers.telnyx.com/docs/voice/programmable-voice/receiving-webhooks)):

* `stream_url`, `stream_track`, `stream_bidirectional_mode` (`mp3` default, `rtp`), `stream_bidirectional_codec` (`PCMU` default, `PCMA`, `G722`, `OPUS`, `AMR-WB`, `L16`) on `answer` and `dial`;
* events `connected`, `start` (`call_control_id`, `from`, `to`, `media_format`), `media`, `dtmf`, `mark`, `stop`, `error`; to Telnyx `media`, `clear`, `mark`; `clear` returns queued marks;
* the WebSocket carries `telnyx-signature-ed25519` and `telnyx-timestamp`, signed text `"<timestamp>|<stream_url>"`; webhooks are signed over `"<timestamp>|<body>"`; a five-minute tolerance is recommended;
* the `call.initiated` payload; dial's fields and `call_control_id` in its answer; detection modes and `result` values (`human`, `machine`, `not_sure`; `ended`, `beep_detected`); `transfer` with `to`.

Not verified: the `hangup` command's page was not among those read (bolo
calls `POST /v2/calls/{id}/actions/hangup`); `hangup_cause` values other
than `timeout`; the form of the public key (bolo takes the Base64 or hex of
the 32 raw bytes; one of Telnyx's examples parses an X.509 wrapper instead);
the dashboard labels. Telnyx's TeXML (`<Connect><Stream>`) is an alternative
bolo does not use.

### Exotel

Verified ([Stream and Voicebot applet](https://developer.exotel.com/docs/agentstream/stream-voicebot-applet),
[Passthru](https://developer.exotel.com/docs/voice-v1/applets/passthru),
[connect to a flow](https://developer.exotel.com/docs/voice-v1/api-reference/connect-to-flow)):

* the Voicebot applet takes a `wss://` URL or an HTTPS endpoint that returns `{"url": ...}`; `sample-rate` 8000 / 16000 / 24000; up to three custom parameters; authentication by Basic credentials in the URL or IP whitelisting, no signatures;
* events `connected`, `start` (`stream_sid`, `call_sid`, `from`, `to`, `custom_parameters`, `media_format`), `media`, `dtmf` (`digit`, `duration`), `stop` (`reason`), `mark`; to Exotel `media`, `mark`, `clear`, each with `stream_sid`;
* audio: 16-bit linear PCM, mono, little-endian, base64; chunks of at least 3.2 kB, at most 100 kB, in multiples of 320 bytes;
* Passthru: `200` goes to one branch, `302` to the other;
* the connect-to-flow API (`From`, `CallerId`, `Url`, `StatusCallback`, `CustomField`) and its `Call.Sid`.

Not verified: what Exotel sends to a dynamic-URL endpoint and to a Passthru
applet (bolo reads `CallSid`, `CallFrom`, `CallTo` from the query string);
whether closing the WebSocket from bolo's side moves the flow to the next
applet (the page says the stream closes before the next applet, not who may
close it); the `.json` suffix bolo uses to get a JSON answer from the API;
the status callback's fields; the Connect applet's dynamic "dial whom"
format (so the transfer number has to be set in the applet); regional API
hosts; answering-machine detection (none documented).

### Check it yourself

Two opt-in tests exist for the owner of an account (skipped by default):

```bash
# 1. a webhook captured from the real provider, checked against bolo's signature code
BOLO_TEST_REAL=1 BOLO_TEST_TELEPHONY_CAPTURE=capture.json pytest tests/test_telephony.py -k real_webhook_signature
#    capture.json: {"provider": "plivo", "url": "https://...", "method": "POST", "headers": {...},
#                   "form": {...}, "body": "...", "secret": "<auth token or Telnyx public key>"}

# 2. one real outbound call through a running, publicly reachable `bolo serve` (costs money, rings a phone)
BOLO_TEST_REAL=1 BOLO_TEST_TELEPHONY=twilio BOLO_TEST_TELEPHONY_TO=+1... \
  BOLO_TEST_TELEPHONY_SERVER=https://voice.example.org BOLO_API_KEY=... pytest tests/test_telephony.py -k real_outbound_call
```
