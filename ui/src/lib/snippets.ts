export interface SnippetContext {
  /** Server base URL without a trailing slash, e.g. http://127.0.0.1:8000 */
  baseUrl: string
  sttModel: string
  ttsModel: string
  voice: string
  auth: boolean
}

export interface Snippet {
  id: string
  label: string
  language: string
  blocks: { title: string; code: string }[]
  note?: string
}

function wsBase(baseUrl: string): string {
  return baseUrl.replace(/^http/, 'ws')
}

function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).hostname
  } catch {
    return '127.0.0.1'
  }
}

/** Copy-paste client examples, filled in with this server's address and the chosen models. */
export function buildSnippets(c: SnippetContext): Snippet[] {
  const key = c.auth ? '$BOLO_API_KEY' : 'not-needed'
  const curlAuth = c.auth ? ` \\\n  -H "Authorization: Bearer $BOLO_API_KEY"` : ''
  const wsKey = c.auth ? '&api_key=$BOLO_API_KEY' : ''
  return [
    {
      id: 'curl',
      label: 'curl',
      language: 'bash',
      blocks: [
        {
          title: 'Transcribe a file',
          code: `curl ${c.baseUrl}/v1/audio/transcriptions${curlAuth} \\\n  -F file=@speech.wav \\\n  -F model=${c.sttModel}`,
        },
        {
          title: 'Synthesise speech',
          code: `curl ${c.baseUrl}/v1/audio/speech${curlAuth} \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${c.ttsModel}", "voice": "${c.voice}", "input": "Hello from bolo."}' \\\n  --output hello.mp3`,
        },
        {
          title: 'Pull a model',
          code: `curl -X POST ${c.baseUrl}/api/v1/models/pull${curlAuth} \\\n  -H "Content-Type: application/json" \\\n  -d '{"id": "${c.sttModel}"}'`,
        },
      ],
    },
    {
      id: 'python',
      label: 'Python',
      language: 'python',
      note: 'Uses the official OpenAI SDK: pip install openai',
      blocks: [
        {
          title: 'Transcribe and speak',
          code: `from openai import OpenAI

client = OpenAI(base_url="${c.baseUrl}/v1", api_key="${c.auth ? 'YOUR_BOLO_API_KEY' : key}")

with open("speech.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(model="${c.sttModel}", file=audio)
print(transcript.text)

with client.audio.speech.with_streaming_response.create(
    model="${c.ttsModel}",
    voice="${c.voice}",
    input="Hello from bolo.",
    response_format="wav",
) as response:
    response.stream_to_file("hello.wav")`,
        },
      ],
    },
    {
      id: 'javascript',
      label: 'JavaScript',
      language: 'javascript',
      note: 'Uses the official OpenAI SDK: npm install openai',
      blocks: [
        {
          title: 'Transcribe and speak',
          code: `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${c.baseUrl}/v1", apiKey: ${c.auth ? 'process.env.BOLO_API_KEY' : `"${key}"`} });

const transcript = await client.audio.transcriptions.create({
  model: "${c.sttModel}",
  file: fs.createReadStream("speech.wav"),
});
console.log(transcript.text);

const speech = await client.audio.speech.create({
  model: "${c.ttsModel}",
  voice: "${c.voice}",
  input: "Hello from bolo.",
});
fs.writeFileSync("hello.mp3", Buffer.from(await speech.arrayBuffer()));`,
        },
      ],
    },
    {
      id: 'websocket',
      label: 'WebSocket',
      language: 'python',
      note: 'Streaming recognition with partial results: pip install websockets',
      blocks: [
        {
          title: 'Stream 16 kHz PCM and print partials',
          code: `import asyncio, json, wave
import websockets

URL = "${wsBase(c.baseUrl)}/api/v1/stt/stream?model=${c.sttModel}&sample_rate=16000&encoding=pcm_s16le&interim=true&vad=true${wsKey}"

async def main():
    async with websockets.connect(URL) as ws:
        async def send():
            with wave.open("speech.wav", "rb") as wav:  # 16 kHz, mono, 16-bit
                while chunk := wav.readframes(320):      # 20 ms
                    await ws.send(chunk)
                    await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "stop"}))

        sender = asyncio.create_task(send())
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "partial":
                print("…", event["text"], f'({event["latency_ms"]} ms)')
            elif event["type"] == "final":
                print("✓", event["text"], f'({event["latency_ms"]} ms)')
            elif event["type"] == "done":
                break
        await sender

asyncio.run(main())`,
        },
        {
          title: 'Telephony audio (8 kHz μ-law) uses the same endpoint',
          code: `${wsBase(c.baseUrl)}/api/v1/stt/stream?model=${c.sttModel}&sample_rate=8000&encoding=mulaw${wsKey}`,
        },
      ],
    },
    {
      id: 'home-assistant',
      label: 'Home Assistant',
      language: 'yaml',
      note: 'bolo speaks the Wyoming protocol on TCP port 10300 (speech to text and text to speech).',
      blocks: [
        {
          title: 'Add the integration',
          code: `Settings → Devices & services → Add integration → Wyoming Protocol

Host: ${hostOf(c.baseUrl)}
Port: 10300`,
        },
        {
          title: 'Then pick bolo in your voice assistant',
          code: `Settings → Voice assistants → your assistant
  Speech-to-text: bolo (${c.sttModel})
  Text-to-speech: bolo (${c.ttsModel})`,
        },
      ],
    },
  ]
}
