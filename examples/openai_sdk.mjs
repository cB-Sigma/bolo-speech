// bolo through the official OpenAI Node SDK.
//
//   npm install openai
//   node examples/openai_sdk.mjs recording.wav
//
// NOTE: this file was written against the SDK documentation and has NOT been
// executed by the maintainers (the Python example next to it has).
import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: (process.env.BOLO_URL ?? "http://127.0.0.1:8000") + "/v1",
  apiKey: process.env.BOLO_API_KEY ?? "not-needed",
});

const speech = await client.audio.speech.create({
  model: "tts-1",
  voice: "alloy",
  input: "Hello from bolo.",
  response_format: "wav",
});
fs.writeFileSync("hello.wav", Buffer.from(await speech.arrayBuffer()));

const file = process.argv[2] ?? "hello.wav";
const text = await client.audio.transcriptions.create({
  model: "whisper-1",
  file: fs.createReadStream(file),
});
console.log(text.text);

const stream = await client.audio.transcriptions.create({
  model: "whisper-1",
  file: fs.createReadStream(file),
  stream: true,
});
for await (const event of stream) {
  if (event.type === "transcript.text.delta") process.stdout.write(event.delta);
}
process.stdout.write("\n");
