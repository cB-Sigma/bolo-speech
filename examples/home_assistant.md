# bolo in Home Assistant (Wyoming protocol)

> **Not run by the maintainers.** bolo's Wyoming server is tested with the
> `wyoming` Python client (see `wyoming_client.py` and the test-suite), which
> is the library Home Assistant uses, but no Home Assistant instance was
> available while bolo was developed. Please report how it goes.

Home Assistant talks to local speech services through the
[Wyoming protocol](https://www.home-assistant.io/integrations/wyoming/).
bolo serves both speech-to-text and text-to-speech on one Wyoming port, so it
can stand in for the separate Whisper and Piper add-ons.

1. Start bolo where Home Assistant can reach it. Wyoming has no
   authentication, so bind it to a trusted interface only:

   ```bash
   bolo pull moonshine-tiny-en          # or faster-whisper-small for more languages
   bolo pull kokoro-v1.0-int8           # English/Mandarin voices with no extra installs
   bolo serve --wyoming --wyoming-host 0.0.0.0 --wyoming-port 10300
   ```

   Voices in Piper format (for example `piper-hi-in-rohan-medium`) additionally
   need `espeak-ng` on the machine that runs bolo: `sudo apt install espeak-ng`.

2. In Home Assistant: **Settings → Devices & services → Add integration →
   Wyoming Protocol**, then enter the host and port `10300`. One integration
   entry appears with a speech-to-text and a text-to-speech entity.

3. **Settings → Voice assistants → Add assistant**: choose *bolo* for both
   Speech-to-text and Text-to-speech. The language and voice lists come from
   bolo's installed models: every installed STT model is offered, and every
   voice of every installed TTS model. Multi-voice models appear once per
   language (for example `kokoro-v1.0-int8:en-US`) with their voices as speakers.

4. Wake words: with a wake-word model installed (`bolo pull kws-zipformer-gigaspeech`, or
   `pip install 'bolo-speech[wake]'` for the microWakeWord models), the same entry also offers a wake-word entity,
   and the assistant's wake word can be detected by bolo for satellites that stream to Home Assistant. A keyword
   spotter takes a phrase of your own as `<model>:<phrase>` (for example `kws-zipformer-gigaspeech:hey computer`)
   where a client lets you type the name. The openWakeWord phrase models are CC BY-NC-SA 4.0 (non-commercial).

Notes

* Home Assistant sends a whole voice command, then waits for the transcript,
  so the latency that matters is bolo's batch decode time. A realtime-tier
  model keeps that well under a second on a small PC (see the README for
  measurements).
* `describe` only lists installed models. Pull the models you want first, or
  leave `auto_pull` on and call them once through the HTTP API.
* The same server keeps serving the OpenAI-compatible API and the web console
  on port 8000, which is handy for watching latency while you talk to Assist.
