# Pipelines

A pipeline chains bolo's models into one job: clean a recording, find who spoke when, transcribe it, restore
punctuation, mask personal information, ask a question about it, call a webhook with the answer. Each piece is a
**step** that runs one of the operations bolo already serves on its own endpoints (the same code, models, scheduler,
placement and keep-alive), so a pipeline does nothing a series of API calls could not; it does them in one request,
on the server, with the intermediate audio never leaving it.

* [A first pipeline](#a-first-pipeline)
* [The definition](#the-definition): steps, inputs and references, options, models
* [Step types](#step-types): model steps and utility steps, with their inputs and outputs
* [Control flow](#control-flow): conditions, switches, per-speaker fan-out, filters
* [Validation](#validation)
* [Running](#running): synchronous, asynchronous jobs, webhooks, results and files
* [API](#api) and [CLI](#cli)
* [Templates](#templates)
* [Settings](#settings)

## A first pipeline

```json
{
  "id": "meeting-notes",
  "name": "Meeting notes",
  "steps": [
    {"id": "clean", "type": "denoise"},
    {"id": "who", "type": "diarization"},
    {"id": "stt", "type": "stt", "inputs": {"audio": "{{steps.clean.audio}}", "turns": "{{steps.who.turns}}"}},
    {"id": "punct", "type": "punctuation"},
    {"id": "summary", "type": "llm",
     "options": {"prompt": "Summarise this meeting as notes with action items:\n\n{{steps.punct.text}}"}}
  ],
  "outputs": {"notes": "{{steps.summary.text}}", "transcript": "{{steps.punct.transcript}}"}
}
```

```bash
bolo pipeline create meeting-notes.json
bolo pipeline run meeting-notes standup.wav
curl -F file=@standup.wav http://127.0.0.1:8000/v1/pipelines/meeting-notes/run
```

The response lists every step with its model, status, time and output, then the pipeline's `outputs`.

## The definition

A pipeline is a JSON object (YAML is accepted wherever JSON is: files ending in `.yaml`/`.yml`, and request bodies
sent as `application/yaml`).

| Field | Meaning |
| --- | --- |
| `id` | Lower-case letters, digits, `-` and `_` (taken from `name` when left out). `runs`, `templates`, `validate` and `step-types` are reserved. |
| `name`, `description` | For people. |
| `mode` | `file` (the default and, for now, the only mode): runs on a recording or a text. |
| `vars` | Default values the steps can use as `{{vars.name}}`; a run can override them. |
| `steps` | The steps, in order (required, at least one). |
| `outputs` | What the run returns, by name: references or templates. Left out: the last step's outputs. |
| `version` | Set by the server: every saved change is a new version. |

### Steps

```json
{"id": "stt", "type": "stt", "model": "auto", "inputs": {"audio": "{{steps.clean.audio}}"},
 "options": {"language": "{{vars.language}}"}, "when": {"value": "{{steps.lid.language}}", "in": ["en", "hi"]},
 "on_error": "continue"}
```

| Field | Meaning |
| --- | --- |
| `id` | Unique in the pipeline (letters, digits, `_`, `-`); later steps refer to its outputs as `steps.<id>.<output>`. Defaults to the type with `-` written `_`, numbered when repeated (`stt`, `stt_2`, `split_speakers`). |
| `type` | A [step type](#step-types). |
| `model` | A catalog model id (or alias) of the step's kind, or `auto` (the default): the server's default for that kind (`default_stt`, `default_denoise`, ...). Cloud models (`openai-gpt-4o-transcribe`, ...) are catalog models like any other. |
| `inputs` | The step's inputs by slot name. Each is a reference (`{{steps.clean.audio}}`), a text with references in it (`"Caller said: {{steps.stt.text}}"`), or a literal value. Slots left out take their default (below). |
| `options` | The step's options (each type lists its own; strings may hold references). Unknown options are an error. |
| `when` | A [condition](#conditions): the step runs only when it holds (otherwise it is `skipped`). |
| `on_error` | `fail` (default: the run stops, the steps before keep their outputs) or `continue` (the step is marked `error`, its outputs are empty and the run goes on). |

### References

Inside `{{ }}`:

| Reference | Value |
| --- | --- |
| `input.audio` | The audio the run was given (file upload or URL). |
| `input.text` | The text the run was given (`text` field). |
| `input.filename`, `input.duration` | The upload's file name and length in seconds. |
| `vars.<name>` | A variable (the definition's `vars`, overridden by the run's). |
| `steps.<id>.<output>` | An output of an earlier step. A path goes deeper: `steps.stt.segments.0.text`, `steps.decide.answer`. |
| `speaker.audio`, `speaker.label`, `speaker.seconds`, `speaker.turns` | Inside a [split-speakers](#per-speaker-fan-out) step. |
| `item` | Inside a [filter](#filters): the element being tested (`item.speaker`). |
| `run.id`, `pipeline.id`, `pipeline.version` | The run and the pipeline. |

A value that is a single reference keeps its type (audio stays audio, a list stays a list). A reference inside a longer
text is written out as text: a transcript as its text, numbers and booleans as written, lists and objects as JSON.
Audio cannot be written into text; that is a validation error.

Default inputs: a slot of type `audio` left out takes the audio of the closest earlier step that produced audio
(`denoise`, `tts`, `voice-conversion`, `watermark`), else `input.audio`; a slot of type `text` takes the text of the
closest earlier step that produced text, else `input.text`. Validation reports what each default resolved to, so the
console can show it.

### Value types

`audio`, `text`, `transcript` (an object with `text`, `language`, `duration`, `segments` and `words`; it is accepted
wherever text is), `turns` (speaker turns: `[{"speaker", "start", "end"}]`), `number`, `boolean`, `list`, `object`.

## Step types

Model steps (the kind is the catalog task the `model` must have):

| Type | Kind | Inputs | Options | Outputs |
| --- | --- | --- | --- | --- |
| `denoise` | denoise | `audio` | `strength` (0 to 1) | `audio` |
| `vad` | vad | `audio` | `threshold`, `min_silence_ms`, `min_speech_ms`, `speech_pad_ms` | `segments` (list of `{start, end}` seconds), `speech_seconds`, `count` |
| `diarization` | diarization | `audio` | `num_speakers`, `min_speakers`, `max_speakers`, `threshold` | `turns`, `speakers` (labels), `num_speakers` |
| `separation` | separation | `audio` | `diarization_model`, `min_overlap_s`, `allow_narrowband` | `turns`, `separation` (what was separated, each track as a WAV data URL) |
| `speaker-identification` | speaker-embedding | `audio` | `speakers` (enrolled ids; default all), `threshold` | `speaker` (name or empty), `speaker_id`, `similarity`, `match` |
| `speaker-enrolment` | speaker-embedding | `audio`, `name` | | `speaker_id`, `clips` |
| `stt` | stt | `audio`, `turns` (optional: labels each segment with the speaker who talks most during it) | `language`, `prompt`, `hotwords`, `temperature`, `diarize` (diarize inside the recogniser), `vad_filter`, `word_timestamps` | `text`, `language`, `duration`, `segments`, `words`, `transcript` |
| `translation` | stt (audio in) or the language model (text in) | `audio` or `text` | `target_language` (default `en`), `source_language` | `text`, `language` |
| `align` | align | `audio`, `text` | `language` | `words` (`[{word, start, end}]`), `text` |
| `punctuation` | punctuation | `text` | | `text`, `transcript` |
| `itn` | itn | `text` | | `text`, `transcript` |
| `pii` | pii | `text` | `entities` (`PERSON`, `PHONE_NUMBER`, ...) | `text`, `transcript`, `redactions`, `count` |
| `lid` | lid | `audio` | | `language`, `probability`, `languages` |
| `audio-tagging` | audio-tagging | `audio` | `top_k`, `threshold`, `labels` | `tags` (`[{label, score}]`), `top` |
| `emotion` | emotion | `audio` | | `emotion`, `score`, `emotions` |
| `audio-llm` | audio-llm | `audio`, `prompt` | `max_tokens`, `temperature` | `text` |
| `decision` | decision | `text` or `audio` | `question`, `choices` (list; left out: yes/no), `instructions` | `answer`, `confidence`, `reason` |
| `tts` | tts | `text` | `voice` (a model voice or a saved, cloned voice), `speed`, `language`, `instructions`, `watermark` | `audio` |
| `voice-conversion` | voice-conversion | `audio` | `voice` | `audio` |
| `watermark` | watermark | `audio` | | `audio` |
| `watermark-detect` | watermark | `audio` | | `detected`, `score` |

Utility steps (no model):

| Type | Inputs | Options | Outputs |
| --- | --- | --- | --- |
| `template` | | `template` (text with references) | `text` |
| `llm` | | `prompt`, `system`, `json` (true: ask for and parse a JSON object), `max_tokens`, `temperature` | `text`, `json` |
| `webhook` | `body` (default: every step's outputs so far, as JSON) | `url`, `method` (`POST`), `headers`, `timeout_s` | `status`, `ok`, `body`, `json` |
| `save` | `value` | `name` (a file name; `.wav`, `.mp3`, `.flac`, `.txt`, `.json`, `.srt`, `.vtt`) | `file`, `url` |
| `filter` | `items` | `where` (a condition on `item`) | `items`, `count` |
| `condition` | | `if`, `then` (steps), `else` (steps) | `branch` (`then` or `else`) |
| `switch` | `value` | `cases` (value: steps), `default` (steps) | `branch` (the case taken, or `default`) |
| `split-speakers` | `audio`, `turns` | `steps` (run once per speaker), `min_seconds` | `speakers` (`[{speaker, seconds, outputs}]`), `count` |
| `stop` | | `reason` | ends the run successfully here |

`llm` uses the configured language model (`llm_base_url`, `llm_model`); `audio-llm` the audio language model
(`audio_llm_model`, or a catalog `audio-llm` model given as `model`). `decision` asks the configured language model
for a JSON answer about a text (about a recording when it has no text: the audio language model) and fails when the
answer is not one of `choices`. `translation` of a text uses the language model; of a recording, a recogniser that
translates (`model`: Whisper, Canary, Voxtral, ...).

`webhook` steps sign their body ([below](#webhooks-and-signatures)); an answer of HTTP 400 or more fails the
step. `save` writes audio as WAV, MP3, FLAC or Opus, a transcript or a list of segments as SRT or WebVTT, any value
as JSON, and text as it is.

## Control flow

The steps run in order. Branches and fan-out hold steps of their own; a step inside one can use everything before it
outside, and steps after the branch can use the outputs of steps inside it (empty when that branch did not run).

### Conditions

```json
{"value": "{{steps.decide.answer}}", "equals": "yes"}
{"value": "{{steps.emotion.score}}", "gte": 0.7}
{"all": [{"value": "{{steps.lid.language}}", "in": ["en", "hi"]}, {"not": {"value": "{{steps.stt.text}}", "matches": "^\\s*$"}}]}
```

Operators: `equals`, `not_equals`, `in`, `not_in`, `contains`, `matches` (a regular expression), `gt`, `gte`, `lt`,
`lte`, `exists` (true or false), `truthy` (true or false); combined with `all`, `any` and `not`. Text comparisons ignore
case and surrounding spaces.

```json
{"id": "route", "type": "condition", "options": {
  "if": {"value": "{{steps.decide.answer}}", "equals": "urgent"},
  "then": [{"id": "page", "type": "webhook", "options": {"url": "https://ops.example/page"}}],
  "else": [{"id": "log", "type": "save", "options": {"name": "voicemail.txt"}, "inputs": {"value": "{{steps.stt.text}}"}}]
}}
```

`switch` takes one value and a list of steps per case:

```json
{"id": "dept", "type": "switch", "inputs": {"value": "{{steps.route.answer}}"}, "options": {
  "cases": {"billing": [...], "technical": [...]}, "default": [...]}}
```

### Per-speaker fan-out

`split-speakers` cuts the audio into one track per speaker (from `turns`, by default the closest earlier
`diarization` step) and runs its `steps` once per speaker, with `speaker.audio` as the default audio:

```json
{"id": "per", "type": "split-speakers", "options": {"steps": [
  {"id": "mood", "type": "emotion"},
  {"id": "words", "type": "stt"}
]}}
```

Its output `speakers` is `[{"speaker": "A", "seconds": 41.2, "outputs": {"mood": {...}, "words": {...}}}]`.

### Filters

`filter` keeps the elements of a list for which `where` holds: `{"type": "filter", "inputs": {"items":
"{{steps.stt.segments}}"}, "options": {"where": {"value": "{{item.speaker}}", "equals": "A"}}}`.

## Validation

`POST /api/v1/pipelines/validate` (and every create, update and run) checks the whole definition before anything
runs, and lists every problem at once, each with the path of the field (`steps[2].inputs.audio`):

* unknown step types, duplicate or malformed ids, unknown options and options of the wrong type;
* models that are not in the catalog, or not of the step's kind (`whisper-small` in a `tts` step);
* references to steps that do not exist or come later, to outputs the step does not have, and to unknown namespaces;
* type mismatches (a `transcript` into an `audio` slot; audio written into a text);
* required inputs with nothing to default to (an `align` step with no text anywhere before it);

Warnings do not stop a run: a model not installed yet (it is pulled first when `auto_pull` is on), a branch step whose
output is used after the branch, a language model step when no language model is configured.

The answer is `{"valid": bool, "errors": [{"path", "message"}], "warnings": [...], "steps": [...]}`, where `steps`
lists each step with its resolved model, its inputs after defaults and its outputs with their types.

## Running

### Synchronously

`POST /v1/pipelines/{id}/run`, multipart:

| Field | Meaning |
| --- | --- |
| `file` | The recording. |
| `url` | Or an `http(s)` URL the server downloads the recording from (see `pipeline_remote_urls`). |
| `text` | A text input (`input.text`), for pipelines that start from text. |
| `vars` | A JSON object overriding the pipeline's `vars`. |
| `version` | Run an older version. |
| `async` | `true`: start a job and answer at once (below). |
| `callback_url` | With `async`: POST the result here when the run ends. |

`POST /v1/pipelines/run` runs a definition sent in the `pipeline` field (JSON or YAML) without saving it (the
console's test runs).

The answer is the run:

```json
{
  "id": "run_8f2c1d9a0b3e", "object": "pipeline.run", "pipeline": "meeting-notes", "version": 3,
  "status": "ok", "created_at": "...", "finished_at": "...", "ms": 5310.4,
  "steps": [
    {"id": "clean", "type": "denoise", "model": "gtcrn-simple", "status": "ok", "ms": 410.2,
     "output": {"audio": {"type": "audio", "seconds": 61.3, "sample_rate": 16000, "file": "clean.wav",
                          "url": "/api/v1/pipelines/runs/run_8f2c1d9a0b3e/files/clean.wav"}}},
    {"id": "stt", "type": "stt", "model": "whisper-small", "status": "ok", "ms": 3120.9, "output": {"text": "..."}}
  ],
  "outputs": {"notes": "..."},
  "error": null
}
```

Step `status` is `ok`, `error`, `skipped` (its `when` did not hold, or its branch was not taken) or `pending`
(the run stopped before it). A failed run has `status: "error"` and `error: {"step", "message", "code"}`, and keeps
every output made before the failure. The HTTP status is 200 for a run that ran (even when a step failed: read
`status`) and 400 for a definition or input that does not validate.

Audio outputs are written to the run's folder as WAV and listed with a `url`; `save` steps write files there too.

### As a job, with a webhook

With `async=true` the answer is `202` with the run in state `queued`; poll `GET /api/v1/pipelines/runs/{run_id}`
(also visible as a job at `GET /api/v1/jobs/{job_id}`). With `callback_url`, the finished run is POSTed there.
This also serves as an asynchronous transcription job: a one-step `stt` pipeline with a callback.

### Webhooks and signatures

Callbacks and `webhook` steps carry:

* `X-Bolo-Event`: `pipeline.run.completed`, `pipeline.run.failed` or `pipeline.step` (a webhook step);
* `X-Bolo-Run`: the run id;
* `X-Bolo-Signature`: `t=<unix seconds>,v1=<hex HMAC-SHA256 of "<t>.<raw body>" with the webhook secret>`.

The secret is the `webhook_secret` setting; when it is unset bolo makes one on first use and keeps it in
`<data_dir>/webhook-secret` (`bolo pipeline secret` prints it). To check a delivery, recompute the HMAC over
`t + "." + body` and compare in constant time; reject old `t` values to stop replays. A callback that fails is tried
three times (after 1, 5 and 25 seconds); the run records each delivery under `callback`.

### History

Runs are kept in `<data_dir>/pipelines/runs/` (the newest `pipeline_runs_keep`, default 200), listed by
`GET /api/v1/pipelines/runs?pipeline=<id>`.

## API

| Method | Path | |
| --- | --- | --- |
| `GET` | `/api/v1/pipelines` | List pipelines (latest versions) |
| `POST` | `/api/v1/pipelines` | Create a pipeline (JSON or YAML body); 409 when the id exists |
| `GET` | `/api/v1/pipelines/{id}` | One pipeline (`?version=`) |
| `PUT` | `/api/v1/pipelines/{id}` | Save a new version |
| `DELETE` | `/api/v1/pipelines/{id}` | Delete a pipeline and its versions |
| `GET` | `/api/v1/pipelines/{id}/versions` | Its versions |
| `POST` | `/api/v1/pipelines/validate` | Validate a definition without saving it |
| `GET` | `/api/v1/pipelines/step-types` | Every step type with its kind, inputs, options and outputs |
| `GET` | `/api/v1/pipelines/templates` | Ready-made pipelines |
| `POST` | `/v1/pipelines/{id}/run` | Run a saved pipeline |
| `POST` | `/v1/pipelines/run` | Run a definition without saving it |
| `GET` | `/api/v1/pipelines/runs` | Run history |
| `GET` | `/api/v1/pipelines/runs/{run_id}` | One run |
| `GET` | `/api/v1/pipelines/runs/{run_id}/files/{name}` | A file a run wrote |
| `POST` | `/api/v1/pipelines/runs/{run_id}/cancel` | Stop a queued or running run |

With API keys, managing pipelines needs the `manage` scope; running one needs `stt`.

## CLI

```bash
bolo pipeline create meeting-notes.yaml        # or .json; --id to rename
bolo pipeline create --template call-qa        # from a template
bolo pipeline list
bolo pipeline show meeting-notes [--version 2] [--json]
bolo pipeline validate draft.yaml
bolo pipeline run meeting-notes call.wav [--text ...] [--var language=hi] [--json]
bolo pipeline run draft.yaml call.wav          # a file runs without saving it
bolo pipeline rm meeting-notes
bolo pipeline templates
bolo pipeline secret                           # the webhook signing secret
```

`create`, `list`, `show`, `validate` and `rm` work on the data directory, with the server running or not. `run` runs
in the CLI's own process, like `bolo transcribe`.

## Templates

| Template | Steps |
| --- | --- |
| `meeting-notes` | denoise → diarization → STT (speaker-labelled) → punctuation → LLM summary with action items |
| `call-qa` | STT → PII redaction → decisions (was the disclaimer read? the caller's sentiment, 1 to 5) → webhook |
| `podcast-chapters` | STT → punctuation → LLM chapters as JSON (`title`, `start`) → save as JSON |
| `dub` | STT → translation → TTS with a saved (cloned) voice → save |
| `voicemail-triage` | STT → decision (route: sales, support, billing, urgent, spam) → switch → webhook |
| `after-call-summary` | STT → PII redaction → LLM summary → decisions (resolved? follow-up needed?) → webhook |

`GET /api/v1/pipelines/templates` lists them; `bolo pipeline create --template <name>` saves one to edit.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `pipeline_runs_keep` | 200 | Runs kept in the history (their files too). |
| `pipeline_max_jobs` | 2 | Asynchronous runs at once; more wait in the queue. |
| `pipeline_remote_urls` | unset | Whether a run may name a `url` to fetch or a `callback_url` (unset: on a loopback bind, or when requests need an API key). |
| `webhook_secret` | unset | The secret webhooks are signed with (unset: generated into the data directory). |
| `pipeline_webhook_timeout_s` | 10 | Timeout of a webhook step or callback. |
