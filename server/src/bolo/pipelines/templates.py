"""Ready-made pipelines (``GET /api/v1/pipelines/templates``, ``bolo pipeline create --template``)."""

from __future__ import annotations

import copy
from typing import Any

from bolo.errors import InvalidRequest

TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "meeting-notes",
        "name": "Meeting notes",
        "description": "Clean the recording, find who spoke when, transcribe it with speaker labels, restore "
        "punctuation and write notes with action items.",
        "steps": [
            {"id": "clean", "type": "denoise"},
            {"id": "who", "type": "diarization", "inputs": {"audio": "{{steps.clean.audio}}"}},
            {"id": "stt", "type": "stt", "inputs": {"audio": "{{steps.clean.audio}}", "turns": "{{steps.who.turns}}"}},
            {"id": "punct", "type": "punctuation", "inputs": {"text": "{{steps.stt.transcript}}"}},
            {"id": "summary", "type": "llm", "options": {
                "system": "You write concise meeting notes.",
                "prompt": "Write meeting notes for this transcript: a short summary, the decisions taken and the "
                          "action items with their owners.\n\n{{steps.punct.transcript}}",
            }},
        ],
        "outputs": {"notes": "{{steps.summary.text}}", "transcript": "{{steps.punct.transcript}}",
                    "speakers": "{{steps.who.speakers}}"},
    },
    {
        "id": "call-qa",
        "name": "Call QA",
        "description": "Transcribe a support call, mask personal information, check that the disclaimer was read "
        "and score the caller's sentiment, then post the result to a webhook.",
        "vars": {"webhook_url": "https://example.com/hooks/call-qa",
                 "disclaimer": "this call may be recorded for quality and training purposes"},
        "steps": [
            {"id": "stt", "type": "stt"},
            {"id": "redact", "type": "pii"},
            {"id": "disclaimer", "type": "decision", "options": {
                "question": "Did the agent tell the caller: \"{{vars.disclaimer}}\" (or words to that effect)?",
            }},
            {"id": "sentiment", "type": "decision", "inputs": {"text": "{{steps.redact.text}}"}, "options": {
                "question": "How satisfied does the caller sound by the end of the call, from 1 (angry) to 5 "
                            "(delighted)?",
                "choices": ["1", "2", "3", "4", "5"],
            }},
            {"id": "report", "type": "webhook", "options": {"url": "{{vars.webhook_url}}"}, "inputs": {"body": {
                "transcript": "{{steps.redact.text}}",
                "disclaimer_read": "{{steps.disclaimer.answer}}",
                "sentiment": "{{steps.sentiment.answer}}",
                "sentiment_reason": "{{steps.sentiment.reason}}",
            }}},
        ],
        "outputs": {"disclaimer_read": "{{steps.disclaimer.answer}}", "sentiment": "{{steps.sentiment.answer}}",
                    "transcript": "{{steps.redact.text}}"},
    },
    {
        "id": "podcast-chapters",
        "name": "Podcast chapters",
        "description": "Transcribe an episode with timestamps and let the language model divide it into titled "
        "chapters; saved as JSON and subtitles.",
        "steps": [
            {"id": "stt", "type": "stt"},
            {"id": "punct", "type": "punctuation", "inputs": {"text": "{{steps.stt.transcript}}"}},
            {"id": "lines", "type": "template", "options": {"template": "{{steps.punct.transcript.segments}}"}},
            {"id": "chapters", "type": "llm", "options": {
                "json": True,
                "prompt": "Here are the segments of a podcast episode as JSON (start and end in seconds). Divide "
                          "the episode into 3 to 10 chapters. Answer {\"chapters\": [{\"start\": <seconds>, "
                          "\"title\": \"...\", \"summary\": \"...\"}]}.\n\n{{steps.lines.text}}",
            }},
            {"id": "save_chapters", "type": "save", "inputs": {"value": "{{steps.chapters.json}}"},
             "options": {"name": "chapters.json"}},
            {"id": "save_subtitles", "type": "save", "inputs": {"value": "{{steps.punct.transcript}}"},
             "options": {"name": "episode.srt"}},
        ],
        "outputs": {"chapters": "{{steps.chapters.json.chapters}}", "transcript": "{{steps.punct.text}}"},
    },
    {
        "id": "dub",
        "name": "Dub or translate",
        "description": "Transcribe, translate the text with the language model and speak it in a saved (cloned) "
        "voice.",
        "vars": {"target_language": "es", "voice": "my-voice"},
        "steps": [
            {"id": "stt", "type": "stt"},
            {"id": "translate", "type": "translation", "options": {"target_language": "{{vars.target_language}}"}},
            {"id": "speak", "type": "tts", "options": {"voice": "{{vars.voice}}",
                                                       "language": "{{vars.target_language}}"}},
            {"id": "save", "type": "save", "inputs": {"value": "{{steps.speak.audio}}"},
             "options": {"name": "dubbed.wav"}},
        ],
        "outputs": {"text": "{{steps.translate.text}}", "audio": "{{steps.speak.audio}}"},
    },
    {
        "id": "voicemail-triage",
        "name": "Voicemail triage",
        "description": "Transcribe a voicemail, decide where it goes and notify that team; urgent messages page "
        "the on-call person.",
        "vars": {"webhook_url": "https://example.com/hooks/voicemail", "urgent_url": "https://example.com/hooks/page"},
        "steps": [
            {"id": "stt", "type": "stt"},
            {"id": "route", "type": "decision", "options": {
                "question": "Which team should handle this voicemail?",
                "choices": ["sales", "support", "billing", "urgent", "spam"],
            }},
            {"id": "dispatch", "type": "switch", "inputs": {"value": "{{steps.route.answer}}"}, "options": {
                "cases": {
                    "spam": [{"id": "drop", "type": "stop", "options": {"reason": "spam"}}],
                    "urgent": [{"id": "page", "type": "webhook", "options": {"url": "{{vars.urgent_url}}"},
                                "inputs": {"body": {"text": "{{steps.stt.text}}", "reason": "{{steps.route.reason}}"}}}],
                },
                "default": [{"id": "notify", "type": "webhook", "options": {"url": "{{vars.webhook_url}}"},
                             "inputs": {"body": {"team": "{{steps.route.answer}}", "text": "{{steps.stt.text}}"}}}],
            }},
        ],
        "outputs": {"team": "{{steps.route.answer}}", "text": "{{steps.stt.text}}"},
    },
    {
        "id": "after-call-summary",
        "name": "After-call summary and QA",
        "description": "At the end of every phone call, on its transcript: redact it, summarise it, check whether "
        "the issue was resolved and whether a follow-up is needed, and post it all to a webhook.",
        "vars": {"webhook_url": "https://example.com/hooks/after-call"},
        "triggers": [{"type": "call_end"}],
        "steps": [
            {"id": "redact", "type": "pii", "inputs": {"text": "{{input.text}}"}},
            {"id": "summary", "type": "llm", "options": {
                "prompt": "Summarise this support call in three sentences:\n\n{{steps.redact.text}}"}},
            {"id": "resolved", "type": "decision", "inputs": {"text": "{{steps.redact.text}}"},
             "options": {"question": "Was the caller's issue resolved during the call?"}},
            {"id": "follow_up", "type": "decision", "inputs": {"text": "{{steps.redact.text}}"},
             "options": {"question": "Does someone need to follow up with the caller?"}},
            {"id": "post", "type": "webhook", "options": {"url": "{{vars.webhook_url}}"}, "inputs": {"body": {
                "summary": "{{steps.summary.text}}", "resolved": "{{steps.resolved.answer}}",
                "follow_up": "{{steps.follow_up.answer}}", "transcript": "{{steps.redact.text}}",
                "call": "{{input.call}}",
            }}},
        ],
        "outputs": {"summary": "{{steps.summary.text}}", "resolved": "{{steps.resolved.answer}}",
                    "follow_up": "{{steps.follow_up.answer}}"},
    },
    {
        "id": "support-line",
        "name": "Support line",
        "mode": "live",
        "description": "Per caller turn: an angry or upset caller is passed to a person at once; otherwise the "
        "question is routed to a department and the language model answers as that desk (streamed, with the agent's "
        "tools).",
        "steps": [
            {"id": "mood", "type": "decision", "options": {
                "question": "Is the caller angry, very upset or asking for a human?"}},
            {"id": "escalate", "type": "condition", "options": {
                "if": {"value": "{{steps.mood.answer}}", "equals": "yes"},
                "then": [{"id": "to_person", "type": "transfer", "options": {
                    "say": "I am sorry about this. Let me put you through to a colleague right away."}}],
            }},
            {"id": "department", "type": "decision", "options": {
                "question": "Which department should answer the caller?",
                "choices": ["billing", "technical", "sales", "other"]}},
            {"id": "answer", "type": "respond", "options": {
                "instructions": "You are answering for the {{steps.department.answer}} desk. Answer briefly; if the "
                                "question belongs to another desk, say which one."}},
        ],
    },
    {
        "id": "appointment-line",
        "name": "Appointment booking line",
        "mode": "live",
        "description": "Per caller turn: what the caller wants (book, change, cancel); for a booking the free "
        "slots come from your scheduling webhook and the language model offers them and confirms one.",
        "vars": {"slots_url": "https://example.com/hooks/slots"},
        "steps": [
            {"id": "intent", "type": "decision", "options": {
                "question": "What does the caller want?", "choices": ["book", "change", "cancel", "other"]}},
            {"id": "route", "type": "switch", "inputs": {"value": "{{steps.intent.answer}}"}, "options": {
                "cases": {
                    "book": [
                        {"id": "slots", "type": "webhook", "on_error": "continue",
                         "options": {"url": "{{vars.slots_url}}"},
                         "inputs": {"body": {"request": "{{input.text}}", "turn": "{{input.turn}}"}}},
                        {"id": "offer", "type": "respond", "options": {
                            "instructions": "Free slots (JSON from the calendar; empty if it did not answer): "
                                            "{{steps.slots.body}}. Offer two or three and confirm the one the caller "
                                            "picks. Never invent a slot."}},
                    ],
                    "other": [{"id": "explain", "type": "reply", "inputs": {
                        "text": "I can book, move or cancel appointments. What would you like to do?"}}],
                },
                "default": [{"id": "manage", "type": "respond", "options": {
                    "instructions": "The caller wants to {{steps.intent.answer}} an appointment. Ask for the date "
                                    "and name it was booked under, then confirm the change."}}],
            }},
        ],
    },
]  # fmt: skip


def all_templates() -> list[dict[str, Any]]:
    return copy.deepcopy(TEMPLATES)


def template(name: str) -> dict[str, Any]:
    for t in TEMPLATES:
        if t["id"] == name:
            return copy.deepcopy(t)
    raise InvalidRequest(f"no template '{name}' (templates: {', '.join(t['id'] for t in TEMPLATES)})")
