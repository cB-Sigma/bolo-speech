"""Phone calls: telephony providers connected to bolo's voice agents.

* :mod:`bolo.telephony.adapters`: Twilio Media Streams, Plivo audio streams, Telnyx media streaming and the
  Exotel Voicebot applet, each reduced to a thin adapter;
* :mod:`bolo.telephony.session`: the provider-neutral call session that bridges a call into a
  :class:`bolo.realtime.RealtimeSession`;
* :mod:`bolo.telephony.config`: named agents, provider credentials, the number → agent map;
* :mod:`bolo.telephony.records`: per-call timeline, latency, transcript, outcome and summary;
* :mod:`bolo.telephony.simulator`: a scripted caller behind a fake provider, for tests and benchmarks;
* :mod:`bolo.telephony.sip`: a minimal inbound SIP endpoint (G.711 RTP) for trunks and PBXs.

User documentation: ``docs/telephony.md``.
"""

from bolo.telephony.config import AgentConfig, TelephonyConfig
from bolo.telephony.manager import TelephonyManager
from bolo.telephony.records import CallRecord, CallStore

__all__ = ["AgentConfig", "CallRecord", "CallStore", "TelephonyConfig", "TelephonyManager"]
