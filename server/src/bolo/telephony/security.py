"""Webhook signatures of the telephony providers, and the Ed25519 arithmetic Telnyx needs.

Each provider signs what it sends in its own way; bolo checks every one of them before it answers a call:

* **Twilio**: ``X-Twilio-Signature`` = Base64(HMAC-SHA1(auth token, URL + every POST field name and value,
  sorted by name)). The URL is the public one, query string included.
* **Plivo**: ``X-Plivo-Signature-V3`` (comma-separated when the account has several tokens) =
  Base64(HMAC-SHA256(auth token, URL + "." + sorted POST fields + "." + nonce)), nonce from
  ``X-Plivo-Signature-V3-Nonce``. ``X-Plivo-Signature-Ma-V3`` is the same with the main account's token.
* **Telnyx**: ``telnyx-signature-ed25519`` = Base64(Ed25519 signature of ``"<telnyx-timestamp>|<raw body>"``)
  with the account's public key; for the media WebSocket the signed text is ``"<timestamp>|<stream_url>"``.
* **Exotel** signs nothing: its stream applet sends HTTP Basic credentials placed in the URL.

The ``sign_*`` functions are the same computations the other way round. The call simulator and the tests use them
to play the provider. Ed25519 is implemented here from RFC 8032 (verification of one signature per webhook
needs no speed), so Telnyx works without a cryptography package.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from collections.abc import Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit

#: Telnyx recommends refusing webhooks whose timestamp is more than five minutes from the server's clock
TELNYX_TOLERANCE_S = 300.0


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


# -- Twilio -----------------------------------------------------------------------------------------------
def sign_twilio(auth_token: str, url: str, params: Mapping[str, str]) -> str:
    text = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    return base64.b64encode(hmac.new(auth_token.encode(), text.encode(), hashlib.sha1).digest()).decode()


def verify_twilio(auth_token: str, url: str, params: Mapping[str, str], signature: str | None) -> bool:
    return bool(signature) and _eq(sign_twilio(auth_token, url, params), str(signature))


# -- Plivo ------------------------------------------------------------------------------------------------
def _plivo_base(url: str, params: Mapping[str, str], method: str) -> str:
    """The string Plivo signs (as its SDKs build it): the URL with its query sorted; for POST a "." and the
    sorted form fields as name+value with nothing between them."""
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    query = sorted(parse_qsl(parts.query, keep_blank_values=True))
    if method.upper() == "GET":
        query = sorted([*query, *params.items()])
        return base + ("?" + "&".join(f"{k}={v}" for k, v in query) if query else "")
    if query:
        base += "?" + "&".join(f"{k}={v}" for k, v in query)
    if params:
        base += "." + "".join(f"{k}{params[k]}" for k in sorted(params))
    return base


def sign_plivo(auth_token: str, url: str, params: Mapping[str, str], nonce: str, method: str = "POST") -> str:
    text = _plivo_base(url, params, method) + "." + nonce
    return base64.b64encode(hmac.new(auth_token.encode(), text.encode(), hashlib.sha256).digest()).decode()


def verify_plivo(
    auth_token: str, url: str, params: Mapping[str, str], nonce: str | None, signatures: Iterable[str | None],
    method: str = "POST",
) -> bool:  # fmt: skip
    if not nonce:
        return False
    expected = sign_plivo(auth_token, url, params, nonce, method)
    presented = [s.strip() for header in signatures if header for s in header.split(",")]
    return any(_eq(expected, s) for s in presented if s)


# -- Ed25519 (RFC 8032, section 6) --------------------------------------------------------------------------
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)
_Point = tuple[int, int, int, int]


def _add(a: _Point, b: _Point) -> _Point:
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    e = (y1 - x1) * (y2 - x2) % _P
    f = (y1 + x1) * (y2 + x2) % _P
    g = 2 * t1 * t2 * _D % _P
    h = 2 * z1 * z2 % _P
    return ((f - e) * (h - g) % _P, (h + g) * (f + e) % _P, (h - g) * (h + g) % _P, (f - e) * (f + e) % _P)


def _mul(k: int, point: _Point) -> _Point:
    out: _Point = (0, 1, 1, 0)
    while k > 0:
        if k & 1:
            out = _add(out, point)
        point = _add(point, point)
        k >>= 1
    return out


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        return None
    return _P - x if (x & 1) != sign else x


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0) or 0
_G: _Point = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(point: _Point) -> bytes:
    x, y, z, _ = point
    zinv = pow(z, _P - 2, _P)
    x, y = x * zinv % _P, y * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(data: bytes) -> _Point | None:
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _hash_int(*parts: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"".join(parts)).digest(), "little")


def _expand(secret: bytes) -> tuple[int, bytes]:
    digest = hashlib.sha512(secret).digest()
    a = int.from_bytes(digest[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, digest[32:]


def ed25519_public_key(secret: bytes) -> bytes:
    """The public key of a 32-byte Ed25519 seed."""
    return _compress(_mul(_expand(secret)[0], _G))


def ed25519_sign(secret: bytes, message: bytes) -> bytes:
    a, prefix = _expand(secret)
    public = _compress(_mul(a, _G))
    r = _hash_int(prefix, message) % _L
    big_r = _compress(_mul(r, _G))
    s = (r + _hash_int(big_r, public, message) % _L * a) % _L
    return big_r + int.to_bytes(s, 32, "little")


def ed25519_verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    a = _decompress(public)
    r = _decompress(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if a is None or r is None or s >= _L:
        return False
    h = _hash_int(signature[:32], public, message) % _L
    left = _mul(s, _G)
    right = _add(r, _mul(h, a))
    # equal as projective points: x1*z2 == x2*z1 and y1*z2 == y2*z1
    return (left[0] * right[2] - right[0] * left[2]) % _P == 0 and (left[1] * right[2] - right[1] * left[2]) % _P == 0


# -- Telnyx -----------------------------------------------------------------------------------------------
def telnyx_public_key(value: str) -> bytes:
    """The 32-byte key from what the Mission Control Portal shows (Base64); a hex key is accepted too."""
    text = value.strip()
    try:
        raw = bytes.fromhex(text) if len(text) == 64 else base64.b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, binascii.Error) as exc:
        raise ValueError("the Telnyx public key must be the Base64 text from the Mission Control Portal") from exc
    if len(raw) != 32:
        raise ValueError("the Telnyx public key must decode to 32 bytes (an Ed25519 public key)")
    return raw


def sign_telnyx(secret: bytes, timestamp: str, payload: bytes) -> str:
    return base64.b64encode(ed25519_sign(secret, timestamp.encode() + b"|" + payload)).decode()


def verify_telnyx(
    public_key: str, timestamp: str | None, payload: bytes, signature: str | None, *, now: float | None = None
) -> bool:
    if not timestamp or not signature:
        return False
    try:
        age = abs((time.time() if now is None else now) - float(timestamp))
        raw = base64.b64decode(signature, validate=True)
        key = telnyx_public_key(public_key)
    except (ValueError, binascii.Error):
        return False
    if age > TELNYX_TOLERANCE_S:
        return False
    return ed25519_verify(key, timestamp.encode() + b"|" + payload, raw)


# -- HTTP Basic (Exotel) ------------------------------------------------------------------------------------
def basic_header(username: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()


def verify_basic(header: str | None, username: str, password: str) -> bool:
    return bool(header) and _eq(str(header).strip(), basic_header(username, password))
