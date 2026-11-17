"""Language codes as language-identification models write them, made comparable with the catalog's.

Models answer in ISO 639-1 (``hi``), ISO 639-3 (``hin``), or legacy codes (``iw`` for Hebrew, ``jw`` for Javanese
in Whisper and VoxLingua107). :func:`normalize` turns each into the code the catalog uses for the language: the
two-letter code where there is one, else the three-letter one (``bho``, ``mai``, ``sat``).
"""

from __future__ import annotations

#: ISO 639-3 (and a few 639-2/B and macrolanguage members) -> (ISO 639-1, English name)
ISO639_3: dict[str, tuple[str, str]] = {
    "abk": ("ab", "Abkhazian"), "afr": ("af", "Afrikaans"), "amh": ("am", "Amharic"), "ara": ("ar", "Arabic"),
    "arb": ("ar", "Arabic"), "asm": ("as", "Assamese"), "aze": ("az", "Azerbaijani"), "azj": ("az", "Azerbaijani"),
    "bak": ("ba", "Bashkir"), "bel": ("be", "Belarusian"), "ben": ("bn", "Bengali"), "bod": ("bo", "Tibetan"),
    "bos": ("bs", "Bosnian"), "bre": ("br", "Breton"), "bul": ("bg", "Bulgarian"), "cat": ("ca", "Catalan"),
    "ces": ("cs", "Czech"), "cmn": ("zh", "Chinese (Mandarin)"), "zho": ("zh", "Chinese"), "cym": ("cy", "Welsh"),
    "dan": ("da", "Danish"), "deu": ("de", "German"), "ell": ("el", "Greek"), "eng": ("en", "English"),
    "epo": ("eo", "Esperanto"), "est": ("et", "Estonian"), "ekk": ("et", "Estonian"), "eus": ("eu", "Basque"),
    "fao": ("fo", "Faroese"), "fas": ("fa", "Persian"), "pes": ("fa", "Persian"), "fin": ("fi", "Finnish"),
    "fra": ("fr", "French"), "ful": ("ff", "Fulah"), "gle": ("ga", "Irish"), "glg": ("gl", "Galician"),
    "glv": ("gv", "Manx"), "grn": ("gn", "Guarani"), "guj": ("gu", "Gujarati"), "hat": ("ht", "Haitian Creole"),
    "hau": ("ha", "Hausa"), "heb": ("he", "Hebrew"), "hin": ("hi", "Hindi"), "hrv": ("hr", "Croatian"),
    "hun": ("hu", "Hungarian"), "hye": ("hy", "Armenian"), "ibo": ("ig", "Igbo"), "ina": ("ia", "Interlingua"),
    "ind": ("id", "Indonesian"), "isl": ("is", "Icelandic"), "ita": ("it", "Italian"), "jav": ("jv", "Javanese"),
    "jpn": ("ja", "Japanese"), "kan": ("kn", "Kannada"), "kas": ("ks", "Kashmiri"), "kat": ("ka", "Georgian"),
    "kaz": ("kk", "Kazakh"), "khm": ("km", "Khmer"), "kir": ("ky", "Kyrgyz"), "kor": ("ko", "Korean"),
    "lao": ("lo", "Lao"), "lat": ("la", "Latin"), "lav": ("lv", "Latvian"), "lvs": ("lv", "Latvian"),
    "lin": ("ln", "Lingala"), "lit": ("lt", "Lithuanian"), "ltz": ("lb", "Luxembourgish"), "lug": ("lg", "Ganda"),
    "mal": ("ml", "Malayalam"), "mar": ("mr", "Marathi"), "mkd": ("mk", "Macedonian"), "mlg": ("mg", "Malagasy"),
    "plt": ("mg", "Malagasy"), "mlt": ("mt", "Maltese"), "mon": ("mn", "Mongolian"), "khk": ("mn", "Mongolian"),
    "mri": ("mi", "Maori"), "msa": ("ms", "Malay"), "zlm": ("ms", "Malay"), "zsm": ("ms", "Malay"),
    "mya": ("my", "Burmese"), "nep": ("ne", "Nepali"), "npi": ("ne", "Nepali"), "nld": ("nl", "Dutch"),
    "nno": ("nn", "Norwegian Nynorsk"), "nob": ("no", "Norwegian"), "nor": ("no", "Norwegian"),
    "nya": ("ny", "Chichewa"), "oci": ("oc", "Occitan"), "orm": ("om", "Oromo"), "gaz": ("om", "Oromo"),
    "ori": ("or", "Odia"), "ory": ("or", "Odia"), "pan": ("pa", "Punjabi"), "pol": ("pl", "Polish"),
    "por": ("pt", "Portuguese"), "pus": ("ps", "Pashto"), "pbt": ("ps", "Pashto"), "ron": ("ro", "Romanian"),
    "rus": ("ru", "Russian"), "san": ("sa", "Sanskrit"), "sin": ("si", "Sinhala"), "slk": ("sk", "Slovak"),
    "slv": ("sl", "Slovenian"), "sna": ("sn", "Shona"), "snd": ("sd", "Sindhi"), "som": ("so", "Somali"),
    "spa": ("es", "Spanish"), "sqi": ("sq", "Albanian"), "als": ("sq", "Albanian"), "srp": ("sr", "Serbian"),
    "sun": ("su", "Sundanese"), "swe": ("sv", "Swedish"), "swa": ("sw", "Swahili"), "swh": ("sw", "Swahili"),
    "tam": ("ta", "Tamil"), "tat": ("tt", "Tatar"), "tel": ("te", "Telugu"), "tgk": ("tg", "Tajik"),
    "tgl": ("tl", "Tagalog"), "fil": ("tl", "Filipino"), "tha": ("th", "Thai"), "tuk": ("tk", "Turkmen"),
    "tur": ("tr", "Turkish"), "ukr": ("uk", "Ukrainian"), "urd": ("ur", "Urdu"), "uzb": ("uz", "Uzbek"),
    "uzn": ("uz", "Uzbek"), "vie": ("vi", "Vietnamese"), "wol": ("wo", "Wolof"), "xho": ("xh", "Xhosa"),
    "yid": ("yi", "Yiddish"), "ydd": ("yi", "Yiddish"), "yor": ("yo", "Yoruba"), "zul": ("zu", "Zulu"),
    "kin": ("rw", "Kinyarwanda"), "tir": ("ti", "Tigrinya"), "kur": ("ku", "Kurdish"),
    "kmr": ("ku", "Kurdish (Kurmanji)"), "sco": ("sco", "Scots"),
}  # fmt: skip

#: three-letter codes the catalog uses as they are (no two-letter code), with names
OTHER_NAMES: dict[str, str] = {
    "ast": "Asturian", "ceb": "Cebuano", "ckb": "Central Kurdish", "haw": "Hawaiian", "kam": "Kamba",
    "kea": "Kabuverdianu", "luo": "Luo", "nso": "Northern Sotho", "umb": "Umbundu", "war": "Waray", "yue": "Cantonese",
    "sco": "Scots", "bho": "Bhojpuri", "mai": "Maithili", "mag": "Magahi", "hne": "Chhattisgarhi", "brx": "Bodo",
    "doi": "Dogri", "kok": "Konkani", "gom": "Konkani (Goan)", "mni": "Manipuri", "sat": "Santali", "raj": "Rajasthani",
    "awa": "Awadhi", "bgc": "Haryanvi", "tcy": "Tulu", "kfy": "Kumaoni", "gbm": "Garhwali", "bns": "Bundeli",
    "lus": "Mizo", "kha": "Khasi", "bpy": "Bishnupriya", "sck": "Sadri", "unr": "Mundari", "hoc": "Ho",
    "khn": "Khandeshi", "wbr": "Wagdi", "mup": "Malvi", "bhb": "Bhili",
}  # fmt: skip

#: two-letter codes and names, for codes that come without a name
NAMES_1: dict[str, str] = dict(ISO639_3.values())
NAMES_1.update({"zh": "Chinese", "fa": "Persian", "jv": "Javanese", "he": "Hebrew", "ne": "Nepali", "ms": "Malay"})

#: legacy and alternative codes -> the catalog's
LEGACY = {"iw": "he", "jw": "jv", "in": "id", "ji": "yi", "mo": "ro", "nb": "no", "zh-cn": "zh", "fil": "tl"}

#: codes that name the same language for routing (the catalog keeps Whisper's ``jw`` for Javanese)
EQUIVALENT: dict[str, frozenset[str]] = {
    "jv": frozenset({"jv", "jw"}),
    "he": frozenset({"he", "iw"}),
    "no": frozenset({"no", "nb"}),
    "kok": frozenset({"kok", "gom"}),
}


def normalize(code: str) -> str:
    """The catalog's code for a language code written by a model: ``hin`` -> ``hi``, ``iw`` -> ``he``,
    ``bho`` stays ``bho``; regions and scripts are dropped (``hi-IN`` -> ``hi``)."""
    c = (code or "").strip().lower().replace("_", "-")
    if c in LEGACY:
        return LEGACY[c]
    c = c.split("-")[0]
    if c in LEGACY:
        return LEGACY[c]
    if len(c) == 3 and c in ISO639_3:
        return ISO639_3[c][0]
    return c


def name(code: str) -> str:
    """An English name for a (normalised) language code, or the code itself."""
    c = normalize(code)
    return NAMES_1.get(c) or OTHER_NAMES.get(c) or c


def same_language(a: str, b: str) -> bool:
    """Whether two codes name the same language (``hi`` and ``hi-IN``, ``jv`` and ``jw``, ``he`` and ``iw``)."""
    x, y = normalize(a), normalize(b)
    if x == y:
        return True
    return y in EQUIVALENT.get(x, frozenset()) or x in EQUIVALENT.get(y, frozenset())
