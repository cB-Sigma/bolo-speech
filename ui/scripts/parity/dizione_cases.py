"""Write what dizione answers for a corpus of texts, so the TypeScript port the browser demo uses for tiny-voca
(src/lib/browser-tts/dizione) is held to the same tokens (src/lib/browser-tts/dizione/dizione.test.ts).

    python ui/scripts/parity/dizione_cases.py PHONEMIZER_DIR   # from the repository root, with NumPy

PHONEMIZER_DIR holds the voice's phonemiser data: the phonemizer/ folder of n4ze3m/tiny-voca at the revision the
demo pins (ui/src/lib/browser-tts/tiny-voca.ts). The phonemiser is bolo's copy of dizione (bolo/vendor/dizione).

Besides each text's token ids and stress, the fixture keeps the part of the data the corpus reaches, so the test
runs without the 5 MB of data: every dictionary entry the phonemiser looked up (a word it looked up and did not
find stays out, so it is not found in the test either), the alignment table and the homograph models, and what the
unknown-word model answered for each spelling. The model itself is checked against those answers when the test is
given the data (BOLO_DIZIONE_DATA=PHONEMIZER_DIR). Rerun after changing the corpus or dizione, and commit the JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "src"))

from bolo.vendor.dizione.engine import Phonemizer  # noqa: E402
from bolo.vendor.dizione.symbols import tokenize  # noqa: E402

OUT = ROOT / "ui" / "src" / "lib" / "browser-tts" / "dizione" / "fixtures" / "cases.json"

CORPUS = [
    # plain prose
    "Hello, world.",
    "The quick brown fox jumps over the lazy dog.",
    "I read the book yesterday, and I will read it again.",
    "They live in a live music venue.",
    "Please record the record before it ends.",
    "The wind was too strong to wind the clock.",
    "She wound the bandage around the wound.",
    "The bass player caught a bass.",
    "Lead the way to the lead mine.",
    "I object to that object.",
    "We must polish the Polish furniture.",
    "The dove dove into the bushes.",
    "He wanted to present the present to her.",
    "Does he like the does in the forest?",
    "Close the door, it's close to the road.",
    "Tear the paper without shedding a tear.",
    "Minute details matter more than a minute of time.",
    "The soldier decided to desert his post in the desert.",
    "Can you separate the separate parts?",
    "I'm sure you'll love it; we've tried it before.",
    "Don't stop believing.",
    "Isn't it wonderful?",
    "Who's there?",
    "What a beautiful day it is!",
    "Where are you going?",
    "How much does it cost?",
    "That's not what I meant.",
    "Well... I'm not sure.",
    'She said, "Come here!"',
    "He's been working (mostly) from home.",
    "This is a test — only a test.",
    "Wait—what?",
    "Really?! No way.",
    "One, two, three, four.",
    "Our neighbour's colour television was a favourite.",
    "The organisation realised its mistake.",
    "Kayleigh and Ryleigh went to the coronavirus briefing.",
    "The houseplant and the sandpiper.",
    "Unbelievably, the rewritten manuscript was overpriced.",
    "Happiness is a warm puppy.",
    "Sooo goood, cooool!",
    "Heyyy, what's uuup?",
    "Shh! Ss! Tsk.",
    "Hmm, mm, okay.",
    "k, thanks",
    "see you there, k?",
    "i, j, k are integers.",
    "lol that was gr8, see u 2morrow",
    "brb, ttyl",
    "idk tbh",
    "omg thx!!!",
    "The iPhone and YouTube on macOS.",
    "JavaScript, TypeScript and eSIM.",
    "NASA and the FBI met the CEO of IBM.",
    "The US and the UK signed it.",
    "IT IS VERY LOUD IN HERE",
    "THE QUICK BROWN FOX",
    "He works at the UN in New York.",
    "AI and ML are fields of study.",
    "World War II and Chapter IV.",
    "Henry VIII had six wives.",
    "Plan A failed, so we tried plan B.",
    "Take vitamin A daily.",
    "A dog and a cat.",
    "I think, therefore I am.",
    "An apple a day keeps the doctor away.",
    "The the the.",
    "Mr. Smith met Dr. Jones on Main St. yesterday.",
    "St. Louis is far from Elm St.",
    "Prof. Brown, e.g. the dean.",
    "i.e. the end of the U.S. tour.",
    "It's 5 p.m. Then we leave.",
    "Meet me at 7 a.m. tomorrow.",
    "approx. ten people came",
    "Est. arrival is noon.",
    "Fig. 3 shows the result.",
    "No. 5 is next, ref no. 12.",
    "max. load is high.",
    "Mrs. Robinson, Jr. and Sr. editions.",
    "vs. the champions",
    "etc. and so on",
    # numbers, money, quantities
    "It costs $3.50.",
    "That is $1,234,567.89 in total.",
    "Only 99¢ today.",
    "£20 or €15 or ¥500.",
    "It sold for $2.5 million.",
    "We raised $10k and $3M.",
    "The price is 20 €.",
    "It was 12,50 € or 0,99 €.",
    "1.299,50 EUR is the price.",
    "USD 400 is fine.",
    "I owe you -$5.",
    "Inflation was 3.5% last year.",
    "Only 3,5% voted.",
    "The score was 3-2.",
    "Pages 10-15 and 2019-2021.",
    "It weighs 5 kg and is 3 m tall.",
    "Drive 60 mph or 100 km/h.",
    "Add 1,5 kg of flour.",
    "It's 120 m² and 3 m3.",
    "The room is 3 × 4 metres.",
    "Multiply 3 x 4 now.",
    "Set it to 350°F or 180°C.",
    "The 3rd, 21st and 102nd entries.",
    "In the 1990s and the '80s.",
    "He was born in 1984.",
    "The year 2005 was good.",
    "By 1900 it was over.",
    "In 2026 we moved.",
    "There were 2026 people.",
    "The RTX 4070 and ISO 9001.",
    "Call 911 now.",
    "Call me at 555-1234.",
    "Dial (800) 555-0199 today.",
    "My number is +1 415 555 2671.",
    "Text 07700 900123 please.",
    "Enter code 4471.",
    "Your PIN is 1234.",
    "Order #A7-4419 shipped.",
    "Ticket 12345 is open.",
    "Account number 9876543210.",
    "Tracking 1Z999AA10123456784.",
    "The serial is AB-4471-Z.",
    "Use code SAVE-20 at checkout.",
    "COVID-19 and pre-2020 data.",
    "The F-16s and B-12 vitamins.",
    "A 5-year plan, 3-D glasses.",
    "A 20-year-old car.",
    "Room 101 and flight BA2490.",
    "The B12, PS5 and X7K.",
    "1 000 000 people",
    "750 000 $",
    "It took 2:03:45.",
    "The video is 48:12 long.",
    "We open at 9:30 and close at 17:00.",
    "At 12:00 midnight or 0:00.",
    "Meet at 7.30 pm.",
    "Open 9.30 to 11.30 daily, hours vary.",
    "The meeting is at 3 PM EST.",
    "From 9am-5pm.",
    "Ratio 16:9 and 3:1.",
    "The scale is 1:25,000.",
    "The aspect ratio is 4:3.",
    "On 3/14/2015 we met.",
    "Due 14/03/2026.",
    "On 12/10 we start.",
    "The card expires 08/27.",
    "Half is 1/2, a quarter 1/4.",
    "Add 2 1/2 cups.",
    "½ and 1½ and ¾.",
    "It's 24/7.",
    "It's 50/50.",
    "The ratio was 3/8.",
    "On 2026-11-15 it shipped.",
    "On 15.11.2026 it arrived.",
    "January 5, 2026",
    "5 Jan. 2026 was cold.",
    "22 February is a Sunday.",
    "Mon.-Fri. only.",
    "Jan-Mar sales.",
    "November 3-17 and 3-17 November.",
    "Python 3.12 and version 2.7.18.",
    "v1.04 and v2.10.3-1.",
    "Update to iOS 17.4 or 17.5.",
    "Upgraded from 3.11 to 3.12.",
    "1080p and 4320p video.",
    "7 of 10 people agree.",
    "The temperature is −5 degrees.",
    "x^2 + y^2 = z^2",
    "2^10 = 1024",
    "10-3=7",
    "67 > 66 and 5 < 9.",
    "see if b>a",
    "for n > 4, any q > 0",
    "x >= 0 and y <= 1 and z != 2",
    "a=10, b = 20",
    "k = k + 1",
    "f(a, b) and (a)",
    "a) the first point",
    "Use <b>bold</b> text.",
    "Vec<u64> and map<int, int>.",
    "File>Open the menu.",
    "p<.05",
    "≥ 5 and ≤ 10 and ≠ 3",
    "about ~5 minutes",
    "#hashtag and #1 fan.",
    "Tom & Jerry @ home.",
    "50% + 20% = 70%",
    "3.14159 is pi.",
    "Pi is π.",
    "0.5 and .75 and 007.",
    "1,000,000 and 1,234.",
    "123456789012345678901234",
    "Item 42 of 1000.",
    "It's 1000 miles.",
    "2,500 feet up.",
    "3 people, 12 children.",
    "-40 degrees",
    # addresses, files, symbols, emoji
    "Visit https://www.example.com/path?x=1 today.",
    "Email john.doe@example.co.uk now.",
    "Go to www.bolo.dev.",
    "Open report_final.pdf and data.csv.",
    "Edit /etc/hosts or ~/notes.txt.",
    "The end.Then it started.",
    "I love it 😀👍!",
    "Nice :) see you ;-) <3",
    "you<3",
    "Great ¯\\_(ツ)_/¯",
    "→ next step",
    "w/o sugar, w/ milk, n/a",
    "rock 'n' roll",
    "'Tis the season, 'em all.",
    "'cause I said so",
    "the dogs' bones and the dog's bone",
    "James's car",
    "café, naïve, résumé, rosé, jalapeño",
    "Ærøskøbing and Łódź and straße",
    "fiancée and déjà vu",
    "{HH AH0 L OW1} there",
    "say tomato{T AH0 M EY1 T OW2} please",
    "{h ə l ˈoʊ} again",
    "x_{i} and {n}",
    "Tung tung tung sahur!",
    "Hello? Hello!! Hello...",
    "Yes; no: maybe.",
    "(Parenthetical) remark.",
    "A \"quoted\" word and 'single' ones.",
    "Sizes XS-XL and S-M.",
    "RTX 4070 Ti",
    "Ref. No. 5 applies.",
    # unknown words and names
    "Okafor met Nguyen and Siobhan in Llanfair.",
    "The flibbertigibbet zorbled the quixotic snark.",
    "Supercalifragilisticexpialidocious!",
    "Blorptastic gizmos and wugs.",
    "Xiaoling and Saoirse joined Aleksandr.",
    "The Brexiteers and the anti-vaxxers.",
    "Covfefe!",
    "Grammarly and Zoombombing.",
    "A cryptocurrency influencer.",
    "Tensorflow and pytorch are libraries.",
    "Kubernetes clusters.",
    "Hyperparameters and embeddings.",
    "Mx. Taylor uses they/them.",
    "The Lakers beat the Celtics.",
    "Antidisestablishmentarianism.",
    "Pneumonoultramicroscopic.",
    "unhappiness, overreaction, misunderstanding",
    "rethink, preorder, nonprofit, semifinal, underwater, outrun",
    "stormier, happily, carrying, starring, waited, hoped, hopped",
    "knives, wolves, leaves, ladies, boxes, churches",
    "friendship, childhood, warlike, kingdom, foolish, readable",
    "toward, homewards, breakage, optimist, realism",
]

TEMPLATES = [
    ("It costs ${}.", ["0.01", "1", "1.00", "12.5", "99.99", "1,000", "250000"]),
    ("We need {} items.", ["0", "7", "13", "40", "101", "999", "1001", "12345", "1000000"]),
    ("At {} we leave.", ["6:05", "11:45 am", "1:00 PM", "23:59", "12:30"]),
    ("Born on {}.", ["July 4, 1776", "1 April 2001", "2000-01-01", "31/12/1999", "Dec 25"]),
    ("I {} it.", ["read", "live", "lead", "wind", "record", "present", "object", "close", "use", "permit"]),
    ("The {} is here.", ["read", "live", "lead", "wind", "record", "present", "object", "use", "permit", "bow"]),
    ("Please {} now.", ["refuse", "contest", "conduct", "produce", "project", "subject", "suspect", "increase"]),
]


class Recorder(dict):
    """The dictionary, noting every spelling the phonemiser asks for."""

    def __init__(self, data):
        super().__init__(data)
        self.asked = set()

    def get(self, key, default=None):
        self.asked.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.asked.add(key)
        return super().__contains__(key)

    def __getitem__(self, key):
        self.asked.add(key)
        return super().__getitem__(key)


def main(directory: Path) -> None:
    texts = list(CORPUS)
    for template, values in TEMPLATES:
        texts += [template.format(value) for value in values]
    phonemizer = Phonemizer(directory)
    phonemizer.lexicon = Recorder(phonemizer.lexicon)
    predictions = {}
    predict = phonemizer.model.predict

    def recorded(word):
        out = predict(word)
        predictions[word] = list(out)
        return out

    phonemizer.model.predict = recorded
    cases = []
    for text in texts:
        try:
            out = tokenize(phonemizer.pieces(text))
            cases.append({"text": text, "ids": list(out.ids), "stress": list(out.stress)})
        except Exception as exc:  # the port must fail on the same texts
            cases.append({"text": text, "error": type(exc).__name__})
    lexicon = {
        word: phonemizer.lexicon[word]
        for word in sorted(phonemizer.lexicon.asked)
        if dict.__contains__(phonemizer.lexicon, word)
    }
    fixture = {
        "source": "bolo/vendor/dizione with the phonemizer/ data of n4ze3m/tiny-voca; ui/scripts/parity/dizione_cases.py",
        "cases": cases,
        "lexicon": lexicon,
        "align": json.loads((directory / "align.json").read_text(encoding="utf-8")),
        "homographs": json.loads((directory / "homographs.json").read_text(encoding="utf-8")),
        "g2p": json.loads((directory / "g2p.json").read_text(encoding="utf-8")),
        "predictions": predictions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(cases)} texts, {len(lexicon)} dictionary entries, {len(predictions)} model answers -> {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
