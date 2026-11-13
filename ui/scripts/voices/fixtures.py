"""Token-id fixtures for the browser front end (ui/src/lib/browser-tts/fixtures/sherpa-tokens.json).

For each case the ids are made the way the browser makes them (characters looked up in tokens.txt, a blank
between every pair and at both ends), run through the voice with the noise switched off, passed through
sherpa-onnx's pause shortening, and compared with what sherpa-onnx itself produces for the same text. Only
cases that agree (same length, largest difference below 1e-3) are written.

    python ui/scripts/voices/fixtures.py ARCHIVE_DIR OUT.json     # ARCHIVE_DIR holds <voice id>.tar.gz

Needs numpy, onnxruntime and sherpa-onnx (the bolo server's environment has all three).
"""
import json, shutil, sys, tarfile, tempfile, unicodedata
import numpy as np, onnxruntime as ort, sherpa_onnx as so

CASES = {
    'syspin-hi-female': ['नमस्ते, आप कैसे हैं?', 'आज मौसम बहुत अच्छा है।'],
    'syspin-mai-female': ['गामक चरवाहा जाड़ में बहुत धन प्राप्त कयलक।', 'प्रणाम! अहाँ केहन छी?'],
    'syspin-bho-female': ['पानी बिल हर महीना भरे के पड़ी.'],
    'syspin-mr-female': ['नमस्कार, तुम्ही कसे आहात?'],
    'syspin-kn-female': ['ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?'],
    'syspin-en-female': ['hello, how are you today?'],
}
def scale_silence(x, sr, scale=0.2):
    # sherpa-onnx GeneratedAudio::ScaleSilence: pauses (|x| <= 0.01) longer than 0.2 s keep `scale` of their length.
    thr = int(sr * 0.2); iv = []; last = -1; n = len(x)
    for i in range(n):
        if abs(x[i]) <= 0.01:
            if last == -1: last = i
            continue
        if last != -1 and i - last < thr:
            last = -1; continue
        if last != -1:
            iv.append((last, i)); last = -1
    if last != -1 and n - last > thr: iv.append((last, n))
    out = []; i = 0
    for s0, e0 in iv:
        out.append(x[i:s0]); out.append(x[s0:s0 + int((e0 - s0) * scale)]); i = e0
    out.append(x[i:])
    return np.concatenate(out)

archives, target = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp(prefix='bolo-fixtures-')
out = []
for vid, texts in CASES.items():
    d = f'{work}/{vid}'
    with tarfile.open(f'{archives}/{vid}.tar.gz') as tar:
        tar.extractall(work, filter='data')
    tok = {}
    for line in open(d + '/tokens.txt', encoding='utf-8'):
        line = line.rstrip('\n')
        if not line:
            continue
        s, _, i = line.rpartition(' ')
        tok.setdefault(s if s else ' ', int(i))
    sess = ort.InferenceSession(d + '/model.onnx')
    blank = int(sess.get_modelmeta().custom_metadata_map['blank_id'])
    tts = so.OfflineTts(so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
        vits=so.OfflineTtsVitsModelConfig(model=d + '/model.onnx', tokens=d + '/tokens.txt', noise_scale=0.0, noise_scale_w=0.0),
        num_threads=2)))
    for text in texts:
        assert unicodedata.normalize('NFC', text).lower() == text
        ids = [tok[c] for c in text if c in tok]
        seq = [blank] * (2 * len(ids) + 1)
        seq[1::2] = ids
        y = sess.run(None, {'input': np.array([seq]), 'input_lengths': np.array([len(seq)]),
                            'scales': np.array([0, 1, 0], dtype=np.float32)})[0].reshape(-1)
        a = np.asarray(tts.generate(text).samples, dtype=np.float32)
        y = scale_silence(y, 22050)
        same = len(a) == len(y) and float(np.abs(a - y).max()) < 1e-3
        print(vid, text, len(y), len(a), same, flush=True)
        if same:
            used = {c: tok[c] for c in sorted(set(text)) if c in tok}
            out.append({'voice': vid, 'text': text, 'symbols': used, 'blank_id': blank, 'ids': seq, 'samples': len(y)})
json.dump({
    'about': 'Made by ui/scripts/voices/fixtures.py: for each case these ids, run through the voice with the noise off, '
             'give the same waveform as sherpa-onnx 2.0.1 for the same text (same length, max difference below 1e-3).',
    'cases': out}, open(target, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
shutil.rmtree(work)
