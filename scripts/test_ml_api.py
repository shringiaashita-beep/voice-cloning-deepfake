import sys, os, urllib.request, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_presets_live import generate_preset_wav

for p in ['elevenlabs', 'rvc', 'bark', 'human']:
    wav_bytes = generate_preset_wav(p)
    boundary = '----WebKitFormBoundaryVoxGuardBench'
    body = (
        b'--' + boundary.encode() +
        b'\r\nContent-Disposition: form-data; name="file"; filename="' + p.encode() + b'.wav"\r\nContent-Type: audio/wav\r\n\r\n' +
        wav_bytes +
        b'\r\n--' + boundary.encode() + b'--\r\n'
    )
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/analyze?model=ml_classifier',
        data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        cls = res['classification']
        print(f"{p.upper():12s}: Label={cls['label']}, Probs={cls['probabilities']}, SynthRisk={cls['metadata'].get('synthetic_risk_score')}%, Arch={cls['metadata'].get('voice_clone_architecture')}")
