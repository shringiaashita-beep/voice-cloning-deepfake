import urllib.request
import json
import wave
import struct
import math
import io

# Generate 1-second 16kHz sine wave WAV file
buf = io.BytesIO()
with wave.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    samples = [int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)]
    wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
wav_data = buf.getvalue()

boundary = '----WebKitFormBoundaryVoxGuardTest'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test_audio.wav"\r\n'
    f'Content-Type: audio/wav\r\n\r\n'
).encode('utf-8') + wav_data + (
    f'\r\n--{boundary}\r\n'
    f'Content-Disposition: form-data; name="model"\r\n\r\n'
    f'voice_clone_detector\r\n'
    f'--{boundary}--\r\n'
).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/v1/analyze?model=voice_clone_detector',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('HTTP Status:', resp.status)
        print('SUCCESS!')
        print('Success flag:', res.get('success'))
        print('Classification:', json.dumps(res.get('classification'), indent=2))
        forensic_report = res.get('forensic_report', {})
        print('Evidence indicators count:', len(forensic_report.get('evidence_indicators', [])))
        print('Acoustic observations keys:', list(forensic_report.get('acoustic_observations', {}).keys()))
        viz = res.get('visualization', {})
        print('Spectrogram dimensions:', viz.get('spectrogram', {}).get('freq_bins'), 'x', viz.get('spectrogram', {}).get('time_bins'))
        print('Waveform points:', viz.get('waveform', {}).get('target_points'))
        print('Top indicators:', json.dumps(forensic_report.get('evidence_indicators', [])[:3], indent=2))
except Exception as e:
    print('ERROR:', e)
