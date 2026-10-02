import urllib.request
import json
import wave
import struct
import math
import io

def generate_preset_wav(preset: str, duration: float = 2.0, sample_rate: int = 16000) -> bytes:
    num_frames = int(duration * sample_rate)
    samples = []
    
    if preset == 'elevenlabs':
        # Pure harmonic speech with 0 micro-jitter and sharp cutoff above 3.5 kHz
        base_freq = 210.0
        harmonics = [1, 2, 3, 4, 5, 7, 9, 12, 15]
        for i in range(num_frames):
            t = i / sample_rate
            val = sum((1.0 / math.sqrt(h)) * math.sin(2 * math.pi * (base_freq * h) * t) for h in harmonics)
            env = math.sin(math.pi * (i / num_frames))
            sample = max(-1.0, min(1.0, val * 0.25 * env))
            samples.append(int(sample * 32767))
            
    elif preset == 'rvc':
        # Step jumps between frequencies
        phase = 0.0
        for i in range(num_frames):
            t = i / sample_rate
            note_idx = int(t * 3.5) % 3
            f = 220.0 if note_idx == 0 else (260.0 if note_idx == 1 else 330.0)
            phase += (2 * math.pi * f) / sample_rate
            val = math.sin(phase) + 0.6 * math.sin(2 * phase) + 0.4 * math.sin(3 * phase)
            env = math.sin(math.pi * (i / num_frames))
            sample = max(-1.0, min(1.0, val * 0.3 * env))
            samples.append(int(sample * 32767))
            
    elif preset == 'bark':
        # High spectral flux
        phase = 0.0
        for i in range(num_frames):
            t = i / sample_rate
            f = 180.0 + 80.0 * math.sin(2 * math.pi * 1.5 * t)
            phase += (2 * math.pi * f) / sample_rate
            dither = (math.sin(i * 123.456) % 1.0) * 0.15
            val = math.sin(phase) + dither
            env = math.sin(math.pi * (i / num_frames))
            sample = max(-1.0, min(1.0, val * 0.35 * env))
            samples.append(int(sample * 32767))
            
    else:  # human
        # Natural jitter, rich resonances, breathing noise
        base_f = 145.0
        phase = 0.0
        for i in range(num_frames):
            t = i / sample_rate
            jitter = 1.0 + 0.015 * math.sin(2 * math.pi * 7.2 * t) + 0.008 * math.sin(2 * math.pi * 14.1 * t)
            f_inst = base_f * jitter
            phase += (2 * math.pi * f_inst) / sample_rate
            val = math.sin(phase) + 0.8 * math.sin(2 * phase) + 0.6 * math.sin(3 * phase) + 0.3 * math.sin(4 * phase)
            # Add aspiration noise
            noise = (math.sin(i * 987.654) % 1.0 - 0.5) * 0.04
            env = math.sin(math.pi * (i / num_frames))
            sample = max(-1.0, min(1.0, (val * 0.3 + noise) * env))
            samples.append(int(sample * 32767))

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
    return buf.getvalue()

def test_preset(name: str):
    wav_bytes = generate_preset_wav(name)
    boundary = '----WebKitFormBoundaryVoxGuardBench'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{name}_test.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'
    ).encode('utf-8') + wav_bytes + (
        f'\r\n--{boundary}--\r\n'
    ).encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/v1/analyze?model=voice_clone_detector',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        cls = res['classification']
        meta = cls.get('metadata', {})
        print(f"[{name.upper()}]")
        print(f"  Label: {cls.get('label')}")
        print(f"  Probabilities: {cls.get('probabilities')}")
        print(f"  Confidence Score: {cls.get('confidence_score')}")
        print(f"  Detected Architecture: {meta.get('voice_clone_architecture')}")
        print(f"  Synthetic Risk Level: {meta.get('synthetic_risk_level')}")
        print(f"  Architecture Scores: {meta.get('architecture_scores')}")
        print("-" * 50)

if __name__ == '__main__':
    print("Testing All 4 Presets on Live Backend:")
    print("=" * 50)
    for p in ['elevenlabs', 'rvc', 'bark', 'human']:
        test_preset(p)
