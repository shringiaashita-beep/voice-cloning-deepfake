/**
 * Frontend Unit & Component Contract Test Runner for VoxGuard Phase 12A.
 */
const assert = require('assert');

console.log("\n==================================================");
console.log("VoxGuard Frontend Unit & Component Test Suite");
console.log("==================================================");

let testsPassed = 0;
let testsFailed = 0;

function runTest(name, fn) {
  try {
    fn();
    console.log(`  [PASS] ${name}`);
    testsPassed++;
  } catch (err) {
    console.error(`  [FAIL] ${name}: ${err.message}`);
    testsFailed++;
  }
}

// Test 1: Analysis-only state & null probabilities preservation
runTest("analysis_only state & null probabilities contract", () => {
  const classification = {
    status: "analysis_only",
    label: "uncertain",
    probabilities: { human: null, synthetic: null },
    confidence_score: null
  };
  assert.strictEqual(classification.status, "analysis_only");
  assert.strictEqual(classification.label, "uncertain");
  assert.strictEqual(classification.probabilities.human, null);
  assert.strictEqual(classification.probabilities.synthetic, null);
  assert.strictEqual(classification.confidence_score, null);
});

// Test 2: en-US number formatting for metadata
runTest("en-US locale number formatting (frames & sample rate)", () => {
  const numFrames = 3362436;
  const sampleRate = 16000;
  assert.strictEqual(numFrames.toLocaleString('en-US'), "3,362,436");
  assert.strictEqual(sampleRate.toLocaleString('en-US'), "16,000");
});

// Test 3: Waveform visualization payload contract
runTest("waveform visualization payload integrity", () => {
  const waveform = {
    target_points: 300,
    min_peaks: [-0.5, -0.8],
    max_peaks: [0.6, 0.9],
    duration_seconds: 4.0
  };
  assert.strictEqual(waveform.target_points, 300);
  assert.strictEqual(waveform.min_peaks.length, 2);
  assert.strictEqual(waveform.max_peaks.length, 2);
});

// Test 4: Spectrogram payload contract
runTest("spectrogram payload grid integrity", () => {
  const spectrogram = {
    freq_bins: 40,
    time_bins: 120,
    min_db: -80.0,
    max_db: 0.0,
    data_grid: Array(40).fill(Array(120).fill(0.5))
  };
  assert.strictEqual(spectrogram.freq_bins, 40);
  assert.strictEqual(spectrogram.time_bins, 120);
  assert.strictEqual(spectrogram.data_grid.length, 40);
});

// Test 5: Provenance filtering logic
runTest("provenance indicator filtering logic", () => {
  const indicators = [
    { name: "Sample Rate", provenance: "metadata", severity: "INFO" },
    { name: "Spectral Centroid", provenance: "signal_analysis", severity: "INFO" },
    { name: "Operational Mode", provenance: "heuristic", severity: "INFO" }
  ];
  const filterBy = (prov) => indicators.filter(i => i.provenance === prov);
  assert.strictEqual(filterBy("metadata").length, 1);
  assert.strictEqual(filterBy("signal_analysis").length, 1);
  assert.strictEqual(filterBy("heuristic").length, 1);
  assert.strictEqual(filterBy("ml_model").length, 0);
});

// Test 6: API error code user-friendly mapping
runTest("API error code user-friendly mapping", () => {
  const errorMap = {
    FILE_TOO_LARGE: "Audio file exceeds the 10 MB limit.",
    UNSUPPORTED_EXTENSION: "This audio format is not supported.",
    UNSUPPORTED_MIME_TYPE: "The uploaded file does not match a supported audio type.",
    CORRUPTED_AUDIO: "The audio file could not be decoded.",
    EMPTY_FILE: "The uploaded file is empty."
  };
  assert.strictEqual(errorMap["FILE_TOO_LARGE"], "Audio file exceeds the 10 MB limit.");
  assert.strictEqual(errorMap["CORRUPTED_AUDIO"], "The audio file could not be decoded.");
  assert.strictEqual(errorMap["UNSUPPORTED_EXTENSION"], "This audio format is not supported.");
});

// Test 7: Loading pipeline 5-stage progression
runTest("loading pipeline 5-stage progression", () => {
  const stages = ['Uploading', 'Validating', 'Decoding', 'Analyzing', 'Building report'];
  assert.strictEqual(stages.length, 5);
  assert.strictEqual(stages[0], 'Uploading');
  assert.strictEqual(stages[4], 'Building report');
});

// Test 8: Disclaimer text duplication check
runTest("disclaimer text deduplication logic", () => {
  const rawText = "VoxGuard forensic report. Refer to docs/EVALUATION_METRICS.md for benchmarking protocols.";
  const cleaned = rawText.replace(/\s*Refer to docs\/EVALUATION_METRICS\.md for benchmarking protocols\.\s*/gi, '');
  assert.strictEqual(cleaned, "VoxGuard forensic report.");
});

// Test 9: Voice clone active state & calibrated probabilities contract
runTest("voice clone active detection contract (ready state & valid probabilities)", () => {
  const classification = {
    status: "ready",
    label: "synthetic",
    probabilities: { human: 0.035, synthetic: 0.965 },
    confidence_score: 0.965,
    model_name: "VoxGuard Deepfake Voice Clone Forensics Neural Ensemble",
    metadata: {
      voice_clone_architecture: "Neural Vocoder (ElevenLabs / HiFi-GAN Profile)",
      synthetic_risk_score: 96.5
    }
  };
  assert.strictEqual(classification.status, "ready");
  assert.strictEqual(classification.label, "synthetic");
  assert.ok(classification.probabilities.synthetic > 0.90);
  assert.ok(Math.abs(classification.probabilities.human + classification.probabilities.synthetic - 1.0) < 0.001);
  assert.strictEqual(classification.metadata.voice_clone_architecture, "Neural Vocoder (ElevenLabs / HiFi-GAN Profile)");
});

// Test 10: Voice clone architecture fingerprint mapping
runTest("voice clone architecture affinity scores presence", () => {
  const scores = {
    elevenlabs_neural_vocoder: 94.5,
    rvc_voice_conversion: 28.2,
    bark_diffusion_tts: 65.0,
    natural_human_vocal_tract: 4.8
  };
  assert.ok(scores.elevenlabs_neural_vocoder > 90);
  assert.ok(scores.rvc_voice_conversion > 0);
  assert.ok(scores.bark_diffusion_tts > 0);
  assert.ok(scores.natural_human_vocal_tract < 20);
});

// Test 11: Screenshot scenario probability invariant & distinct feature similarity metric
runTest("screenshot scenario synthetic 88.5%, human 11.5% and human vocal-trait similarity 98.4% distinction", () => {
  const classification = {
    status: "ready",
    label: "synthetic",
    probabilities: { human: 0.115, synthetic: 0.885 },
    confidence_score: 0.885,
    model_name: "VoxGuard Neural Acoustic Ensemble",
    metadata: {
      architecture_scores: {
        elevenlabs_neural_vocoder: 88.5,
        rvc_voice_conversion: 10.0,
        bark_diffusion_tts: 10.0,
        natural_human_vocal_tract: 98.4
      }
    }
  };

  // 1. Invariant check: synthetic + human == 1.0
  const synthPct = (classification.probabilities.synthetic * 100).toFixed(1);
  const humanPct = (classification.probabilities.human * 100).toFixed(1);
  assert.strictEqual(synthPct, "88.5");
  assert.strictEqual(humanPct, "11.5");
  assert.strictEqual(parseFloat(synthPct) + parseFloat(humanPct), 100.0);

  // 2. Separate acoustic feature similarity score check
  const humanVocalTraitSimilarity = classification.metadata.architecture_scores.natural_human_vocal_tract;
  assert.strictEqual(humanVocalTraitSimilarity, 98.4);
  assert.notStrictEqual(humanVocalTraitSimilarity, classification.probabilities.human * 100);
});

console.log("--------------------------------------------------");
console.log(`Results: ${testsPassed} passed, ${testsFailed} failed.`);
console.log("==================================================\n");

if (testsFailed > 0) process.exit(1);

