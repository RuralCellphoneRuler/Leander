# Leander-TTS: Architecture Design Document

## Overview

Leander-TTS is a streaming, on-device, German-first text-to-speech model that combines the
strongest innovations from 30+ SOTA open-source TTS models (2024-2026). It targets human-level
naturalness with zero-shot voice cloning, real-time streaming, and on-device deployment.

## Design Goals

| Requirement | Target |
|-------------|--------|
| Primary language | German (with multilingual extensibility) |
| Streaming | Yes, <200ms time-to-first-audio |
| Voice cloning | Zero-shot from 3-10 seconds of reference audio |
| On-device | Yes, via 4-bit quantization + llama.cpp/ONNX |
| Parameters | 0.6B (small) / 1.5B (medium) / 2B (large) |
| Audio quality | 24kHz, MOS ≥ 4.0, WER ≤ 3% on German test sets |
| Real-time factor | RTF < 0.5 on consumer GPU, < 1.0 on mobile (quantized) |
| License | Apache 2.0 (all components) |
| Emotion control | Inline tags for expressive speech |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Leander-TTS                          │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Speaker   │   │   Text       │   │  Emotion Tags  │  │
│  │ Encoder   │   │   Input      │   │  (optional)    │  │
│  └─────┬─────┘   └──────┬───────┘   └───────┬────────┘  │
│        │                │                    │           │
│        ▼                ▼                    ▼           │
│  ┌──────────┐   ┌──────────────────────────────────┐    │
│  │ Global   │   │         Token Embedding           │    │
│  │ Speaker  │──▶│    (text + tags + speaker cond.)  │    │
│  │ Embedding│   └──────────────┬────────────────────┘    │
│  └──────────┘                  │                         │
│                                ▼                         │
│                   ┌────────────────────────┐             │
│                   │    AR Decoder (LLM)    │             │
│                   │    Qwen2.5-1.5B        │             │
│                   │    (causal, KV-cache)  │             │
│                   └───────────┬────────────┘             │
│                               │                          │
│                               ▼ (streaming SNAC tokens)  │
│                   ┌────────────────────────┐             │
│                   │  Streaming Detokenizer │             │
│                   │  (CNN sliding window)  │             │
│                   └───────────┬────────────┘             │
│                               │                          │
│                               ▼                          │
│                        24kHz Audio Stream                │
└─────────────────────────────────────────────────────────┘
```

## Component Design

### Component 1: Audio Codec — SNAC (Multi-Scale Neural Audio Codec)

**Choice rationale:** SNAC provides the best balance of compression, quality, and streaming
compatibility. Proven in Orpheus TTS for real-time streaming.

**Specifications:**
- Input: 24kHz raw audio waveform
- Output: Multi-scale token sequence
  - Level 1: 12 Hz (coarsest — content, speaker identity)
  - Level 2: 23 Hz (prosodic features)
  - Level 3: 47 Hz (acoustic details, high-frequency content)
- Codebook size: 4096 entries per level
- Bitrate: 0.98 kbps
- Flattened token rate: 7 tokens per frame (1 + 2 + 4 across levels)
- Total tokens per second of audio: ~82 tokens/sec

**Why not alternatives:**
- EnCodec (75 Hz × 8 codebooks = 600 tokens/sec — too many for AR generation)
- Mimi (12.5 Hz × 8 codebooks = 100 tokens/sec — good but less proven for TTS quality)
- DAC (86 Hz — designed for 44.1kHz, overkill for speech)
- WavTokenizer (single codebook — quality concerns for voice cloning fidelity)

**Implementation plan:**
- Use pre-trained SNAC 24kHz model (MIT license) as starting point
- Fine-tune on German speech data (MLS-German) to optimize for German phonetics
- Validate reconstruction quality: target WER < 2% on reconstructed German speech

```
Audio Codec Token Structure (per frame = 1/12 second):

  L1: [c1]                     — 12 Hz, coarse content
  L2: [c2a, c2b]               — 23 Hz, prosody
  L3: [c3a, c3b, c3c, c3d]     — 47 Hz, fine acoustic detail

  Flattened: [c1, c2a, c2b, c3a, c3b, c3c, c3d]  (7 tokens per frame)
```

### Component 2: Speaker Encoder — BiCodec-Inspired Global Embedding

**Choice rationale:** Spark-TTS's BiCodec demonstrated that cleanly separating speaker identity
(global embedding) from content (sequential tokens) produces superior zero-shot cloning.
This is better than:
- In-context learning (VALL-E style): Wastes sequence length on reference audio tokens
- External speaker verification network (MetaVoice): Adds inference latency and complexity

**Architecture:**
```
Reference Audio (3-10s, 24kHz)
        │
        ▼
┌─────────────────────┐
│    SNAC Encoder      │  (shared with main codec)
│    (frozen)          │
└─────────┬───────────┘
          │
          ▼
   SNAC token sequence
          │
          ▼
┌─────────────────────┐
│  Speaker Encoder     │
│  - 6-layer Transformer
│  - Cross-attention pooling
│  - Projects to d_model
│  - Output: [1, d_model] global embedding
└─────────┬───────────┘
          │
          ▼
  Global Speaker Embedding (1 × 1536 vector)
```

**Design details:**
- Input: SNAC tokens from reference audio (variable length, typically 3-10 seconds)
- Architecture: 6-layer Transformer encoder with cross-attention pooling
  - Self-attention over reference SNAC tokens
  - Learnable query token attends to all positions → single embedding vector
  - Layer norm + linear projection to d_model (1536 for 1.5B Qwen)
- Output: Single global embedding vector [1, 1536]
- Injection: Added to the LLM's hidden states via:
  - Prepended as a special token embedding (position 0)
  - AND added via cross-attention at every LLM layer (deeper conditioning)
- Parameters: ~25M (small relative to LLM backbone)

**Why global embedding over sequential conditioning:**
- Fixed-size regardless of reference length → predictable memory/compute
- Clean disentanglement: embedding captures WHO, tokens capture WHAT
- Enables voice mixing, interpolation, and controllable voice creation
- No wasted context window on reference audio tokens

### Component 3: AR Decoder — Fine-Tuned Qwen2.5-1.5B

**Choice rationale:** Qwen2.5-1.5B is the optimal backbone because:
- Strong multilingual capability including German text understanding
- 1.5B parameters sits in the sweet spot of quality vs. on-device feasibility
- Apache 2.0 license
- Compatible with llama.cpp / GGUF for on-device deployment
- Proven in TTS applications (OuteTTS uses Qwen, Spark-TTS uses Qwen)
- RoPE positional encoding works well for variable-length audio generation

**Architecture modification:**
```
Standard Qwen2.5-1.5B
├── Token embedding layer
│   ├── Original text vocabulary (151,936 tokens)
│   └── + Extended SNAC vocabulary:
│       ├── SNAC L1 tokens: [SNAC_L1_0 ... SNAC_L1_4095]  (4,096 tokens)
│       ├── SNAC L2 tokens: [SNAC_L2_0 ... SNAC_L2_4095]  (4,096 tokens)
│       ├── SNAC L3 tokens: [SNAC_L3_0 ... SNAC_L3_4095]  (4,096 tokens)
│       ├── Level markers: [L1_START, L2_START, L3_START]  (3 tokens)
│       ├── Special tokens: [AUDIO_START, AUDIO_END]       (2 tokens)
│       ├── Emotion tags: [EMO_LAUGH, EMO_SIGH, ...]       (20 tokens)
│       └── Total new tokens: ~12,317
│
├── 28 Transformer layers (Qwen2.5 architecture)
│   ├── Grouped-Query Attention (GQA)
│   │   └── + Speaker cross-attention injection (every 4th layer)
│   ├── SwiGLU FFN
│   └── RMSNorm
│
└── Output head
    ├── Text logits (unused during audio generation)
    └── SNAC logits (separate heads per level for efficiency)
```

**Vocabulary extension:**
- Original Qwen2.5 vocabulary: 151,936 tokens
- Added SNAC tokens: 3 levels × 4,096 = 12,288
- Added special tokens: ~29 (markers, emotion tags)
- Total vocabulary: ~164,253 tokens
- Embedding matrix expansion: only new rows initialized randomly; original weights frozen initially

**Speaker conditioning injection:**
- The global speaker embedding is injected via learned cross-attention layers
- Added every 4th Transformer layer (7 injection points in 28 layers)
- Each injection: LayerNorm → CrossAttn(Q=hidden, KV=speaker_emb) → residual add
- Parameters per injection: ~9M → total ~63M for speaker conditioning
- This is deeper than simple prepending and produces better voice consistency

**Sequence format during training:**
```
[BOS] <text tokens> [AUDIO_START] <speaker_emb conditioning>
[L1_START] c1_t0 [L2_START] c2a_t0 c2b_t0 [L3_START] c3a_t0 c3b_t0 c3c_t0 c3d_t0
[L1_START] c1_t1 [L2_START] c2a_t1 c2b_t1 [L3_START] c3a_t1 c3b_t1 c3c_t1 c3d_t1
... (repeat for each frame)
[AUDIO_END] [EOS]
```

**Inference (streaming):**
1. Encode text + speaker embedding
2. Generate SNAC tokens autoregressively with KV-cache
3. After each complete frame (7 tokens), send to streaming detokenizer
4. Detokenizer outputs audio chunk immediately
5. Continue until [AUDIO_END] token or max length

**Token generation rate requirement:**
- Real-time playback: 12 frames/sec × 7 tokens/frame = 84 tokens/sec minimum
- Target: 150+ tokens/sec on consumer GPU for comfortable margin
- On-device (quantized): target 100+ tokens/sec

### Component 4: Streaming CNN Detokenizer

**Choice rationale:** Orpheus TTS demonstrated that a CNN-based sliding window approach
eliminates audio popping/clicking artifacts at chunk boundaries that plague naive
frame-by-frame SNAC decoding.

**Architecture:**
```
SNAC Tokens (streaming, frame-by-frame)
        │
        ▼
┌──────────────────────────────┐
│   Token Embedding (per level) │
│   L1: Embed(4096, 256)        │
│   L2: Embed(4096, 256)        │
│   L3: Embed(4096, 256)        │
└──────────┬───────────────────┘
           │
           ▼
    Concatenate + Upsample to uniform rate
           │
           ▼
┌──────────────────────────────┐
│   Causal Conv1D Stack         │
│   - 8 layers                  │
│   - Kernel size: 7            │
│   - Channels: 512             │
│   - Dilations: [1,2,4,8,1,2,4,8]
│   - Snake activation          │  ← from BigVGAN
│   - Causal padding (no future) │
│   - Sliding window: 16 frames │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   Transposed Conv Upsampler   │
│   - Upsample to 24kHz         │
│   - Anti-aliased (low-pass)   │  ← from BigVGAN
└──────────┬───────────────────┘
           │
           ▼
    24kHz Audio Chunk (~83ms per frame)
```

**Key design decisions:**
- **Causal convolutions**: No future lookahead → true streaming
- **Snake activations**: Periodic inductive bias from BigVGAN captures harmonics better than ReLU
- **Anti-aliased upsampling**: Low-pass filters after transposed convolutions prevent aliasing (from BigVGAN)
- **Sliding window (16 frames)**: Provides ~1.3 seconds of context for smooth transitions without unbounded memory growth
- **Parameters**: ~15M (lightweight, runs on any device)

**Latency budget:**
- LLM generates first frame: ~100-150ms (TTFB)
- Detokenizer processes frame: ~5ms
- Total time-to-first-audio: ~105-155ms
- Subsequent chunks: continuous streaming at 12 frames/sec

### Component 5: Emotion and Expression Control

**Design:** Inline text tags (inspired by Orpheus and Dia) with German-native labels.

**Supported tags:**
```
Tag                 │ Effect
────────────────────┼──────────────────────────────
<lacht>             │ Laughter inserted
<kichert>           │ Giggling
<seufzt>            │ Sighing
<flüstert>          │ Whispering mode
<ruft>              │ Shouting/calling mode
<weint>             │ Crying/sobbing
<hustet>            │ Cough inserted
<ähm>               │ Hesitation/filler
<pause>             │ Natural pause (0.5-1.0s)
<atmet>             │ Breathing sound
<gähnt>             │ Yawning
<stöhnt>            │ Groaning
<schnauft>          │ Heavy breathing / panting
<räuspert>          │ Throat clearing
<singt>             │ Singing mode
<betont>...</betont> │ Emphasized span
<leise>...</leise>  │ Soft/quiet span
<schnell>...</schnell> │ Fast speaking rate span
<langsam>...</langsam> │ Slow speaking rate span
```

**Implementation:**
- Each tag maps to a special token in the extended vocabulary
- Tags are placed inline with text: `"Hallo <lacht> das ist ja lustig"`
- The LLM learns to produce appropriate audio tokens after encountering these tags
- Span tags (betont, leise, schnell, langsam) modify generation for the enclosed text
- Training: Tag-augmented transcripts aligned with expressive speech data

## Model Variants

| Variant | LLM Backbone | Total Params | Target Device | Quant Size |
|---------|-------------|-------------|---------------|-----------|
| Leander-S | Qwen3-0.6B | ~700M | Mobile / Edge | ~400MB (Q4) |
| Leander-M | Qwen2.5-1.5B | ~1.6B | Laptop / Desktop | ~900MB (Q4) |
| Leander-L | Qwen2.5-3B | ~3.2B | GPU Server | ~1.8GB (Q4) |

Primary development target: **Leander-M (1.5B)**, then distill to Leander-S.

## Training Pipeline

### Phase 1: Codec Preparation
```
Task: Fine-tune SNAC on German speech
Data: MLS-German (~1,900 hours) + CommonVoice-DE (~1,200 hours)
      + Thorsten dataset (~40 hours, high quality single speaker)
Goal: WER < 2% on reconstructed German speech
Duration: ~2-3 days on 1× A100
```

### Phase 2: Speaker Encoder Pre-training
```
Task: Train speaker encoder on multi-speaker German data
Data: MLS-German (multi-speaker) + VCTK (English, for speaker diversity)
      + CommonVoice-DE
Objective: Contrastive learning (same speaker = close, different = far)
           + reconstruction loss through frozen SNAC decoder
Goal: Speaker similarity > 0.85 on held-out speakers
Duration: ~1-2 days on 1× A100
```

### Phase 3: LLM Fine-tuning (Main Training)
```
Task: Fine-tune Qwen2.5-1.5B to predict SNAC tokens from text + speaker
Data: All German speech data (~3,000+ hours)
      Augmented with emotion tags where available
Strategy:
  - Stage 3a: Freeze original LLM weights, train only new components
              (extended embeddings, speaker cross-attention, output heads)
              for 10K steps — prevents catastrophic forgetting
  - Stage 3b: Unfreeze all weights, train end-to-end with lower LR
              for 50K-100K steps
  - Stage 3c: Fine-tune on high-quality expressive German speech
              with emotion tags for 10K steps
Objective: Cross-entropy loss on SNAC token prediction
           + auxiliary speaker consistency loss
Batch size: Effective 256 (via gradient accumulation)
Sequence length: 4096 tokens (~45 seconds of audio)
Duration: ~5-7 days on 4× A100 (or equivalent)
```

### Phase 4: Streaming Detokenizer Training
```
Task: Train CNN detokenizer for streaming SNAC → audio
Data: Same German speech data, SNAC-encoded
Objective: Multi-scale STFT loss + adversarial loss (HiFi-GAN discriminator)
           + feature matching loss
Goal: Transparent quality (indistinguishable from SNAC offline decoder)
Duration: ~2-3 days on 1× A100
```

### Phase 5: End-to-End Optimization
```
Task: Joint fine-tuning of full pipeline
Data: High-quality German speech subset
Strategy: Generate audio end-to-end, compute losses on final waveform
          Backpropagate through detokenizer (frozen) to LLM
Goal: Optimize for end-to-end quality metrics (MOS, WER, speaker sim)
Duration: ~1-2 days on 4× A100
```

## Training Data Plan

### Primary Datasets

| Dataset | Language | Hours | Speakers | Quality | License |
|---------|----------|-------|----------|---------|---------|
| MLS-German | de | ~1,900 | ~250 | Medium-High (audiobook) | CC-BY-4.0 |
| CommonVoice-DE | de | ~1,200 | ~10K+ | Variable (crowd-sourced) | CC-0 |
| Thorsten | de | ~40 | 1 | Very High (studio) | CC-0 |
| Thorsten-Emotional | de | ~4 | 1 | Very High (expressive) | CC-0 |
| LibriTTS-R | en | ~585 | ~2,400 | High (enhanced) | CC-BY-4.0 |
| GigaSpeech | en | ~10K | ~30K+ | Variable | Apache 2.0 |

**Total: ~13,700+ hours** (primarily German, English for speaker diversity)

### Data Processing Pipeline

```
Raw Audio → Resampling (24kHz) → VAD / Silence Trimming
         → SNR Filtering (> 15dB)
         → Duration Filtering (2-30 seconds)
         → SNAC Tokenization
         → Text Normalization (German-specific):
           - Number expansion ("42" → "zweiundvierzig")
           - Abbreviation expansion
           - Umlaut handling
           - Compound word handling
         → CTC Alignment (for emotion tag insertion where applicable)
         → Speaker Clustering (for pseudo-speaker labels in CommonVoice)
         → Train/Val/Test Split (stratified by speaker)
```

## Inference Pipeline

### Standard Generation (Streaming)
```python
# Pseudocode for streaming inference
def generate_streaming(text: str, speaker_ref: Audio, emotion_tags: bool = True):
    # 1. Encode reference audio
    ref_snac_tokens = snac_encoder(speaker_ref)           # [T_ref, 7]
    speaker_embedding = speaker_encoder(ref_snac_tokens)   # [1, 1536]

    # 2. Tokenize text (with optional emotion tags)
    text_tokens = tokenizer.encode(text)                   # [T_text]

    # 3. Build prompt
    prompt = [BOS] + text_tokens + [AUDIO_START]

    # 4. Initialize KV-cache with speaker conditioning
    kv_cache = llm.init_cache(speaker_embedding=speaker_embedding)

    # 5. Autoregressive streaming generation
    audio_buffer = StreamingBuffer(window_size=16)
    frame_tokens = []

    for token in llm.generate(prompt, kv_cache=kv_cache, stream=True):
        frame_tokens.append(token)

        if len(frame_tokens) == 7:  # Complete frame
            audio_chunk = detokenizer.decode_frame(
                frame_tokens, context=audio_buffer
            )
            yield audio_chunk  # Stream to speaker
            audio_buffer.push(frame_tokens)
            frame_tokens = []

        if token == AUDIO_END:
            break
```

### Voice Cloning
```python
# Zero-shot: just provide reference audio
audio = generate_streaming(
    text="Hallo, ich bin ein Klon deiner Stimme.",
    speaker_ref=load_audio("reference.wav")  # 3-10 seconds
)

# Fine-tuned (higher quality): LoRA fine-tune on target speaker
# Requires 5-30 minutes of target speaker data
lora = train_speaker_lora(speaker_data, base_model, steps=1000)
audio = generate_streaming(text, speaker_ref, lora=lora)
```

## On-Device Deployment

### Quantization Strategy
```
Model Component     │ Precision  │ Size (Leander-M)
────────────────────┼────────────┼──────────────
SNAC Encoder        │ FP16       │ ~30 MB
Speaker Encoder     │ INT8       │ ~25 MB
LLM (Qwen2.5-1.5B) │ Q4_K_M     │ ~900 MB
Streaming Detokenizer│ FP16      │ ~30 MB
────────────────────┼────────────┼──────────────
Total               │ Mixed      │ ~985 MB
```

### Runtime Options
1. **llama.cpp**: GGUF export of the LLM backbone + custom SNAC/detokenizer plugins
2. **ONNX Runtime**: Full pipeline export for cross-platform deployment
3. **MLX** (Apple Silicon): Optimized for macOS/iOS
4. **TensorRT** (NVIDIA): Server-side optimization

### Target Performance (Leander-M, Q4)
| Platform | Tokens/sec | RTF | Real-time? |
|----------|-----------|-----|-----------|
| RTX 4090 | 300+ | ~0.28 | Yes (3.6x) |
| RTX 3060 | 120+ | ~0.70 | Yes (1.4x) |
| M2 MacBook | 100+ | ~0.84 | Yes (1.2x) |
| Snapdragon 8 Gen 3 | 60-80 | ~1.0 | Borderline |
| Raspberry Pi 5 | 15-25 | ~3.5 | No (offline) |

## Evaluation Plan

### Metrics
1. **WER (Word Error Rate)**: ASR on generated speech using Whisper-large-v3
   - Target: ≤ 3% on German test sentences
2. **Speaker Similarity (SIM-o)**: Cosine similarity of speaker embeddings (generated vs reference)
   - Target: ≥ 0.80
3. **MOS (Mean Opinion Score)**: Human evaluation of naturalness (1-5 scale)
   - Target: ≥ 4.0
4. **RTF (Real-Time Factor)**: Generation time / audio duration
   - Target: < 0.5 on consumer GPU
5. **TTFA (Time to First Audio)**: Latency from text input to first audio chunk
   - Target: < 200ms on GPU, < 500ms on device
6. **UTMOS**: Automated MOS prediction
   - Target: ≥ 3.8

### Test Sets
- MLS-German test split (in-domain)
- CommonVoice-DE test split (diverse speakers)
- Custom German expressiveness test set (emotion tags)
- Zero-shot cloning test set (unseen speakers, 5s reference)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| German quality lags English | High | High | Focus training data on German; use German-specific text normalization |
| Voice cloning speaker leakage | Medium | Medium | Contrastive training; privacy filtering of training data |
| Streaming artifacts at boundaries | Medium | High | CNN sliding window (proven in Orpheus); extensive testing |
| On-device too slow | Medium | Medium | Aggressive quantization; distill to 0.6B variant |
| SNAC codec quality ceiling | Low | High | Fine-tune SNAC on German; fallback to DAC if needed |
| Emotion tags not learned well | Medium | Medium | Curate expressive training data; tag augmentation |

## Project Structure

```
leander-tts/
├── configs/
│   ├── model/
│   │   ├── leander_s.yaml          # 0.6B config
│   │   ├── leander_m.yaml          # 1.5B config (primary)
│   │   └── leander_l.yaml          # 3B config
│   ├── training/
│   │   ├── phase1_codec.yaml
│   │   ├── phase2_speaker.yaml
│   │   ├── phase3_lm.yaml
│   │   ├── phase4_detokenizer.yaml
│   │   └── phase5_e2e.yaml
│   └── inference/
│       ├── streaming.yaml
│       └── quantized.yaml
├── leander_tts/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── codec.py                # SNAC wrapper + fine-tuning
│   │   ├── speaker_encoder.py      # BiCodec-style speaker encoder
│   │   ├── lm.py                   # Extended Qwen2.5 with SNAC heads
│   │   ├── detokenizer.py          # Streaming CNN detokenizer
│   │   ├── attention.py            # Speaker cross-attention layers
│   │   └── leander.py              # Full model composition
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Training dataset classes
│   │   ├── preprocessing.py        # Audio preprocessing pipeline
│   │   ├── text_normalizer.py      # German text normalization
│   │   ├── tokenizer.py            # Extended tokenizer with SNAC vocab
│   │   └── emotion_tags.py         # Emotion tag processing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── losses.py               # Loss functions
│   │   ├── scheduler.py            # LR schedulers
│   │   └── callbacks.py            # Logging, checkpointing
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── streaming.py            # Streaming generation pipeline
│   │   ├── voice_clone.py          # Voice cloning utilities
│   │   └── server.py               # HTTP API server
│   ├── export/
│   │   ├── __init__.py
│   │   ├── gguf.py                 # llama.cpp GGUF export
│   │   ├── onnx.py                 # ONNX export
│   │   └── quantize.py             # Quantization utilities
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # WER, SIM, UTMOS computation
│       └── benchmark.py            # Full evaluation pipeline
├── scripts/
│   ├── prepare_data.py             # Data download + preprocessing
│   ├── train_codec.py              # Phase 1
│   ├── train_speaker.py            # Phase 2
│   ├── train_lm.py                 # Phase 3
│   ├── train_detokenizer.py        # Phase 4
│   ├── train_e2e.py                # Phase 5
│   ├── evaluate.py                 # Run full evaluation
│   └── demo.py                     # Interactive demo
├── tests/
│   ├── test_codec.py
│   ├── test_speaker_encoder.py
│   ├── test_lm.py
│   ├── test_detokenizer.py
│   ├── test_streaming.py
│   └── test_e2e.py
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Dependencies

```
# Core
torch >= 2.2.0
transformers >= 4.40.0
accelerate >= 0.30.0
snac                          # SNAC codec
tokenizers                    # Fast tokenizer

# Training
deepspeed >= 0.14.0           # Distributed training
wandb                         # Experiment tracking
datasets                      # HuggingFace datasets
librosa                       # Audio processing
soundfile                     # Audio I/O
torchaudio                    # Audio transforms

# Inference
onnxruntime                   # On-device inference
llama-cpp-python              # llama.cpp bindings

# Evaluation
whisper                       # ASR for WER
resemblyzer                   # Speaker similarity
pesq                          # Perceptual quality
```

## Timeline Estimate

| Phase | Task | Duration | Compute |
|-------|------|----------|---------|
| 0 | Project setup, data download + preprocessing | 1 week | Minimal |
| 1 | SNAC fine-tuning on German | 3 days | 1× A100 |
| 2 | Speaker encoder training | 2 days | 1× A100 |
| 3 | LLM fine-tuning (main training) | 5-7 days | 4× A100 |
| 4 | Streaming detokenizer training | 3 days | 1× A100 |
| 5 | End-to-end optimization | 2 days | 4× A100 |
| 6 | Evaluation + iteration | 1 week | 1× A100 |
| 7 | Quantization + on-device export | 3 days | Minimal |
| **Total** | | **~5-6 weeks** | **~200 A100-hours** |
