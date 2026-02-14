# State-of-the-Art Open-Source Text-to-Speech Models (2024–2026)

## Comprehensive Research Survey

**Date:** February 2026
**Scope:** Open-source and open-weight TTS models, neural audio codecs, vocoders, and related architectures released or significantly updated between 2024 and early 2026.

---

## Table of Contents

1. [Flow-Matching & Diffusion-Based TTS](#1-flow-matching--diffusion-based-tts)
   - 1.1 F5-TTS
   - 1.2 E2 TTS
   - 1.3 Voicebox
   - 1.4 P-Flow
   - 1.5 Matcha-TTS
   - 1.6 Grad-TTS
   - 1.7 SpeechFlow
2. [Neural Audio Codecs & LLM-Based TTS](#2-neural-audio-codecs--llm-based-tts)
   - 2.1 VALL-E
   - 2.2 VALL-E 2
   - 2.3 EnCodec
   - 2.4 SpeechTokenizer
   - 2.5 SNAC
   - 2.6 Descript Audio Codec (DAC)
   - 2.7 WavTokenizer
   - 2.8 Mimi (Kyutai)
   - 2.9 AudioDec
   - 2.10 FunCodec
3. [Streaming & Efficient TTS](#3-streaming--efficient-tts)
   - 3.1 Orpheus TTS
   - 3.2 Dia / Dia2
   - 3.3 OuteTTS
   - 3.4 MARS5-TTS
   - 3.5 MetaVoice-1B
   - 3.6 Pheme
4. [Voice Cloning & Multi-Speaker TTS](#4-voice-cloning--multi-speaker-tts)
   - 4.1 XTTS v2 (Coqui)
   - 4.2 OpenVoice v2
   - 4.3 Kokoro
   - 4.4 Spark-TTS
   - 4.5 CSM (Sesame)
   - 4.6 GPT-SoVITS
   - 4.7 Fish Speech
5. [Vocoders & Waveform Generators](#5-vocoders--waveform-generators)
   - 5.1 HiFi-GAN
   - 5.2 BigVGAN / BigVGAN-v2
   - 5.3 Vocos
   - 5.4 WaveGrad
   - 5.5 UnivNet
6. [German Language TTS Support](#6-german-language-tts-support)
7. [Comparative Summary Tables](#7-comparative-summary-tables)
8. [Key Architectural Trends](#8-key-architectural-trends)

---

## 1. Flow-Matching & Diffusion-Based TTS

### 1.1 F5-TTS (Fairytaler that Fakes Fluent and Faithful Speech)

**Paper:** [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) (October 2024)
**GitHub:** [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)

**Architecture:**
- Non-autoregressive, flow-matching-based text-to-speech system
- Uses a **Diffusion Transformer (DiT)** with **ConvNeXt V2** architecture as the backbone
- Text input is padded with filler tokens to match the length of the mel spectrogram, then character-level representations are fed directly into the DiT — eliminating the need for a separate text encoder, duration model, or phoneme alignment
- Generates mel spectrograms, decoded to waveform via a vocoder (Vocos recommended)
- Builds upon the E2 TTS paradigm but replaces the UNet with a DiT + ConvNeXt for improved efficiency

**Key Innovations:**
- **No phonemizer or aligner needed**: Text is represented at the character level with padding, dramatically simplifying the pipeline
- **Sway Sampling**: A novel inference-time strategy that adjusts the flow-matching trajectory, improving generation quality and reducing the number of required diffusion steps (typically 16–32 steps)
- **ConvNeXt V2 backbone**: Provides better local pattern capture than pure attention, improving convergence speed (F5-TTS converges significantly faster than E2 TTS)
- **In-context learning** for zero-shot voice cloning: A reference audio clip is concatenated with the target generation in the time dimension; the model learns to match speaker characteristics from context

**Zero-Shot Voice Cloning:** Yes. Uses in-context learning — a reference audio clip (~3–15 seconds) is prepended in the diffusion process. No fine-tuning needed. Supports cross-lingual voice cloning.

**Streaming Support:** Not natively streaming. Inference is non-autoregressive (generates full mel spectrogram). Chunked inference can be implemented with overlapping windows. Community implementations exist for chunk-based streaming (e.g., F5-TTS-ONNX).

**Parameters:** ~335M (DiT backbone). Lightweight compared to LLM-based approaches.

**Real-Time Factor (RTF):**
- RTF ~0.15 on A100 (with 32 diffusion steps)
- Significantly faster than E2 TTS (~7.7x speedup reported)
- With ONNX optimization and reduced steps, can approach real-time on consumer GPUs

**Audio Quality Metrics:**
- **WER (word error rate):** ~1.8–2.5% on LibriSpeech test sets (competitive with SOTA)
- **Speaker Similarity (SIM-o):** 0.68–0.72 (comparable to or better than E2 TTS and NaturalSpeech 3)
- **Naturalness MOS:** Reported in the range of 3.8–4.1 in evaluation
- Achieves quality competitive with closed-source systems like VoiceBox and NaturalSpeech 3

**Training Data:**
- Emilia dataset (~95K hours multilingual speech, CC-BY-NC-4.0)
- Wenetspeech4TTS (~12K hours filtered Chinese data, CC-BY-4.0)
- Can be trained on custom datasets; community fine-tunes exist for many languages

**License:** CC-BY-NC-4.0 (model weights); MIT (code)

**Multilingual:** English, Chinese natively. Community fine-tunes for Japanese, French, Hindi, German, Spanish, and more.

---

### 1.2 E2 TTS (Embarrassingly Easy Text-to-Speech)

**Paper:** [arXiv:2406.18009](https://arxiv.org/abs/2406.18009) (June 2024, Microsoft)
**GitHub:** No official release; F5-TTS is the primary open-source implementation of this paradigm

**Architecture:**
- Flow-matching-based generative model
- Uses a **UNet** backbone (standard diffusion architecture)
- Text is converted to characters, padded with filler tokens to match mel spectrogram length
- Audio generation is conditioned on this character sequence via cross-attention
- Fully non-autoregressive

**Key Innovations:**
- **"Embarrassingly easy" design philosophy**: Eliminates phoneme conversion, grapheme-to-phoneme (G2P) modules, duration prediction, and text-audio alignment entirely
- Demonstrates that with sufficient data and flow-matching, a simple character-input + diffusion model can match or exceed complex multi-stage TTS pipelines
- Inspired F5-TTS, Voicebox, and similar architectures

**Zero-Shot Voice Cloning:** Yes, via in-context learning (same approach as F5-TTS).

**Streaming Support:** No. Full mel spectrogram generation.

**Parameters:** ~335M (UNet backbone)

**RTF:** Slower than F5-TTS due to UNet architecture; ~32+ diffusion steps needed.

**Audio Quality Metrics:**
- Demonstrated competitive WER and speaker similarity with NaturalSpeech 3 and Voicebox
- Naturalness MOS ~3.6–3.9 (slightly lower than F5-TTS variant)

**Training Data:** Internal Microsoft datasets (details not fully public). Emilia used in reproductions.

**License:** Research paper only; no official model release. F5-TTS serves as the open-source proxy.

---

### 1.3 Voicebox (Meta)

**Paper:** [arXiv:2306.15687](https://arxiv.org/abs/2306.15687) (June 2023, updated 2024)
**GitHub:** No official open-source release from Meta

**Architecture:**
- Non-autoregressive, flow-matching-based model
- Uses a **Transformer** backbone conditioned on both text (phonemes via IPA) and audio context
- Audio representation: 80-dim log mel spectrograms at 16kHz, 100 frames/sec
- **Audio context masking**: During training, random spans of audio are masked, and the model learns to infill them — enabling speech editing, noise removal, and style transfer as natural capabilities
- Uses an external duration model (phoneme aligner) for text-audio alignment

**Key Innovations:**
- **Infilling paradigm**: By training to infill masked audio spans, Voicebox naturally supports: zero-shot TTS, speech editing (replace words in existing audio), noise removal, style conversion, and cross-lingual style transfer
- **Conditional flow matching (CFM)**: Uses optimal transport conditional flow matching for efficient training and inference
- First large-scale demonstration that flow-matching could replace diffusion for high-quality speech synthesis
- Outperformed VALL-E and YourTTS on zero-shot benchmarks at the time of release

**Zero-Shot Voice Cloning:** Yes. 3–10 seconds of reference audio used as context via the infilling mechanism. State-of-the-art speaker similarity at time of release.

**Streaming Support:** No. Non-autoregressive, full-sequence generation.

**Parameters:** ~330M (Transformer backbone)

**RTF:** Efficient; 10–20 flow-matching steps sufficient. Faster than autoregressive approaches for long utterances.

**Audio Quality Metrics:**
- **WER:** 1.9% (vs 5.9% for VALL-E on LibriSpeech continuation)
- **SIM-o:** 0.681 (vs 0.580 for VALL-E)
- **CMOS:** +0.5 over VALL-E in human evaluation
- MOS scores near ground truth for in-domain speakers

**Training Data:** 60K hours of English audiobook speech (LibriLight) + multilingual data for cross-lingual experiments.

**License:** Research only — Meta did not release model weights. Several community reproductions exist (VoiceBox-pytorch, etc.).

---

### 1.4 P-Flow (Prior Flow Matching)

**Paper:** [arXiv:2305.02561](https://arxiv.org/abs/2305.02561) (May 2023)
**GitHub:** No official release; referenced implementations exist

**Architecture:**
- Flow-matching generative model with a speech-prompted text encoder
- Two-stage: (1) Text encoder produces a prior distribution conditioned on phonemes + speaker embedding, (2) Flow-matching decoder transforms this prior into mel spectrograms
- Uses a **Transformer** for both stages
- External duration predictor and phoneme aligner required

**Key Innovations:**
- **Speech-prompted text encoder**: The text encoder is conditioned on a reference speech sample, allowing it to produce speaker-aware linguistic features before the flow-matching step
- Demonstrates that flow-matching with a good prior produces higher quality with fewer sampling steps than diffusion
- Single-stage generation (no separate acoustic model + vocoder needed beyond final waveform synthesis)

**Zero-Shot Voice Cloning:** Yes, via the speech-prompted encoder. Reference audio (~3s) conditions the entire generation.

**Streaming Support:** No.

**Parameters:** ~100–200M (estimated; not precisely reported)

**RTF:** Not formally benchmarked. Expected similar to Matcha-TTS.

**Audio Quality Metrics:**
- MOS ~4.0 on LJSpeech
- Competitive speaker similarity for zero-shot scenarios
- Outperformed Grad-TTS and VITS in naturalness at time of publication

**Training Data:** LJSpeech, LibriTTS, VCTK for multi-speaker experiments.

**License:** Research paper; no official open-source release.

---

### 1.5 Matcha-TTS

**Paper:** [arXiv:2309.03199](https://arxiv.org/abs/2309.03199) (September 2023, published ICASSP 2024)
**GitHub:** [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

**Architecture:**
- Non-autoregressive, **optimal-transport conditional flow matching (OT-CFM)** model
- Backbone: **1D U-Net** with Transformer blocks (combines convolutional efficiency with attention)
- Text encoder: Transformer-based, converts phonemes to hidden representations
- Duration predictor: Deterministic, predicts mel spectrogram frame durations from text encodings
- Generates mel spectrograms decoded via external vocoder (HiFi-GAN / BigVGAN / Vocos)

**Key Innovations:**
- **OT-CFM**: Uses optimal-transport paths for conditional flow matching, yielding straighter flow trajectories and faster convergence compared to standard CFM or diffusion
- **Memory-efficient**: Designed for edge/mobile deployment with small model footprint
- **Few-step inference**: High quality achievable in as few as 2–4 ODE steps (vs 50–1000 for diffusion models like Grad-TTS)
- Probabilistic model capable of generating diverse outputs for the same input text

**Zero-Shot Voice Cloning:** Limited in the base model. Multi-speaker version supports speaker embeddings. Not designed for zero-shot cloning from arbitrary reference audio (would need fine-tuning or speaker encoder).

**Streaming Support:** No native streaming. Non-autoregressive full-sequence generation. Chunk-based inference possible.

**Parameters:** ~18M (small variant) to ~100M depending on configuration. Very lightweight.

**Real-Time Factor (RTF):**
- RTF ~0.03–0.08 on A100 (with 2–4 ODE steps)
- ~10x faster than Grad-TTS
- Can run in real-time on CPU for small models
- Fastest open-source TTS at publication time for comparable quality

**Audio Quality Metrics:**
- **MOS:** 4.1–4.3 on LJSpeech (near ground-truth quality of ~4.5)
- **WER:** Competitive with Grad-TTS and VITS
- With only 4 ODE steps, matches quality that diffusion models need 50+ steps to achieve
- 2-step generation still produces intelligible, natural speech

**Training Data:** LJSpeech (24h, single speaker), VCTK (44h, 109 speakers), LibriTTS for multi-speaker.

**License:** MIT

---

### 1.6 Grad-TTS

**Paper:** [arXiv:2105.06337](https://arxiv.org/abs/2105.06337) (May 2021, ICML 2021)
**GitHub:** [huawei-noah/Speech-Backbones/Grad-TTS](https://github.com/huawei-noah/Speech-Backbones)

**Architecture:**
- Score-based diffusion model for TTS (one of the first to apply diffusion to speech synthesis)
- Encoder: Transformer-based text encoder with monotonic alignment search (MAS) for duration extraction
- Decoder: U-Net-based diffusion model that iteratively refines Gaussian noise into mel spectrograms
- Uses stochastic differential equations (SDEs) for the forward/reverse diffusion process
- External vocoder (HiFi-GAN) for waveform generation

**Key Innovations:**
- **First successful application of score-based diffusion to TTS**: Demonstrated that diffusion models could produce high-quality, natural-sounding speech
- **Monotonic Alignment Search (MAS)**: Unsupervised alignment algorithm that finds optimal monotonic mapping between text and mel frames during training — no external aligner needed
- **Controllable quality-speed tradeoff**: More diffusion steps = higher quality; fewer steps = faster inference. Quality degrades gracefully
- Foundation for subsequent work (Matcha-TTS, P-Flow, etc.)

**Zero-Shot Voice Cloning:** No. Single-speaker or multi-speaker with speaker IDs. Not designed for zero-shot scenarios.

**Streaming Support:** No.

**Parameters:** ~15–30M depending on configuration.

**Real-Time Factor:**
- RTF ~0.3–1.0 on GPU (with 10–50 diffusion steps)
- Significantly slower than Matcha-TTS at comparable quality
- Can achieve real-time with aggressive step reduction but quality suffers

**Audio Quality Metrics:**
- **MOS:** 4.1 on LJSpeech (with 50 steps)
- Competitive with Tacotron 2 and FastSpeech 2 at time of release
- WER competitive with autoregressive baselines

**Training Data:** LJSpeech (single speaker), VCTK (multi-speaker).

**License:** Apache 2.0 (Huawei Noah)

---

### 1.7 SpeechFlow

**Paper:** [arXiv:2307.08505](https://arxiv.org/abs/2307.08505) (July 2023)
**GitHub:** [microsoft/SpeechFlow](https://github.com/microsoft/SpeechFlow) (limited release)

**Architecture:**
- Pre-trained flow-matching generative model for speech
- Uses a masked audio prediction pre-training objective (similar to Voicebox)
- Backbone: Transformer with conditional flow matching
- Designed as a **foundation model** for multiple downstream speech tasks, not just TTS

**Key Innovations:**
- **Generalist speech model**: A single pre-trained model that can be fine-tuned for TTS, speech enhancement, speech separation, and other tasks
- **Self-supervised pre-training** on unlabeled speech data using flow matching
- Demonstrates transfer learning benefits: pre-trained SpeechFlow fine-tuned for TTS outperforms models trained from scratch
- Bridges speech understanding and generation in a unified framework

**Zero-Shot Voice Cloning:** Possible via fine-tuning with in-context learning setup.

**Streaming Support:** No.

**Parameters:** ~200M (estimated)

**RTF:** Not formally benchmarked independently.

**Audio Quality Metrics:**
- Fine-tuned for TTS: matches or exceeds Voicebox on speech quality metrics
- Speech enhancement: competitive with SpecFlow and VoiceFixer
- Specific numbers depend on downstream task and fine-tuning data

**Training Data:** 60K hours unlabeled speech (LibriLight) for pre-training. Task-specific data for fine-tuning.

**License:** Research; limited code release from Microsoft.

---

## 2. Neural Audio Codecs & LLM-Based TTS

### 2.1 VALL-E (Microsoft)

**Paper:** [arXiv:2301.02111](https://arxiv.org/abs/2301.02111) (January 2023)
**GitHub:** No official release; multiple community reimplementations exist (e.g., [lifeiteng/vall-e](https://github.com/lifeiteng/vall-e), [Plachtaa/VALL-E-X](https://github.com/Plachtaa/VALL-E-X))

**Architecture:**
- **Neural codec language model** — treats TTS as a conditional language modeling problem
- Uses **EnCodec** (Meta) as the audio tokenizer: 24kHz audio → 8 RVQ codebook levels at 75 Hz
- Two-stage decoder:
  1. **Autoregressive (AR) decoder**: Transformer that predicts the first (coarsest) codebook tokens sequentially, conditioned on phoneme input + 3-second reference audio enrollment
  2. **Non-autoregressive (NAR) decoder**: Transformer that predicts remaining 7 codebook levels in parallel, conditioned on the AR output
- Text input: Phoneme sequences from G2P conversion

**Key Innovations:**
- **First to frame TTS as language modeling over audio tokens**: Opened the paradigm of treating speech synthesis as next-token prediction over neural codec codes
- **In-context learning for zero-shot cloning**: Uses a 3-second enrollment utterance as prefix context — the model learns to continue in the same voice
- **Emergence of paralinguistic features**: Emotion, prosody, and speaking style transfer emerge naturally from the language modeling approach without explicit conditioning
- Demonstrated that scaling data (60K hours) and model size dramatically improves zero-shot cloning quality
- Spawned an entire lineage of codec language model approaches (VALL-E X, VALL-E 2, SpearTTS, SoundStorm, etc.)

**Zero-Shot Voice Cloning:** Yes, from 3 seconds of reference audio. Quality improves with longer references. Cross-lingual cloning demonstrated in VALL-E X variant.

**Streaming Support:** AR stage is inherently streamable (generates tokens left-to-right). NAR stage is not (parallel generation). Practical streaming requires architectural modifications.

**Parameters:** ~370M total (AR + NAR decoders)

**Real-Time Factor:** Not explicitly benchmarked. AR stage is sequential and relatively slow for long utterances. Community implementations report RTF ~0.5–1.5 on A100.

**Audio Quality Metrics:**
- **WER:** 5.9% on LibriSpeech continuation task (higher than GT due to AR token prediction errors)
- **Speaker Similarity (SIM-o):** 0.580
- **Robustness issues**: Known to produce word repetitions, omissions, and unstable outputs for longer utterances
- **MOS:** ~3.8 (below ground truth ~4.4)

**Training Data:** LibriLight (60K hours of English audiobook speech). Largest TTS training dataset at time of publication.

**License:** Research only; no official model release. Community reimplementations vary (MIT, Apache 2.0).

---

### 2.2 VALL-E 2 (Microsoft)

**Paper:** [arXiv:2406.05370](https://arxiv.org/abs/2406.05370) (June 2024)
**GitHub:** No official release

**Architecture:**
- Enhanced version of VALL-E with the same fundamental codec language model paradigm
- Same two-stage AR + NAR structure using EnCodec tokenization
- Adds two critical modifications to the decoding process:
  1. **Repetition Aware Sampling (RAS)**: Monitors token history during AR decoding; when a token repeats beyond a threshold, it is suppressed from the sampling distribution
  2. **Grouped Code Modeling**: Instead of predicting one codebook token per step, groups multiple tokens and predicts them together, reducing sequence length and improving coherence

**Key Innovations:**
- **Repetition Aware Sampling (RAS)**: Directly addresses the most critical failure mode of VALL-E (word/token repetition loops). Simple but highly effective — monitors recent token window and adjusts sampling probabilities
- **Grouped Code Modeling**: Reduces effective sequence length, improving both speed and long-range coherence
- **First TTS system to claim human parity**: On LibriSpeech and VCTK benchmarks, VALL-E 2 achieved MOS scores statistically indistinguishable from ground truth human recordings
- Maintains the zero-shot capability of VALL-E with dramatically improved robustness

**Zero-Shot Voice Cloning:** Yes, from 3 seconds of reference audio. Significantly more robust than VALL-E.

**Streaming Support:** Same as VALL-E — AR stage is streamable, NAR stage is not.

**Parameters:** Similar to VALL-E (~370M)

**Audio Quality Metrics:**
- **WER:** ~2.0% on LibriSpeech (vs 5.9% for VALL-E — massive improvement)
- **Speaker Similarity:** 0.64+ (improved over VALL-E's 0.58)
- **MOS:** 4.3–4.5 (claimed human parity)
- **Robustness:** Near-elimination of repetition and hallucination issues

**Training Data:** LibriLight (60K hours) — same as VALL-E.

**License:** Research only; no official model release.

---

### 2.3 EnCodec (Meta)

**Paper:** [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) (October 2022)
**GitHub:** [facebookresearch/encodec](https://github.com/facebookresearch/encodec)

**Architecture:**
- Neural audio compression codec based on **Residual Vector Quantization (RVQ)**
- Encoder-decoder architecture with a bottleneck of quantized latent codes:
  1. **Encoder**: Multi-scale 1D convolutional network downsamples audio waveform to latent features
  2. **RVQ Bottleneck**: Quantizes latent features using 8–32 codebook levels (each codebook has 1024 entries). Each level refines the residual from previous levels
  3. **Decoder**: Symmetric 1D convolutional network upsamples quantized codes back to audio waveform
- Supports multiple sample rates: 24kHz (speech) and 48kHz (music/general audio)
- Frame rate: 75 Hz at 24kHz (i.e., 75 frames per second, each frame = 8+ codebook indices)

**Key Innovations:**
- **Scalable bitrate via RVQ depth**: More codebook levels = higher bitrate = better quality. Supports 1.5 kbps to 24 kbps
- **Adversarial training**: Uses multi-scale STFT discriminator + multi-period discriminator for perceptual quality
- **Balancer mechanism**: Novel gradient balancing technique for stable multi-loss training
- **Language model for entropy coding**: Optional small Transformer LM compresses codebook indices further (~25–40% bitrate savings)
- Became the de facto audio tokenizer for codec language models (VALL-E, MusicGen, AudioGen, etc.)

**Bitrate/Quality Tradeoffs:**
| Codebooks | Bitrate (24kHz) | Quality Level |
|-----------|-----------------|---------------|
| 2         | 3.0 kbps        | Low (intelligible but degraded) |
| 4         | 6.0 kbps        | Medium (good for speech) |
| 8         | 12.0 kbps       | High (near-transparent speech) |
| 16        | 24.0 kbps       | Very high (music-quality) |

**Audio Quality Metrics:**
- At 6 kbps: MUSHRA ~75, competitive with Opus at 12 kbps
- At 12 kbps: MUSHRA ~82, approaching lossless quality
- ViSQOL scores: 3.5–4.2 depending on bitrate
- Reconstruction WER: ~2–3% at 12 kbps

**License:** MIT

---

### 2.4 SpeechTokenizer

**Paper:** [arXiv:2308.16692](https://arxiv.org/abs/2308.16692) (August 2023)
**GitHub:** [ZhangXInFD/SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)

**Architecture:**
- RVQ-based neural audio codec specifically designed for speech with **disentangled semantic and acoustic information**
- Encoder-decoder structure similar to EnCodec but with a crucial training modification
- Uses a **semantic teacher** (HuBERT) to guide the first RVQ level:
  - **Level 1 (L0)**: Trained to capture semantic/linguistic content via distillation from HuBERT features
  - **Levels 2–8 (L1–L7)**: Capture acoustic/paralinguistic details (timbre, prosody, recording conditions)
- Frame rate: 50 Hz at 16kHz

**Key Innovations:**
- **Semantic-acoustic disentanglement**: The first RVQ level explicitly encodes semantic content (what is being said), while subsequent levels encode acoustic details (how it sounds). This is critical for TTS models that need to separately control content and style
- **HuBERT distillation**: First codebook is trained with an additional loss term that encourages alignment with HuBERT semantic features
- **Unified tokenizer for speech LMs**: Eliminates the need for separate semantic tokens (HuBERT) and acoustic tokens (EnCodec) used in models like AudioLM
- Used as the tokenizer in Pheme and other systems

**Audio Quality Metrics:**
- Reconstruction quality comparable to EnCodec at equivalent bitrates
- Better semantic preservation: WER on reconstructed speech ~1–2% lower than EnCodec
- Speaker similarity preserved across reconstruction

**Training Data:** LibriSpeech (960h) + additional speech corpora.

**License:** Apache 2.0

---

### 2.5 SNAC (Multi-Scale Neural Audio Codec)

**Paper:** [arXiv:2410.02485](https://arxiv.org/abs/2410.02485) (October 2024)
**GitHub:** [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)

**Architecture:**
- **Multi-scale** residual vector quantization codec
- Unlike standard RVQ where all codebook levels operate at the same temporal resolution, SNAC uses different frame rates for different codebook levels:
  - Level 1: 12 Hz (coarsest, captures slow-varying features like content and speaker identity)
  - Level 2: 23 Hz (intermediate prosodic features)
  - Level 3: 47 Hz (finest, captures acoustic details and high-frequency content)
- Encoder: 1D convolutional network
- Decoder: 1D convolutional network with upsampling
- Total: 7 tokens per frame (1 + 2 + 4 across the three levels) when flattened
- Supports 24kHz and 44kHz audio

**Key Innovations:**
- **Multi-scale temporal resolution**: Different codebook levels operate at different time scales, better matching the natural hierarchy of speech features (slow content changes vs fast acoustic details)
- **Extremely low bitrate**: 0.98 kbps at 24kHz (vs EnCodec's minimum ~1.5 kbps) while maintaining intelligibility
- **Designed for LLM integration**: The flattened multi-scale token sequence is more efficient for autoregressive language models than standard RVQ (shorter effective sequence length)
- Used as the codec in Orpheus TTS and other recent LLM-based TTS systems

**Audio Quality Metrics:**
- Reconstruction WER: 2.25 (on speech benchmarks)
- Speaker similarity: 0.914–0.952
- Perceptual quality competitive with EnCodec at similar bitrates despite lower bitrate capability
- 24kHz model: excellent speech quality; 44kHz model: suitable for music

**Training Data:** Speech and music corpora (exact datasets not fully specified).

**License:** MIT

---

### 2.6 Descript Audio Codec (DAC)

**Paper:** [arXiv:2306.06546](https://arxiv.org/abs/2306.06546) (June 2023)
**GitHub:** [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)

**Architecture:**
- High-fidelity universal neural audio codec
- Encoder-decoder with RVQ, similar to EnCodec but with several improvements:
  - **Snake activations**: Periodic activation function that better captures harmonic structure in audio
  - **Improved quantizer dropout**: During training, random codebook levels are dropped, making each level independently useful and improving quality at lower bitrates
  - **Multi-scale STFT discriminator** with improved architecture
- Frame rate: 86 Hz at 44.1kHz
- Codebook levels: 4–32 (configurable)

**Key Innovations:**
- **Universal codec**: A single model handles speech, music, and environmental audio at 44.1kHz (vs EnCodec's separate speech/music models)
- **Snake activations**: Periodic inductive bias captures the harmonic structure of audio more effectively than standard activations (ReLU, GELU)
- **Quantizer dropout**: Key training technique that makes the codec more robust across different bitrate operating points
- **Higher fidelity than EnCodec**: Consistently outperforms EnCodec at matched bitrates on all audio types (speech, music, environmental)
- Used as the tokenizer in Dia TTS and other systems

**Audio Quality Metrics:**
- ViSQOL: 4.0+ at 8 kbps (vs EnCodec ~3.5)
- MUSHRA: 85+ at 8 kbps for speech
- Transparent quality for speech at ~6 kbps
- Music quality competitive with Opus at 2x the bitrate

**Supported Configurations:**
- 16kHz, 24kHz, 44.1kHz sample rates
- 1.5–16 kbps bitrate range

**Training Data:** Diverse audio including speech (DNS Challenge, DAPS, VCTK, Common Voice), music (MUSDB), and general audio (AudioSet, FSD50K).

**License:** MIT

---

### 2.7 WavTokenizer

**Paper:** [arXiv:2408.16532](https://arxiv.org/abs/2408.16532) (August 2024)
**GitHub:** [jishengpeng/WavTokenizer](https://github.com/jishengpeng/WavTokenizer)

**Architecture:**
- Single-codebook neural audio codec achieving extreme compression
- Encoder-decoder architecture with a **single VQ codebook** (no residual quantization)
- Uses an expanded VQ space (4096 entries) and improved training:
  - **Inverse Fourier Transform decoder**: Directly predicts complex STFT coefficients, then uses ISTFT for waveform reconstruction
  - **Multi-scale discriminator** for adversarial training
  - **Semantic distillation** from WavLM features to enrich the single codebook with linguistic information
- Frame rate: 40 Hz (low) or 75 Hz at 24kHz

**Key Innovations:**
- **Single codebook efficiency**: Achieves quality comparable to multi-codebook codecs (EnCodec with 4 levels) using just one codebook — dramatically simplifying downstream LM modeling (1 token per frame instead of 4–8)
- **Extreme compression**: ~0.6 kbps at 40 Hz with a single codebook
- **Semantic-acoustic fusion**: Single codebook captures both semantic and acoustic information via WavLM distillation
- **IFT decoder**: Frequency-domain decoding approach avoids time-domain aliasing artifacts

**Audio Quality Metrics:**
- At 40 Hz single codebook: WER comparable to EnCodec at 4 codebooks
- UTMOS scores: 3.8+ (competitive with multi-codebook systems)
- Speaker similarity maintained across compression/decompression

**Training Data:** LibriTTS, VCTK, and additional speech data.

**License:** MIT

---

### 2.8 Mimi (Kyutai)

**Paper:** Part of the Moshi project — [arXiv:2410.00037](https://arxiv.org/abs/2410.00037) (September 2024)
**GitHub:** [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)

**Architecture:**
- Streaming neural audio codec designed for real-time conversational AI
- Based on the SEANet encoder-decoder (similar to EnCodec) with key modifications:
  - **Semantic distillation**: First codebook level trained to align with WavLM semantic features
  - **Extremely low frame rate**: 12.5 Hz (80ms per frame) — among the lowest for speech codecs
  - **RVQ**: 8 codebook levels, 2048 entries each
  - **Causal convolutions**: Fully causal encoder and decoder for streaming compatibility
- Supports 24kHz audio

**Key Innovations:**
- **Ultra-low frame rate (12.5 Hz)**: Only 12.5 tokens per second (vs 75 for EnCodec, 50 for SpeechTokenizer). Dramatically reduces sequence length for LLM processing
- **Streaming-native**: Causal architecture means no lookahead — can encode and decode frame-by-frame in real time
- **Designed for Moshi**: Specifically optimized for the Moshi conversational AI system, which requires extremely efficient audio tokenization for real-time full-duplex dialogue
- **Semantic-acoustic disentanglement**: Similar to SpeechTokenizer, first codebook captures semantics

**Audio Quality Metrics:**
- Reconstruction quality optimized for speech (not general audio)
- WER on reconstructed speech: competitive at its bitrate
- Some quality tradeoff vs higher-frame-rate codecs due to extreme compression
- Used in Dia2 for streaming TTS

**Training Data:** Fisher corpus + additional conversational speech data.

**License:** Apache 2.0

---

### 2.9 AudioDec

**Paper:** [arXiv:2305.02765](https://arxiv.org/abs/2305.02765) (May 2023)
**GitHub:** [facebookresearch/AudioDec](https://github.com/facebookresearch/AudioDec)

**Architecture:**
- Two-stage neural audio codec optimized for high-fidelity speech:
  1. **Stage 1 (Encoder + VQ)**: Convolutional encoder + group-residual vector quantization (GRVQ). GRVQ splits the latent into groups and applies RVQ within each group
  2. **Stage 2 (Vocoder enhancement)**: A separate HiFi-GAN-style vocoder is trained to enhance the Stage 1 output, recovering fine details lost in quantization
- Supports 48kHz for highest quality

**Key Innovations:**
- **Group-RVQ (GRVQ)**: Novel quantization scheme that reduces codebook underutilization by splitting latent dimensions into groups
- **Two-stage enhancement**: Dedicated vocoder stage recovers quality lost in quantization, achieving near-transparent quality
- **48kHz support**: One of the first neural codecs optimized for high-sample-rate speech
- **Projector-based speaker adaptation**: Can be fine-tuned for specific speakers with minimal data

**Audio Quality Metrics:**
- PESQ: 3.5+ at moderate bitrates
- MUSHRA: competitive with SoundStream and EnCodec
- Near-transparent quality for narrowband speech at ~6 kbps

**Training Data:** VCTK (48kHz multi-speaker), internal datasets.

**License:** MIT

---

### 2.10 FunCodec

**Paper:** [arXiv:2309.07405](https://arxiv.org/abs/2309.07405) (September 2023)
**GitHub:** [modelscope/FunCodec](https://github.com/modelscope/FunCodec)

**Architecture:**
- Frequency-domain neural audio codec from Alibaba/DAMO Academy
- Operates on **complex spectrogram** representation rather than raw waveform:
  - Encoder: 2D convolutional network processes STFT magnitude and phase
  - VQ: Standard RVQ in the spectrogram domain
  - Decoder: 2D convolutional network reconstructs complex spectrogram → ISTFT → waveform
- Also supports time-domain variants for comparison
- Part of the FunASR/ModelScope ecosystem

**Key Innovations:**
- **Frequency-domain operation**: Processing in STFT domain provides natural frequency decomposition and can be more efficient than time-domain approaches
- **Flexible codec framework**: Supports both time-domain and frequency-domain modes, multiple quantization schemes, configurable bitrates
- **Integrated with speech processing ecosystem**: Designed to work seamlessly with FunASR for speech recognition, speech enhancement, and other tasks
- **Multi-task capability**: Single codec framework for speech, music, and sound effects

**Audio Quality Metrics:**
- Competitive with EnCodec and SoundStream at equivalent bitrates
- Frequency-domain mode shows advantages for speech at lower bitrates
- Specific benchmarks available in the paper

**Training Data:** Multilingual speech data (English + Chinese focus).

**License:** Apache 2.0 (ModelScope)

---

## 3. Streaming & Efficient TTS

### 3.1 Orpheus TTS (Canopy Labs)

**Paper:** No formal paper; technical blog posts and model cards
**GitHub:** [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
**HuggingFace:** [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)

**Architecture:**
- LLM-based TTS built on **Llama-3B** (decoder-only transformer) fine-tuned for speech token prediction
- Audio tokenization: **SNAC** codec at 24kHz with 3 RVQ levels (12/23/47 Hz token rates) producing 7 tokens per frame in a flattened sequence
- Text → LLM → SNAC tokens → CNN detokenizer → audio waveform
- A **CNN-based sliding-window detokenizer** enables streaming audio output without popping artifacts at chunk boundaries
- Maximum sequence length: 8192 tokens

**Key Innovations:**
- **LLM-as-TTS**: Treats speech generation as a standard next-token prediction task on a pre-trained LLM. This means all LLM infrastructure (fine-tuning, KV-cache, speculative decoding, quantization) works out of the box
- **Emotion/intonation tags**: Supports inline tags like `<laugh>`, `<sigh>`, `<gasp>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>` that are injected directly into text to control expressiveness
- **SNAC multi-scale codec**: Ultra-low bitrate (0.98 kbps) codec with hierarchical temporal resolution
- **Streaming CNN detokenizer**: Sliding-window approach eliminates boundary artifacts during streaming playback
- **Model family**: Available in 3B (flagship), 1B, 400M, and 150M parameter variants

**Zero-Shot Voice Cloning:** Yes, via the pretrained base model. Best results with ~50 reference samples; high quality with 300+ samples. Can also be fine-tuned on specific voices.

**Streaming Support:** Yes, with ~200ms latency (first audio chunk). Reducible to:
- ~100ms with input streaming (start generating before full text is available)
- ~25–50ms with KV-cache optimization
- Requires ~91 tokens/sec throughput for real-time playback
- Production deployment: 16–25+ concurrent real-time streams per H100

**Parameters:**
- 3B (flagship)
- 1B
- 400M
- 150M

**Real-Time Factor:**
- < 1.0 on modern GPUs
- TTFB (time to first byte): ~100–130ms in production
- 16–25+ concurrent streams per H100

**Audio Quality Metrics:**
- 24kHz output
- SNAC reconstruction: WER 2.25, speaker similarity 0.914/0.952
- No published MOS
- Community consensus: among top open-source models alongside Kokoro, CSM-1B, F5-TTS

**Training Data:** 100K+ hours of permissive/non-copyrighted English speech. Billions of text tokens. Sequence length 8192.

**License:** Apache 2.0

---

### 3.2 Dia / Dia2 (Nari Labs)

**Paper:** No formal paper; technical blog and model documentation
**GitHub:** [nari-labs/dia](https://github.com/nari-labs/dia) / [nari-labs/dia2](https://github.com/nari-labs/dia2)
**HuggingFace:** [nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) / [nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B)

**Architecture:**
- **Dia v1 (1.6B):** Encoder-decoder transformer trained from scratch (not initialized from a pre-trained LLM)
  - Inspired by **SoundStorm** (parallel NAR decoding) and **Parakeet** (encoder-decoder structure)
  - Audio tokenization via **Descript Audio Codec (DAC)**
  - Single-pass generation of multi-speaker dialogue
- **Dia2 (2B):** Enhanced version with:
  - Streaming capability via **Kyutai Mimi codec** at 12.5 Hz frame rate
  - Conversational audio context conditioning
  - CUDA graph acceleration for faster inference

**Key Innovations:**
- **Multi-speaker dialogue generation**: Single model generates dialogue with distinct speakers using `[S1]`/`[S2]` tags — no separate speaker diarization or TTS pipeline needed
- **Rich nonverbal vocalizations**: 20+ emotion/action tags including `(laughs)`, `(sighs)`, `(applause)`, `(singing)`, `(gasps)`, etc.
- **Audio conditioning**: Can transfer emotion/tone from a reference audio clip
- **Dia2 streaming**: Starts generating audio before the full text input is available
- **Single-pass architecture**: Both speakers generated in a single forward pass, maintaining natural conversational timing and turn-taking

**Zero-Shot Voice Cloning:** Yes, via audio prompt conditioning. Default output produces varied random voices. Consistent voices via seed fixing or audio prompt conditioning.

**Streaming Support:**
- Dia v1: No (full-pass generation)
- Dia2: Yes, real-time streaming with CUDA graph acceleration

**Parameters:**
- Dia v1: 1.6B
- Dia2: 1B and 2B variants

**Real-Time Factor:**
- ~40 tokens/sec on A4000 (86 tokens = 1 sec audio), ~0.47 RTF
- ~2x real-time on RTX 4090
- ~10GB VRAM required

**Audio Quality Metrics:**
- No formal published benchmarks (WER/MOS)
- Community comparisons claim parity with ElevenLabs and superiority over Sesame CSM-1B in naturalness for dialogue scenarios

**Training Data:** Not publicly disclosed. Developed with Google TPU Research Cloud and HuggingFace ZeroGPU grants.

**License:** Apache 2.0

---

### 3.3 OuteTTS (OuteAI)

**Paper:** No formal paper; technical documentation
**GitHub:** [edwko/OuteTTS](https://github.com/edwko/OuteTTS)
**HuggingFace:** [OuteAI/OuteTTS-1.0-0.6B](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B)

**Architecture:**
- **Pure LLM approach** — extends standard language models with TTS capability via zero architectural modifications
- Supported base LLMs: LLaMa, Qwen, OLMo (different versions use different bases)
- Three-step pipeline:
  1. **WavTokenizer** (or DAC in v1.0) encodes audio at 75 tokens/sec
  2. **CTC forced alignment** maps words to audio token positions
  3. **Structured prompts** `[transcription][word][duration][audio tokens]` train the LLM to predict speech as next-token generation
- Compatible with **llama.cpp / GGUF / EXL2** for on-device inference

**Key Innovations:**
- **Zero architectural modification**: The base LLM is used as-is with vocabulary extension. No adapters, no new layers — just fine-tuning on structured speech prompts
- **On-device capability**: Full compatibility with llama.cpp quantization ecosystem (GGUF, EXL2) enables running on phones, laptops, and edge devices
- **Multi-language support**: en, ja, ko, zh, fr, de (v0.3+)
- **Smart text chunking**: Automatic chunking for long-form generation

**Version History:**
| Version | Base Model | Params | Codec | Year |
|---------|-----------|--------|-------|------|
| v0.1    | LLaMa     | 350M   | WavTokenizer | 2024 |
| v0.2    | Qwen-2.5-0.5B | 500M | WavTokenizer | 2024 |
| v0.3    | Qwen-2.5 / OLMo | 500M/1B | WavTokenizer | 2024 |
| v1.0    | Qwen3-0.6B | 0.6B | DAC | 2025 |

**Zero-Shot Voice Cloning:** Yes, using ~10 seconds of reference audio to create a speaker profile.

**Streaming Support:** No dedicated streaming mode. Supports chunked/batched generation via llama.cpp server backends. Smart text chunking for long-form generation.

**Parameters:** 350M to 1B depending on version.

**RTF:** Not officially published. llama.cpp backend significantly faster than Transformers. Described as suitable for real-time on-device.

**Audio Quality Metrics:**
- Limited by small model size; may alter/insert/omit words
- Best with <30s generation windows
- Quality improved substantially from v0.1 to v1.0

**Training Data:** Emilia (CC-BY-NC), Common Voice (CC-0), People's Speech (CC-BY), MLS (CC-BY), VCTK (CC-BY), others. v0.2 trained on 5B+ audio prompt tokens.

**License:** Apache 2.0 (v1.0)

---

### 3.4 MARS5-TTS (Camb.ai)

**Paper:** No formal paper; architecture documentation on GitHub
**GitHub:** [Camb-ai/MARS5-TTS](https://github.com/Camb-ai/MARS5-TTS)
**HuggingFace:** [CAMB-AI/MARS5-TTS](https://huggingface.co/CAMB-AI/MARS5-TTS)

**Architecture:**
- Two-stage pipeline:
  1. **AR Stage (750M)**: Mistral-style decoder-only transformer predicts **EnCodec L0** (coarsest codebook) tokens via next-token prediction with cross-entropy loss. A small encoder-only transformer produces an implicit speaker embedding from reference audio
  2. **NAR Stage (450M)**: Encoder-decoder transformer using **multinomial DDPM** (discrete diffusion) to predict remaining 7 EnCodec codebook levels with cosine diffusion schedule
- Final vocoding via EnCodec decoder at 24kHz
- Text input: BPE-tokenized raw text (no phonemizer)

**Key Innovations:**
- **Multinomial diffusion for codebook refinement**: Novel NAR component using discrete diffusion instead of parallel prediction or iterative refinement
- **Implicit speaker embedding**: Small encoder-only transformer extracts speaker characteristics from raw audio — no external speaker verification model needed
- **Prosody control via text**: Punctuation and capitalization in BPE tokens influence prosody (e.g., "AMAZING!" produces emphatic speech)
- **Dual inference modes**:
  - **Shallow clone**: Only reference audio needed (no transcript) — faster but lower quality
  - **Deep clone**: Reference audio + transcript — higher fidelity voice reproduction
- **RePaint-style inpainting**: Optional diffusion inpainting step can improve quality at cost of speed

**Zero-Shot Voice Cloning:** Yes, from 2–12 seconds of reference audio (optimal ~6s). Deep clone mode yields better quality when transcript is available.

**Streaming Support:** No explicit streaming support documented.

**Parameters:** 1.2B total (750M AR + 450M NAR)

**RTF:** Not formally benchmarked. Requires >= 20GB VRAM.

**Audio Quality Metrics:**
- No published WER/MOS benchmarks
- Known for exceptional prosody in challenging scenarios (sports commentary, anime)
- Quality described as variable: "can produce really great results" but not yet fully consistent
- RePaint inpainting available to improve quality at cost of speed

**Training Data:** 150K+ hours of speech data. Trained on raw audio with BPE text.

**License:** GNU AGPL 3.0 (alternative licensing available on request)

---

### 3.5 MetaVoice-1B

**Paper:** No formal paper; technical documentation
**GitHub:** [metavoiceio/metavoice-src](https://github.com/metavoiceio/metavoice-src)
**HuggingFace:** [metavoiceio/metavoice-1B-v0.1](https://huggingface.co/metavoiceio/metavoice-1B-v0.1)

**Architecture:**
- Four-stage pipeline:
  1. **Causal GPT (1.2B params)**: Predicts first 2 EnCodec codebook hierarchies in flattened interleaved order. Conditioned on speaker embeddings from a separate speaker verification network. Uses custom 512-token BPE text tokenizer. Condition-free sampling for voice cloning
  2. **Non-causal (bidirectional) transformer**: Predicts remaining EnCodec codebook hierarchies from the first 2
  3. **Multi-band diffusion**: Converts EnCodec tokens to waveforms (clearer than standard RVQ/VOCOS decoder but introduces some artifacts)
  4. **DeepFilterNet post-processing**: Removes diffusion artifacts from the generated audio

**Key Innovations:**
- **Skips semantic tokens**: Found that intermediate semantic token prediction (used in AudioLM, SpeechX) is unnecessary — directly predicts acoustic tokens
- **Flattened interleaved codebook prediction**: Instead of predicting all tokens for one timestep then moving to the next, interleaves codebook levels within the AR sequence
- **Multi-band diffusion + DeepFilterNet**: Two-stage audio reconstruction pipeline for high-fidelity waveform generation
- **Speaker conditioning via separate network**: Uses a speaker verification network (not in-context learning) for voice cloning
- **Emotion focus**: Prioritizes emotional speech rhythm and tone over raw intelligibility

**Zero-Shot Voice Cloning:** Yes, for American and British English with 30 seconds of reference audio. Cross-lingual cloning possible with fine-tuning. 1 minute sufficient for Indian English speakers.

**Streaming Support:** Architecture supports streaming token prediction. Uses torch.compile + Flash Decoding + KV-caching for optimized inference.

**Parameters:** 1.2B (base GPT model)

**RTF:** < 1.0 on Ampere/Ada-Lovelace/Hopper GPUs after torch.compile. int4 quantization ~2x faster than bf16. Minimum 12GB VRAM recommended.

**Audio Quality Metrics:**
- No published MOS/WER benchmarks
- Prioritizes emotional speech rhythm and tone
- Quality degrades with longer sequences (practical limit below 2048 tokens)

**Training Data:** 100K hours of speech.

**License:** Apache 2.0

---

### 3.6 Pheme (PolyAI)

**Paper:** [arXiv:2401.02839](https://arxiv.org/abs/2401.02839) (January 2024)
**GitHub:** [PolyAI-LDN/pheme](https://github.com/PolyAI-LDN/pheme)
**Project Page:** [polyai-ldn.github.io/pheme](https://polyai-ldn.github.io/pheme/)

**Architecture:**
- Two-component modular system:
  1. **Text-to-Semantics (T2S)**: T5-style encoder-decoder transformer converts text to semantic tokens
  2. **Acoustics-to-Speech (A2S)**: **SoundStorm-inspired** non-autoregressive transformer with **MaskGIT-style parallel decoding**, conditioned on T2S semantic tokens and speaker embeddings
- Uses **SpeechTokenizer** (50 Hz, 16kHz) to disentangle semantic and acoustic token streams via RVQ hierarchy

**Key Innovations:**
- **MaskGIT parallel decoding**: Provides **~15x speedup** over autoregressive baselines (MQTTS) with equal or better quality. Iteratively unmasks tokens from most confident to least confident across multiple passes
- **Extreme data efficiency**: Trained on **10x less data** than VALL-E/SoundStorm (550–10K hours vs 60K+) with competitive quality
- **Teacher-student distillation**: Larger models distill knowledge into smaller models. Single-speaker quality improves using only synthetic data from the teacher
- **Noisy data training**: Successfully trained on noisy/conversational data (podcasts, GigaSpeech) rather than requiring clean audiobook data
- **Modular architecture**: T2S and A2S components can be upgraded independently

**Zero-Shot Voice Cloning:** Yes, multi-speaker support with unseen voice generalization demonstrated on GigaSpeech test voices.

**Streaming Support:** Supports real-time synthesis via parallel decoding. A ~10 second utterance takes ~1.27 seconds to generate (300M model on A100).

**Parameters:** 100M (small) and 300M (large)

**Real-Time Factor:**
- RTF ~0.13 for 300M model on A100
- **14.5x faster than MQTTS** baseline
- Similar RTFs achievable on A10 GPU

**Audio Quality Metrics:**
- WER: Improves over MQTTS by ~2 percentage points
- MCD and FID (prosody diversity) measured
- Speaker similarity slightly below MQTTS but described as "production-ready"
- Significant WER errors from proper noun misspellings (known limitation)

**Training Data:**
- 100M model: 550 hours preprocessed GigaSpeech
- 300M model: 550h GigaSpeech + 585h LibriTTS + ~10Kh (25% of English MLS)
- All downsampled to 16kHz

**License:** CC-BY-4.0

---

## 4. Voice Cloning & Multi-Speaker TTS

### 4.1 XTTS v2 (Coqui)

**Paper:** [arXiv:2406.04904](https://arxiv.org/abs/2406.04904)
**GitHub:** [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
**HuggingFace:** [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

**Architecture:**
- GPT-2-style autoregressive transformer predicting VQ-VAE audio tokens
- Conditioning: **HuBERT-based speaker encoder** extracts speaker embeddings from reference audio
- VQ-VAE audio tokenizer (custom, not EnCodec)
- HiFi-GAN decoder for waveform generation from VQ tokens
- Text input: Character-level with language ID token

**Key Innovations:**
- **Massively multilingual**: Supports **17 languages** natively: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi
- **Fine-tuning capability**: Can be fine-tuned on custom voices with as little as 6 seconds of audio. Fine-tuning improves quality substantially
- **Production-ready**: Designed for real-world deployment with HTTP API server, streaming support, and model caching
- **Speaker conditioning**: HuBERT speaker encoder provides robust voice cloning even with noisy reference audio
- **Coqui legacy**: After Coqui AI shut down, the open-source community (especially via `coqui-tts` fork) continues maintenance

**Zero-Shot Voice Cloning:** Yes, from 3–30 seconds of reference audio. Quality improves with longer references. Cross-lingual cloning supported (clone an English voice for German output, etc.).

**Streaming Support:** Yes. Supports streaming inference via the built-in TTS server. Chunk-based generation with configurable chunk sizes.

**Parameters:** ~467M

**Real-Time Factor:**
- RTF ~0.3–0.5 on A100
- ~1–2x real-time on consumer GPUs (RTX 3090, 4090)
- CPU inference possible but slow (~5x real-time)

**Audio Quality Metrics:**
- MOS ~3.9–4.1 for native languages (English best)
- WER varies by language: ~3% English, ~5–8% for other languages
- Speaker similarity: 0.65–0.75 (good but below SOTA)
- Some languages (ar, ko, hi) have lower quality than European languages

**Training Data:** Proprietary multilingual dataset. Exact size undisclosed. Likely 10K–50K hours across all languages.

**License:** MPL 2.0 (Mozilla Public License)

**German Language Support:** Native German support. Quality rated as good-to-excellent for German TTS. Fine-tuning on German data further improves results. One of the best open-source options for German TTS.

---

### 4.2 OpenVoice v2 (MyShell)

**Paper:** [arXiv:2312.01479](https://arxiv.org/abs/2312.01479)
**GitHub:** [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)

**Architecture:**
- Two-stage pipeline:
  1. **Base TTS**: MeloTTS (VITS-based) generates speech in the target language with a default voice
  2. **Tone Color Converter**: Separate model that transfers the voice characteristics (tone color) from a reference speaker onto the base TTS output while preserving the content, emotion, and rhythm
- The tone color converter uses a VITS-style architecture with speaker embedding injection

**Key Innovations:**
- **Decoupled voice cloning**: Separates "what to say and how to say it" (base TTS) from "whose voice to use" (tone color converter). This decoupling means:
  - Base TTS improvements automatically improve cloning quality
  - Tone color transfer works across languages even if the reference audio is in a different language
  - Emotion and style can be controlled independently of voice identity
- **Zero-shot cross-lingual cloning**: Clone a voice from language A and synthesize in language B without any bilingual data
- **Lightweight**: The tone color converter is small and fast, adding minimal overhead
- **Granular control**: Separate control over emotion, accent, rhythm, pause, and intonation

**Zero-Shot Voice Cloning:** Yes. Reference audio required for the tone color converter. Works cross-lingually. Only a few seconds of reference audio needed.

**Streaming Support:** Depends on the base TTS model. MeloTTS base supports streaming.

**Parameters:** ~50M for tone color converter + base TTS model parameters.

**RTF:** Very fast — base TTS + tone color converter adds minimal overhead. Near real-time on consumer hardware.

**Audio Quality Metrics:**
- MOS ~3.7–4.0 (depends on base TTS quality)
- Cross-lingual quality slightly lower than same-language
- Tone color similarity: 0.7–0.8

**Training Data:** Multi-speaker, multilingual data (details in paper).

**License:** MIT (v2)

**Languages:** English, Chinese, Japanese, Korean, French, and more via MeloTTS base.

---

### 4.3 Kokoro

**Paper:** No formal paper
**GitHub:** [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
**HuggingFace:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

**Architecture:**
- Lightweight, **StyleTTS 2-based** architecture
- Uses a style/voice embedding approach with a relatively small model footprint
- Text-to-mel spectrogram generation with external vocoder
- Multiple voice presets available (54+ voices in v1.0)
- Versions: v0.19 (82M), v1.0 (200M+)

**Key Innovations:**
- **Extreme efficiency**: 82M parameters (v0.19) achieving quality competitive with models 10–50x its size
- **Broad language support**: English, Chinese, Japanese, Korean, French, German, Italian, Portuguese, Spanish, Hindi, and more (v1.0 added ~8 languages)
- **Production-ready**: Multiple deployment options including ONNX, WebGPU, llama.cpp, MLX (Apple Silicon), Kokoro.js (browser-based)
- **Community favorite**: One of the most popular open-source TTS models in the community
- **Speed**: Generates speech extremely fast due to small model size — faster than real-time on CPU

**Zero-Shot Voice Cloning:** Limited in the base model. Supports pre-defined voice presets. Community fine-tuning tools available for custom voices. Not designed for arbitrary zero-shot cloning like XTTS or OpenVoice.

**Streaming Support:** Not natively streaming (generates full mel spectrogram). However, chunk-based inference and ONNX streaming pipelines exist in the community.

**Parameters:**
- v0.19: 82M
- v1.0: ~200M (estimated)

**Real-Time Factor:**
- Extremely fast: 5–15x real-time on CPU
- Near-instantaneous on GPU
- ONNX variant achieves ~50x real-time on modern CPUs

**Audio Quality Metrics:**
- Rated among top open-source models in community benchmarks
- Naturalness comparable to much larger models
- Clear, expressive speech with good prosody
- Quality improved significantly from v0.19 to v1.0

**Training Data:** Not fully disclosed. Likely a mix of public and curated datasets.

**License:** Apache 2.0

**German Language Support:** Added in v1.0. German voices available in the preset collection. Quality described as good.

---

### 4.4 Spark-TTS (SparkAudio)

**Paper:** [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) (March 2025)
**GitHub:** [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
**HuggingFace:** [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

**Architecture:**
- LLM-based TTS using **Qwen2.5-0.5B** as the language model backbone
- Novel **BiCodec** audio tokenizer that produces:
  - A single **global speaker embedding** vector (captures voice identity)
  - A sequence of **semantic tokens** (captures linguistic/prosodic content)
- The LLM predicts semantic tokens conditioned on text + speaker embedding
- Dedicated decoder converts semantic tokens + speaker embedding → waveform

**Key Innovations:**
- **BiCodec disentanglement**: Cleanly separates voice identity (global embedding) from content (sequential tokens). This enables:
  - Natural voice cloning by swapping the global embedding
  - Controllable voice creation by manipulating embedding dimensions
  - Better generalization to unseen speakers
- **Controllable voice generation**: Can create entirely new voices without reference audio by specifying attributes (gender, pitch range, speaking rate)
- **Efficient tokenization**: Single global embedding + semantic tokens = much shorter sequences than full RVQ approaches
- **Qwen2.5 backbone**: Inherits strong language understanding and multilingual capability

**Zero-Shot Voice Cloning:** Yes. Extract BiCodec global embedding from reference audio → use as conditioning for generation. Clean separation of identity and content.

**Streaming Support:** Not explicitly documented, but LLM backbone supports streaming token generation.

**Parameters:** 0.5B (Qwen2.5 backbone)

**RTF:** Not formally benchmarked. Expected fast due to efficient tokenization.

**Audio Quality Metrics:**
- Competitive with VALL-E and CosyVoice on benchmarks
- Good speaker similarity due to clean identity-content separation
- Naturalness: high marks in early evaluations

**Training Data:** Chinese and English speech data. Exact corpus not fully disclosed.

**License:** Apache 2.0

---

### 4.5 CSM (Conversational Speech Model, Sesame)

**Paper:** No formal paper; technical blog
**GitHub:** [SesameAILabs/csm](https://github.com/SesameAILabs/csm)
**HuggingFace:** [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)

**Architecture:**
- **Two-backbone** transformer architecture:
  1. **Backbone 1 (Llama-based)**: Processes multi-turn conversational context. Takes interleaved text and audio tokens from previous conversation turns and generates a context embedding
  2. **Backbone 2 (smaller Llama)**: Conditioned on Backbone 1's context embedding, generates **Mimi codec** audio tokens for the current utterance
- Audio tokenization: **Mimi** (Kyutai) at 12.5 Hz, producing 32 codebook levels
- Generates audio autoregressively at the frame level

**Key Innovations:**
- **Conversational context modeling**: First open-source TTS model specifically designed for multi-turn conversation. Backbone 1 processes the entire conversation history (both user and assistant turns) to produce contextually appropriate speech
- **Full-duplex potential**: Architecture can process interleaved speaker audio, enabling future full-duplex conversational AI
- **Mimi codec at 12.5 Hz**: Ultra-low frame rate means very short sequences even for long utterances
- **Watermarking**: Includes SilentCipher audio watermarking for generated speech detection
- **Emotional intelligence**: Generates speech with natural conversational prosody including hesitations, emphasis, and turn-taking cues

**Zero-Shot Voice Cloning:** Yes, through audio context conditioning. Previous conversation turns establish the voice. No explicit voice cloning mode — the model naturally maintains consistency from conversation context.

**Streaming Support:** AR generation supports streaming. Mimi codec is streaming-compatible.

**Parameters:** 1B (total across both backbones)

**RTF:** Not formally benchmarked. 12.5 Hz frame rate means fewer generation steps than higher-frame-rate models.

**Audio Quality Metrics:**
- No published formal benchmarks
- Community comparisons: high naturalness for conversational speech
- Some reports of lower quality compared to Dia for dialogue generation
- Strong prosodic variation and emotional expressiveness

**Training Data:** Not disclosed.

**License:** Apache 2.0

---

### 4.6 GPT-SoVITS

**Paper:** No formal paper
**GitHub:** [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

**Architecture:**
- Two-stage pipeline combining **GPT** (autoregressive) and **SoVITS** (VITS-based synthesis):
  1. **GPT Stage**: AR transformer predicts semantic tokens (HuBERT) from text, conditioned on reference audio. Similar to VALL-E's AR stage but with semantic tokens instead of acoustic tokens
  2. **SoVITS Stage**: Modified **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) decoder generates waveform from semantic tokens + text + speaker features
- HuBERT for semantic token extraction
- Multiple reference audio support for better voice capture

**Key Innovations:**
- **Few-shot fine-tuning**: Designed to produce high-quality voice clones with very little data:
  - 5 seconds: Basic cloning with GPT-SoVITS inference
  - 1 minute: Good quality fine-tuning possible
  - 5+ minutes: Excellent quality with full fine-tuning
- **Integrated training pipeline**: Complete GUI-based training pipeline included — dataset preparation, preprocessing, training, and inference all in one package
- **Extremely popular in Asia**: One of the most-used open-source TTS tools in Chinese-speaking communities, widely used for voice acting, dubbing, and content creation
- **Cross-lingual**: Supports Chinese, English, Japanese, Korean, Cantonese
- **WebUI**: Full Gradio-based web interface for non-technical users

**Zero-Shot Voice Cloning:** Yes (basic, from 5 seconds). Much better with fine-tuning on 1–5+ minutes of data.

**Streaming Support:** Supports streaming inference in the v2 WebUI.

**Parameters:** ~200–300M (estimated across both stages)

**RTF:** Fast inference; near real-time on consumer GPUs.

**Audio Quality Metrics:**
- No published formal benchmarks
- Community consensus: excellent voice cloning quality, especially with fine-tuning
- Among the best for Chinese and Japanese voice cloning
- English quality lower than Chinese but still good

**Training Data:** User-provided data for fine-tuning. Base model trained on multi-speaker Chinese/English/Japanese data.

**License:** MIT

---

### 4.7 Fish Speech

**Paper:** No formal paper; technical blog
**GitHub:** [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
**HuggingFace:** [fishaudio/fish-speech-1.5](https://huggingface.co/fishaudio/fish-speech-1.5)

**Architecture:**
- **Dual AR** architecture:
  1. **Semantic AR (Llama-based)**: Predicts semantic tokens from text input, conditioned on reference audio tokens
  2. **Acoustic AR**: Predicts fine-grained acoustic tokens from semantic tokens
- Uses **Firefly-GAN** (custom neural codec) for audio tokenization
- Text input: Direct text processing (no phonemizer for most languages)
- v1.5 architecture significantly improved over earlier versions

**Key Innovations:**
- **Extensive multilingual support**: 13+ languages including English, Chinese, Japanese, Korean, French, German, Arabic, Spanish, Portuguese, Russian, Italian, Hindi, Vietnamese
- **Low VRAM requirement**: Can run inference with as little as 4GB VRAM
- **Fast inference**: Optimized for real-time generation
- **Firefly-GAN codec**: Custom codec designed specifically for Fish Speech's architecture
- **Fine-tuning support**: LoRA fine-tuning for custom voices with minimal data
- **API-first design**: Built-in HTTP API server for production deployment

**Zero-Shot Voice Cloning:** Yes, from ~10–30 seconds of reference audio. Quality improves with more reference data.

**Streaming Support:** Yes, supports streaming inference.

**Parameters:** ~500M–1B (varies by version)

**RTF:** < 1.0 on consumer GPUs. ~150 tokens/sec on RTX 4090.

**Audio Quality Metrics:**
- No published formal benchmarks
- Community: strong quality for Chinese and English
- Good prosody and naturalness
- Speaker similarity: good for fine-tuned voices

**Training Data:** 700K+ hours of multilingual audio data (v1.4 claim). Massive scaling of training data across versions.

**License:** Apache 2.0

---

## 5. Vocoders & Waveform Generators

### 5.1 HiFi-GAN

**Paper:** [arXiv:2010.05646](https://arxiv.org/abs/2010.05646) (NeurIPS 2020)
**GitHub:** [jik876/hifi-gan](https://github.com/jik876/hifi-gan)

**Architecture:**
- GAN-based vocoder (mel spectrogram → waveform)
- **Generator**: Uses transposed convolutions for upsampling with **Multi-Receptive Field Fusion (MRF)** blocks. Each MRF block contains multiple parallel residual branches with different kernel sizes and dilation rates, capturing patterns at multiple scales
- **Discriminator**: Two discriminator types:
  1. **Multi-Period Discriminator (MPD)**: Reshapes 1D audio into 2D with different periods (2, 3, 5, 7, 11) to capture periodic patterns at different frequencies
  2. **Multi-Scale Discriminator (MSD)**: Processes audio at multiple resolutions (1x, 2x, 4x downsampled)

**Key Innovations:**
- **MRF blocks**: Multi-scale residual connections in the generator capture both local and global audio patterns
- **MPD**: Novel discriminator design that exploits the periodic nature of speech by reshaping the signal
- **Speed**: Orders of magnitude faster than autoregressive vocoders (WaveNet, WaveRNN) while achieving comparable quality
- **Foundation vocoder**: Became the default vocoder for most TTS systems (Tacotron 2, FastSpeech 2, Grad-TTS, VITS, etc.)

**Configurations:**
| Model | Params | Speed (A100) | Quality |
|-------|--------|-------------|---------|
| V1    | 14M    | ~890x RT    | Highest |
| V2    | 0.9M   | ~2000x RT   | Good    |
| V3    | 1.5M   | ~1500x RT   | Better  |

**Audio Quality Metrics:**
- MOS: 4.36 (V1, LJSpeech) — near ground truth (4.45)
- PESQ: 3.5+
- Mel-cepstral distortion: competitive with WaveNet

**License:** MIT

---

### 5.2 BigVGAN / BigVGAN-v2 (NVIDIA)

**Paper:** [arXiv:2206.04658](https://arxiv.org/abs/2206.04658) (BigVGAN, 2022), [arXiv:2309.02836](https://arxiv.org/abs/2309.02836) (BigVGAN-v2, 2023)
**GitHub:** [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)

**Architecture:**
- Scaled-up GAN vocoder based on HiFi-GAN with key improvements:
- **Generator**: Replaces standard activations with **anti-aliased multi-periodicity composition (AMP)** modules:
  - **Snake activation**: Periodic activation function `x + (1/a) * sin²(ax)` that naturally captures harmonic structure
  - **Anti-aliasing**: Low-pass filters after each upsampling layer to prevent aliasing artifacts
- **Discriminator**: Multi-resolution spectrogram discriminator (MRSD) instead of MPD+MSD
- BigVGAN-v2 further scales to 112M parameters and 44kHz support

**Key Innovations:**
- **Snake activations**: Periodic inductive bias that captures the harmonic structure of audio signals much better than ReLU/LeakyReLU. Foundation for quality improvement
- **Anti-aliased upsampling**: Prevents frequency aliasing artifacts that plague standard transposed convolution upsampling
- **Universal vocoder**: Single model handles speech, music, and environmental audio (unlike HiFi-GAN which is typically speech-specific)
- **Scaling benefits**: Larger models (112M) show clear quality improvements over smaller ones (14M HiFi-GAN)
- **Zero-shot generalization**: Trained on LibriTTS but generalizes to unseen speakers, languages, and even non-speech audio

**BigVGAN-v2 Specs:**
- 112M parameters
- 24kHz and 44kHz support
- 100-band mel spectrogram input
- Trained on large-scale diverse audio (not just speech)

**Audio Quality Metrics:**
- MOS: 4.4+ (approaching ground truth)
- M-STFT distance: 0.78 (vs 0.88 for HiFi-GAN)
- Significantly better generalization than HiFi-GAN on out-of-domain audio
- Universal quality across speech/music/environmental audio

**RTF:** ~300–500x real-time on A100 (fast enough that it's never the bottleneck).

**License:** MIT

---

### 5.3 Vocos

**Paper:** [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) (June 2023)
**GitHub:** [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)

**Architecture:**
- **Frequency-domain vocoder** — operates entirely in STFT domain rather than generating time-domain waveforms:
  1. Input: Mel spectrogram (or codec features)
  2. **Backbone**: ConvNeXt V2 blocks process the mel spectrogram
  3. **ISTFT Head**: Predicts complex STFT coefficients (magnitude and phase) → ISTFT → waveform
- No transposed convolution upsampling — avoids the aliasing issues that plague time-domain vocoders

**Key Innovations:**
- **ISTFT-based decoding**: By predicting frequency-domain coefficients and using ISTFT for waveform reconstruction, Vocos avoids time-domain artifacts (checkerboard artifacts, metallic ringing) common in GAN vocoders
- **ConvNeXt V2 backbone**: Modern convolutional architecture provides excellent feature extraction
- **Codec compatibility**: Can decode features from EnCodec, SNAC, and other neural codecs directly (not just mel spectrograms)
- **Extremely fast**: No iterative upsampling means faster inference than HiFi-GAN
- **Used in F5-TTS**: Recommended vocoder for F5-TTS and other flow-matching systems

**Audio Quality Metrics:**
- MOS: 4.3+ (competitive with BigVGAN)
- PESQ: 3.6
- Cleaner output than HiFi-GAN with fewer artifacts
- Particularly good for high-frequency content

**RTF:** ~1000x+ real-time on GPU. Essentially instantaneous.

**Parameters:** ~13M

**License:** MIT

---

### 5.4 WaveGrad

**Paper:** [arXiv:2009.00713](https://arxiv.org/abs/2009.00713) (September 2020)
**GitHub:** [ivanvovk/WaveGrad](https://github.com/ivanvovk/WaveGrad)

**Architecture:**
- **Diffusion-based vocoder** (one of the first)
- Iteratively refines Gaussian noise into a waveform conditioned on mel spectrogram
- U-Net-like architecture with:
  - Downsampling blocks (dilated convolutions)
  - Upsampling blocks (transposed convolutions)
  - FiLM (Feature-wise Linear Modulation) conditioning on mel spectrogram and noise level
- Continuous noise schedule (not discrete steps like DDPM)

**Key Innovations:**
- **First successful diffusion vocoder**: Demonstrated that diffusion models could generate high-quality audio waveforms
- **Continuous noise schedule**: Uses a continuous noise level parameter instead of discrete timesteps, enabling flexible inference-time tradeoffs
- **Fewer steps than DDPM**: Achieves good quality with 6–50 iterations (vs 1000+ for standard DDPM)
- **FiLM conditioning**: Efficient conditioning mechanism for injecting mel spectrogram information at each diffusion step

**Audio Quality Metrics:**
- MOS: 4.35 (with 50 steps, LJSpeech) — competitive with HiFi-GAN
- Quality degrades gracefully: 6 steps still produces intelligible speech (MOS ~3.8)
- 1000 steps approaches ground truth quality

**RTF:** ~0.7 (50 steps, V100) — slower than GAN vocoders but faster than WaveNet.

**Parameters:** ~15M

**License:** Apache 2.0

---

### 5.5 UnivNet

**Paper:** [arXiv:2106.07889](https://arxiv.org/abs/2106.07889) (June 2021, Interspeech 2022)
**GitHub:** [mindslab-ai/univnet](https://github.com/mindslab-ai/univnet)

**Architecture:**
- GAN-based vocoder with **multi-resolution spectrogram input**
- **Generator**: Instead of a single mel spectrogram, takes **multi-resolution spectrograms** (computed at different FFT sizes) as input. Uses **Gated Activation Units (GAUs)** and location-variable convolutions (LVC) for adaptive processing
- **Discriminator**: Multi-resolution spectrogram discriminator (MRSD) evaluates generated audio at multiple frequency resolutions

**Key Innovations:**
- **Multi-resolution spectrogram input**: Multiple STFT representations (different window sizes) provide the generator with both fine-grained and coarse frequency information, improving both high-frequency and low-frequency reproduction
- **Location-Variable Convolution (LVC)**: Convolution kernels that vary based on position — different parts of the spectrogram get different processing, adapting to local frequency content
- **Gated Activation Units**: Gating mechanism controls information flow, improving quality over standard residual blocks
- **Efficient**: Achieves quality close to WaveNet at speeds close to HiFi-GAN

**Audio Quality Metrics:**
- MOS: 4.30 (LJSpeech)
- Outperforms HiFi-GAN in some evaluations, especially for high-frequency content
- Better generalization than HiFi-GAN to unseen speakers

**RTF:** ~600x real-time on V100

**Parameters:** ~17M

**License:** BSD 3-Clause

---

## 6. German Language TTS Support

### Overview of German-Capable Open-Source TTS

| Model | German Support | Quality | Notes |
|-------|---------------|---------|-------|
| **Kokoro v1.0** | Native | Good | German voices in preset collection, 200M params, Apache 2.0 |
| **XTTS v2** | Native | Good-Excellent | One of best open-source options for German, zero-shot cloning, MPL 2.0 |
| **Fish Speech** | Native | Good | German among 13+ languages, Apache 2.0 |
| **OpenVoice v2** | Via MeloTTS base | Moderate-Good | Cross-lingual cloning enables German output from any reference voice |
| **F5-TTS** | Community fine-tunes | Moderate-Good | German fine-tunes available on HuggingFace |
| **OuteTTS v0.3+** | Listed | Moderate | German listed as supported language |
| **Piper** | Native | Good | Lightweight TTS (VITS-based) with multiple German voice models |
| **GPT-SoVITS** | Not native | Limited | Primarily Chinese/English/Japanese/Korean; German requires custom training |
| **Bark (Suno)** | Native | Moderate | Supports German among 13 languages; quality variable |

### German-Specific Notes:

**Best options for German TTS (ranked):**
1. **XTTS v2**: Best overall for German. Native multilingual support with zero-shot cloning. Fine-tunable for custom German voices. Production-ready.
2. **Kokoro v1.0**: Best efficiency-to-quality ratio. Very fast, lightweight, good German voices.
3. **Fish Speech v1.5**: Strong multilingual model with German support. Good quality, streamable.
4. **Piper**: Best for on-device/embedded. Multiple high-quality German voices (thorsten, kerstin, etc.). VITS-based, extremely fast on CPU. Widely deployed in Home Assistant and other IoT applications.
5. **F5-TTS (community fine-tune)**: High quality when fine-tuned on German data. Good for custom voice cloning.

**Piper German Voices (Notable):**
- `de_DE-thorsten-high` — Male voice, high quality, trained on Thorsten Müller's donated recordings
- `de_DE-kerstin-low` — Female voice
- `de_DE-eva_k-x_low` — Female voice, extra quality
- Available via [rhasspy/piper](https://github.com/rhasspy/piper) — VITS architecture, ONNX runtime, runs on Raspberry Pi

---

## 7. Comparative Summary Tables

### Table 1: Model Overview

| Model | Year | Params | Architecture | Zero-Shot | Streaming | License |
|-------|------|--------|-------------|-----------|-----------|---------|
| F5-TTS | 2024 | 335M | Flow-matching DiT | Yes | No (chunked) | CC-BY-NC-4.0 |
| E2 TTS | 2024 | 335M | Flow-matching UNet | Yes | No | Research only |
| Voicebox | 2023 | 330M | Flow-matching Transformer | Yes | No | Research only |
| Matcha-TTS | 2024 | 18–100M | OT-CFM UNet | Limited | No | MIT |
| Grad-TTS | 2021 | 15–30M | Diffusion UNet | No | No | Apache 2.0 |
| VALL-E | 2023 | 370M | AR + NAR Codec LM | Yes | Partial | Research only |
| VALL-E 2 | 2024 | 370M | AR + NAR Codec LM | Yes | Partial | Research only |
| Orpheus | 2025 | 150M–3B | Llama + SNAC | Yes | Yes | Apache 2.0 |
| Dia/Dia2 | 2025 | 1.6–2B | Enc-Dec Transformer | Yes | Yes (Dia2) | Apache 2.0 |
| OuteTTS | 2024-25 | 350M–1B | Pure LLM | Yes | No | Apache 2.0 |
| MARS5 | 2024 | 1.2B | AR + NAR Diffusion | Yes | No | AGPL 3.0 |
| MetaVoice | 2024 | 1.2B | GPT + BiDir + MBD | Yes | Yes | Apache 2.0 |
| Pheme | 2024 | 100–300M | T2S + SoundStorm | Yes | Yes | CC-BY-4.0 |
| XTTS v2 | 2024 | 467M | GPT-2 AR + HiFi-GAN | Yes | Yes | MPL 2.0 |
| OpenVoice v2 | 2024 | ~50M | VITS + Tone Converter | Yes | Partial | MIT |
| Kokoro | 2024-25 | 82–200M | StyleTTS 2 | Limited | No | Apache 2.0 |
| Spark-TTS | 2025 | 500M | Qwen2.5 + BiCodec | Yes | Partial | Apache 2.0 |
| CSM | 2025 | 1B | Dual Llama + Mimi | Yes | Yes | Apache 2.0 |
| GPT-SoVITS | 2024 | ~300M | GPT + VITS | Yes (few-shot) | Yes | MIT |
| Fish Speech | 2024-25 | 500M–1B | Dual AR + Firefly-GAN | Yes | Yes | Apache 2.0 |

### Table 2: Audio Codec Comparison

| Codec | Frame Rate | Codebooks | Bitrate | Sample Rate | Key Feature | License |
|-------|-----------|-----------|---------|-------------|-------------|---------|
| EnCodec | 75 Hz | 8–32 | 1.5–24 kbps | 24/48 kHz | Standard RVQ, most widely used | MIT |
| SpeechTokenizer | 50 Hz | 8 | ~4 kbps | 16 kHz | Semantic-acoustic disentanglement | Apache 2.0 |
| SNAC | 12/23/47 Hz | 3 (multi-scale) | 0.98 kbps | 24/44 kHz | Multi-scale temporal resolution | MIT |
| DAC | 86 Hz | 4–32 | 1.5–16 kbps | 44.1 kHz | Universal (speech+music), Snake activation | MIT |
| WavTokenizer | 40–75 Hz | 1 | ~0.6 kbps | 24 kHz | Single codebook, extreme compression | MIT |
| Mimi | 12.5 Hz | 8 | ~1.1 kbps | 24 kHz | Ultra-low frame rate, streaming-native | Apache 2.0 |
| AudioDec | Various | GRVQ | ~6 kbps | 48 kHz | 48kHz support, two-stage enhancement | MIT |
| FunCodec | Various | RVQ | Various | Various | Frequency-domain, flexible framework | Apache 2.0 |

### Table 3: Vocoder Comparison

| Vocoder | Type | Params | RTF (GPU) | MOS | Key Feature | License |
|---------|------|--------|-----------|-----|-------------|---------|
| HiFi-GAN V1 | GAN | 14M | ~890x RT | 4.36 | MPD + MSD, foundation vocoder | MIT |
| BigVGAN-v2 | GAN | 112M | ~300–500x RT | 4.4+ | Snake activation, universal audio | MIT |
| Vocos | GAN (freq-domain) | 13M | ~1000x+ RT | 4.3+ | ISTFT decoding, no aliasing | MIT |
| WaveGrad | Diffusion | 15M | ~0.7 (50 steps) | 4.35 | First diffusion vocoder | Apache 2.0 |
| UnivNet | GAN | 17M | ~600x RT | 4.30 | Multi-res input, LVC | BSD 3-Clause |

---

## 8. Key Architectural Trends

### 8.1 Major Paradigm Shifts (2024–2026)

1. **LLM-as-TTS**: The dominant trend is treating TTS as a language modeling problem. Models like Orpheus, Dia, OuteTTS, Fish Speech, and Spark-TTS fine-tune or extend standard LLMs (Llama, Qwen) to predict audio tokens. This leverages the massive infrastructure ecosystem around LLMs (quantization, KV-cache, speculative decoding, LoRA fine-tuning).

2. **Flow-Matching over Diffusion**: Flow-matching (F5-TTS, Voicebox, Matcha-TTS) has largely superseded traditional diffusion for non-autoregressive TTS. Key advantage: straighter trajectories → fewer inference steps → faster generation.

3. **Neural Codec Evolution**: The audio tokenizer is now a critical architectural choice:
   - Multi-scale (SNAC): Different temporal resolutions for different detail levels
   - Ultra-low frame rate (Mimi at 12.5 Hz): Minimizes sequence length for LLMs
   - Single codebook (WavTokenizer): Simplifies language modeling
   - Semantic disentanglement (SpeechTokenizer, BiCodec): Separates content from style

4. **Streaming-First Design**: Real-time conversational AI demands streaming TTS. Models like Orpheus, Dia2, CSM, and MetaVoice are designed with streaming as a core capability, not an afterthought. Target latency: 100–200ms for first audio chunk.

5. **Multi-Speaker Dialogue**: Dia pioneered single-model multi-speaker dialogue generation. CSM extended this to conversational context. This is a key differentiator for voice AI applications.

6. **Extreme Efficiency**: Models are getting smaller while quality improves:
   - Kokoro: 82M params with competitive quality
   - Matcha-TTS: 18M params, RTF 0.03
   - Pheme: 300M params, 15x faster than autoregressive baseline
   - On-device deployment is now feasible (OuteTTS via llama.cpp, Kokoro via ONNX/WebGPU)

### 8.2 Open Research Directions

1. **Robust long-form generation**: Most models still struggle with utterances longer than 30–60 seconds. Hallucination, repetition, and quality degradation remain challenges.

2. **True zero-shot voice cloning quality**: While many models claim zero-shot capability, quality with <5 seconds of reference audio is still noticeably lower than fine-tuned models.

3. **Emotional and expressive control**: Beyond basic tags (Orpheus, Dia), fine-grained continuous control over emotion, emphasis, and speaking style remains an open challenge.

4. **Multilingual quality parity**: Most models are English-first, with other languages receiving lower quality. German, in particular, benefits from specialized fine-tuning beyond base multilingual models.

5. **Evaluation standardization**: The field lacks standard benchmarks. WER, MOS, SIM-o, and RTF are reported inconsistently across papers, making direct comparison difficult.

---

## Source URLs Summary

### Papers (arXiv)
- F5-TTS: https://arxiv.org/abs/2410.06885
- E2 TTS: https://arxiv.org/abs/2406.18009
- Voicebox: https://arxiv.org/abs/2306.15687
- P-Flow: https://arxiv.org/abs/2305.02561
- Matcha-TTS: https://arxiv.org/abs/2309.03199
- Grad-TTS: https://arxiv.org/abs/2105.06337
- SpeechFlow: https://arxiv.org/abs/2307.08505
- VALL-E: https://arxiv.org/abs/2301.02111
- VALL-E 2: https://arxiv.org/abs/2406.05370
- EnCodec: https://arxiv.org/abs/2210.13438
- SpeechTokenizer: https://arxiv.org/abs/2308.16692
- SNAC: https://arxiv.org/abs/2410.02485
- DAC: https://arxiv.org/abs/2306.06546
- WavTokenizer: https://arxiv.org/abs/2408.16532
- Mimi/Moshi: https://arxiv.org/abs/2410.00037
- AudioDec: https://arxiv.org/abs/2305.02765
- FunCodec: https://arxiv.org/abs/2309.07405
- Pheme: https://arxiv.org/abs/2401.02839
- XTTS: https://arxiv.org/abs/2406.04904
- OpenVoice: https://arxiv.org/abs/2312.01479
- Spark-TTS: https://arxiv.org/abs/2503.01710
- HiFi-GAN: https://arxiv.org/abs/2010.05646
- BigVGAN: https://arxiv.org/abs/2206.04658
- BigVGAN-v2: https://arxiv.org/abs/2309.02836
- Vocos: https://arxiv.org/abs/2306.00814
- WaveGrad: https://arxiv.org/abs/2009.00713
- UnivNet: https://arxiv.org/abs/2106.07889

### GitHub Repositories
- F5-TTS: https://github.com/SWivid/F5-TTS
- Matcha-TTS: https://github.com/shivammehta25/Matcha-TTS
- Grad-TTS: https://github.com/huawei-noah/Speech-Backbones
- EnCodec: https://github.com/facebookresearch/encodec
- SpeechTokenizer: https://github.com/ZhangXInFD/SpeechTokenizer
- SNAC: https://github.com/hubertsiuzdak/snac
- DAC: https://github.com/descriptinc/descript-audio-codec
- WavTokenizer: https://github.com/jishengpeng/WavTokenizer
- Mimi/Moshi: https://github.com/kyutai-labs/moshi
- AudioDec: https://github.com/facebookresearch/AudioDec
- FunCodec: https://github.com/modelscope/FunCodec
- Orpheus TTS: https://github.com/canopyai/Orpheus-TTS
- Dia: https://github.com/nari-labs/dia
- Dia2: https://github.com/nari-labs/dia2
- OuteTTS: https://github.com/edwko/OuteTTS
- MARS5-TTS: https://github.com/Camb-ai/MARS5-TTS
- MetaVoice: https://github.com/metavoiceio/metavoice-src
- Pheme: https://github.com/PolyAI-LDN/pheme
- XTTS v2: https://github.com/coqui-ai/TTS
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- Kokoro: https://github.com/hexgrad/kokoro
- Spark-TTS: https://github.com/SparkAudio/Spark-TTS
- CSM: https://github.com/SesameAILabs/csm
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- Fish Speech: https://github.com/fishaudio/fish-speech
- HiFi-GAN: https://github.com/jik876/hifi-gan
- BigVGAN: https://github.com/NVIDIA/BigVGAN
- Vocos: https://github.com/gemelo-ai/vocos
- WaveGrad: https://github.com/ivanvovk/WaveGrad
- UnivNet: https://github.com/mindslab-ai/univnet
- Piper: https://github.com/rhasspy/piper
