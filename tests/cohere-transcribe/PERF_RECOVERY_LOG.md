# Cohere CPU Perf Recovery Log

Warm benchmark contract used for these notes:

- CPU only
- batch size 1
- model loaded once per run
- primary metric: `transcribe_total_s`
- command:

```bash
./build/bin/cohere-bench \
  --model ./artifacts/gguf/cohere-transcribe-f16.recovery.gguf \
  --file ./samples/jfk.wav \
  --language en \
  --threads 8
```

## Baseline

- Runtime state:
  - profiling/counters enabled
  - reusable `ggml` linear workspace enabled
  - relative-position cache enabled
- Result:
  - `transcribe_total_s = 10.82`
  - biggest buckets:
    - `encoder_total = 6.158855s`
    - `encoder_self_attention = 2.806857s`
    - `encoder_ffn = 2.592336s`
    - `frontend = 1.775815s`
    - `subsampling = 1.529091s`

## Experiment 1: Replace Naive DFT With FFT + Threaded Frontend

- Change:
  - replaced naive frontend DFT with cached Cooley-Tukey FFT
  - parallelized mel frame work across `--threads`
- Result:
  - `transcribe_total_s = 6.216`
  - improvement vs baseline: about `1.74x`
  - frontend dropped from `1.775815s` to `0.003548s`
- Keep:
  - yes

## Experiment 2: Fast Path For Subsampling 1x1 Pointwise Convs

- Change:
  - replaced subsampling pointwise conv scalar loops with matmul-based path
- Result:
  - `transcribe_total_s = 5.154`
  - improvement vs previous experiment: about `1.21x`
  - subsampling dropped from `1.277069s` to `0.129564s`
- Keep:
  - yes

## Current Hotspots

- `encoder_self_attention = 2.023371s`
- `encoder_ffn = 1.482333s`
- `decoder_total = 1.081091s`
- `encoder_conv = 0.398733s`

## Experiment 3: BLAS Encoder Relative-Position Self-Attention

- Change:
  - replaced encoder attention score/value loops with BLAS GEMMs
  - kept exact relative-position indexing and softmax semantics
  - linked `whisper` directly against BLAS so the Cohere path can call CBLAS
- Result:
  - isolated `jfk.wav` warm runs moved from about `5.717s` to `2.164s` to `2.406s` depending on run order, with repeat runs settling near `2.2s`
  - `encoder_self_attention` dropped from about `1.96s` to about `0.36s`
- Keep:
  - yes

## Experiment 4: Fuse FFN Into One ggml Graph

- Change:
  - fused `linear1 + activation + linear2` into a single ggml graph for encoder and decoder FFNs
- Result:
  - graph launches dropped from `2888` to `2504`
  - FFN sub-buckets improved
  - end-to-end moved only slightly on `jfk.wav`, around `2.164s` to `2.138s`
- Keep:
  - yes for now, but this was a modest win rather than a breakthrough

## Experiment 5: BLAS Decoder Attention

- Change:
  - replaced decoder self-attention and cross-attention score/value loops with BLAS GEMVs
- Result:
  - `jfk.wav` moved from about `2.14s` to about `1.99s` to `2.04s`
  - decoder total dropped from about `0.58s` to about `0.46s`
- Keep:
  - yes

## Experiment 6: Single-Token Linear Fast Path With ggml CPU `vec_dot`

- Change:
  - added a direct `vec_dot`-based fast path inside `eval_linear()` for single-column decoder projections
  - used ggml CPU type traits to convert the input once and avoid tiny ggml graph launches
  - kept the LM head on the existing path by limiting the fast path to moderate output sizes
- Result:
  - graph launches dropped from `2504` to `776`
  - decoder total collapsed from about `0.46s` to about `0.15s`
  - `jfk.wav` moved to about `1.58s`, then repeated around `1.69s` after reverting later failed experiments
- Keep:
  - yes

## Experiment 7: Welford Layer Norm

- Change:
  - rewrote layer norm to a 2-pass Welford form with direct column pointers
- Result:
  - neutral to slightly worse on repeated `jfk.wav` runs
- Keep:
  - no, reverted

## Experiment 8: Threaded Conformer Depthwise 1D Conv

- Change:
  - parallelized depthwise conv across channels in the Conformer conv module
- Result:
  - the conv bucket sometimes improved, but end-to-end regressed after repeated runs
- Keep:
  - no, reverted

## Experiment 9: Preconvert Conformer Depthwise Kernels To F32 Once Per Call

- Change:
  - when the depthwise 1D conv weight is F16, convert the whole tiny kernel tensor to F32 once before the time loop instead of converting individual taps inside the innermost multiply
- Result:
  - isolated `jfk.wav` warm runs improved from about `1.69s` to about `1.63s`, then `1.52s`
  - `encoder_conv` dropped from about `0.325s` to about `0.198s`
  - 10-clip smoke compare improved from `2.249x` to `2.884x` GGUF-over-HF warm transcribe speedup
  - 100-clip compare improved from `1.919x` to `2.036x` GGUF-over-HF warm transcribe speedup
- Keep:
  - yes

## Experiment 10: Apple vecLib Oversubscription Guard

- Change:
  - added a macOS Cohere-tool guard that re-execs with `VECLIB_MAXIMUM_THREADS` capped when the user has not already pinned it
  - this is intended to prevent hidden nested BLAS oversubscription during the fixed `--threads 8` CPU benchmark contract
- Result:
  - plain local `jfk.wav` runs stabilized around `~3.0s` instead of drifting up into the `~3.3s` to `~3.6s` range
- Keep:
  - yes as a scheduling guard on macOS
  - do **not** count this as an implementation-only speedup when reporting algorithmic wins

## Experiment 11: Feed Graph Matmuls With Weight `vec_dot_type` Inputs

- Change:
  - stopped forcing graph matmul inputs to F32
  - when the destination weight advertises an F16/F32 `vec_dot_type`, create the ggml activation tensor in that type directly
- Result:
  - 1-clip warm `jfk.wav` improved from about `2.995s` to about `2.948s`
  - 10-clip smoke compare improved from `22.924s` to `21.003s` GGUF transcribe time
  - 10-clip GGUF-over-HF warm transcribe speedup improved from `2.187x` to `2.372x`
- Keep:
  - yes

## Experiment 12: Fuse Shared-Input Projection Graph Launches

- Change:
  - grouped encoder `q/k/v` plus relative-position projection into a single graph launch per layer
  - grouped decoder cross-attention key/value build into a single graph launch per layer
- Result:
  - graph launches dropped from `776` to `624`
  - 1-clip warm `jfk.wav` improved from about `2.948s` to `2.703s`
  - 10-clip smoke compare improved from `21.003s` to `19.961s` GGUF transcribe time
  - 10-clip GGUF-over-HF warm transcribe speedup improved from `2.372x` to `2.604x`
  - correctness stayed flat at `0.59%` WER and `9/10` normalized exact matches on the 10-clip smoke set
- Keep:
  - yes

## Experiment 13: Decoder FFN Single-Token `vec_dot` Fast Path

- Change:
  - when `eval_linear_activation_linear()` sees a single-column input, it now skips the generic ggml graph and runs both FFN projections through the existing `vec_dot` single-token linear path with the scalar activation in between
  - this only changes the decoder-style single-token case; encoder FFNs still use the previous multi-column graph path
- Result:
  - 1-clip `jfk.wav` kept the exact transcript and cut `decoder_ffn` from roughly `0.165s` to about `0.098s`
  - 1-clip graph launches dropped from `624` to `336`
  - 5x repeated-`jfk.wav` loaded-once microbench improved from `15.013s` to `12.350s` total transcribe time in the same experiment loop
  - on that same 5x microbench, aggregate `decoder_ffn` dropped from `1.398s` to `0.443s`
- Keep:
  - yes

## Experiment 14: Cached Decoder Position Table And Direct Embedding Column Read

- Change:
  - cached the decoder sinusoidal position encodings at model load
  - changed `get_decoder_embedding()` to memcpy or fp16-decode the token embedding column directly, then add the cached position column before layer norm
- Result:
  - correctness stayed fine: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - but a same-session 5x repeated-`jfk.wav` loaded-once A/B showed this was slower, not faster
  - cached-position build: `transcribe_total_s = 13.948`
  - reverted baseline in the same session: `transcribe_total_s = 13.472`
  - decoder buckets also got worse in the cached-position build, including `decoder_total = 1.153s -> 1.053s` after reverting and `lm_head = 0.276s -> 0.218s` after reverting
- Keep:
  - no
  - reverted immediately

## Experiment 15: Skip Prompt-Prefill LM Head Work Until The Last Prompt Token

- Change:
  - added a `need_logits` switch to `decoder_step()`
  - prompt-prefill tokens now stop after the decoder blocks unless they are the final prompt token; this skips the otherwise-unused final decoder norm and LM-head projection on earlier prompt tokens
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - counters confirmed the structural reduction exactly as expected on the 5x repeated-`jfk.wav` loaded-once microbench:
    - `eval_linear_calls`: `12910 -> 12865`
    - `ggml_graph_launches`: `1680 -> 1635`
    - `input_tensor_copies`: `1920 -> 1875`
    - `output_tensor_copies`: `2440 -> 2395`
  - same-session total transcribe time improved from `13.472s` on the reverted baseline to `12.973s` and `13.315s` on two prompt-skip runs
  - mean of those two prompt-skip runs: `13.144s`, about `1.025x` faster than the same-session reverted baseline
- Keep:
  - yes

## Experiment 16: Skip Decoder Log-Softmax In The Normal Transcribe Path

- Change:
  - tried using raw decoder logits for greedy decoding in the normal transcribe path, while keeping the debug/parity path on log-softmaxed logits
  - this was meant to save the post-LM-head exp/log reduction because greedy argmax is invariant to log-softmax
- Result:
  - correctness stayed fine on `test-cohere` and `samples/jfk.wav`
  - but the benchmark signal was not good enough to justify keeping it
  - first same-session 5x repeated-`jfk.wav` loaded-once run was slightly slower than the kept prompt-skip baseline: `13.315s -> 13.359s`
  - repeat run did not recover a clean win and came in substantially worse under hot-session noise
- Keep:
  - no
  - reverted immediately

## Experiment 17: Direct Decoder Token-Embedding Column Reads

- Change:
  - kept the existing per-step sinusoidal position math
  - but stopped fetching decoder token embeddings one element at a time through `ggml_get_f32_nd`
  - instead, when the token embedding tensor is contiguous F32 or F16, copy or fp16-decode the full token column directly and then add the positional terms in place
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - 5x repeated-`jfk.wav` loaded-once runs improved from the kept prompt-skip baseline `13.315s` to `13.297s` and `12.922s`
  - mean of those two runs: `13.110s`, about `1.016x` faster than the kept prompt-skip baseline
  - the first run also showed the decoder buckets moving in the expected direction, including `decoder_total = 1.058s -> 0.970s`
- Keep:
  - yes

## Experiment 18: Cache Decoder Tensor Pointers Per Inference

- Change:
  - built a per-inference decoder-layer pointer cache so the hot decode loop could reuse tensor pointers instead of rebuilding decoder layer prefix strings and re-querying the model tensor map on every token
  - no math changed
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - decoder buckets improved in both runs, but the primary total-time metric did not
  - against the kept direct-embedding baseline run at `12.922s`, the cached-pointer runs landed at `13.164s` and `13.283s`
  - that made the mean slower overall despite better decoder-only buckets
- Keep:
  - no
  - reverted immediately

## Experiment 19: Avoid `q/k/v/p` Copy-Out In Encoder Self-Attention

- Change:
  - tried reading contiguous ggml `q/k/v/p` graph outputs directly inside the BLAS encoder-attention path instead of first copying them into temporary `tensor2d` buffers
  - this reduced the `output_tensor_copies` counter significantly
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - the first run looked promising and cut `output_tensor_copies` from `2395` to `1435`
  - but a same-session A/B against an immediately rebuilt reverted baseline did not show a stable total-time win
  - hot-session reverted baseline: `12.635s`
  - copyless run in the same session: `12.671s`
- Keep:
  - no
  - reverted immediately

## Experiment 20: Reuse Encoder-Attention BLAS Scratch Buffers

- Change:
  - tried reusing the large temporary BLAS workspace vectors inside encoder self-attention instead of reallocating them on each layer call
  - no math changed
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - same-session 5x repeated-`jfk.wav` loaded-once benchmark regressed from the fresh reverted baseline `12.635s` to `12.936s`
  - encoder self-attention also moved the wrong way in that run: `2.712s -> 2.975s`
- Keep:
  - no
  - reverted immediately

## Experiment 21: BLAS `saxpy` For Residual `add_scaled`

- Change:
  - replaced the scalar `add_scaled()` loop with a BLAS `copy + saxpy` path on macOS
  - this affected every residual connection in both encoder and decoder
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - same-session 5x repeated-`jfk.wav` loaded-once benchmark regressed from the fresh reverted baseline `12.635s` to `13.156s`
  - the regression showed up most clearly in `encoder_ffn` and `decoder_total`
- Keep:
  - no
  - reverted immediately

## Experiment 22: Fuse Encoder FFN Residual Add Into The GGML Graph

- Change:
  - tried folding the encoder FFN residual add into the FFN graph so the FFN output would be added to the residual inside ggml rather than in a separate `add_scaled()` pass
  - applied only to the two encoder FFNs inside each Conformer block
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - but the graph fusion introduced extra residual input copies and regressed the benchmark
  - same-session 5x repeated-`jfk.wav` loaded-once benchmark regressed from the fresh reverted baseline `12.635s` to `13.283s`
  - `input_tensor_copies` also increased from `1875` to `2355`
- Keep:
  - no
  - reverted immediately

## Experiment 23: Graph-Native Encoder FFN Sub-Block

- Change:
  - tried pushing the whole encoder FFN sub-block into ggml from the residual input onward:
    - layer norm
    - norm affine
    - linear1
    - SiLU
    - linear2
    - residual scale
    - residual add
  - this removed the CPU `layer_norm()` and `add_scaled()` steps for the encoder FFNs
- Result:
  - correctness stayed flat: `test-cohere` stayed green and `samples/jfk.wav` kept the exact transcript
  - but the same-session 5x repeated-`jfk.wav` loaded-once benchmark regressed from the fresh reverted baseline `12.635s` to `12.906s`
  - the regression landed directly in the target bucket: `encoder_ffn = 6.210s -> 6.499s`
- Keep:
  - no
  - reverted immediately

## Current Reproducible Snapshot

- Measurement note:
  - the earlier `10-clip` result with GGUF at `9.967s` came from a different session on the same benchmark contract and same 10-clip subset, where HF measured `28.746s`
  - the current `10-clip` reproducible snapshot below was collected in a later, hotter session, where HF measured `51.976s` and GGUF measured `19.961s`
  - because both HF and GGUF slowed substantially in that later session, this does **not** look like a GGUF-only regression; treat `9.967s` as a historical earlier-session peak and `19.961s` as the current reproducible datapoint under different machine conditions

- 1-clip warm `jfk.wav`:
  - `transcribe_total_s = 2.703`
  - biggest buckets:
    - `encoder_total = 2.245927s`
    - `encoder_ffn = 1.268172s`
    - `encoder_self_attention = 0.571897s`
    - `encoder_conv = 0.390349s`
    - `decoder_total = 0.314164s`
- 10-clip warm LibriSpeech smoke compare:
  - HF native `transcribe_total_s = 51.976s`
  - GGUF `transcribe_total_s = 19.961s`
  - speedup: `gguf_vs_hf_transcribe = 2.604x`
  - both paths matched `0.59%` WER and `9/10` normalized exact matches
- 100-clip warm LibriSpeech validation:
  - HF native `transcribe_total_s = 451.772s`
  - GGUF `transcribe_total_s = 232.828s`
  - speedup: `gguf_vs_hf_transcribe = 1.940x`
  - WER: HF `1.10%`, GGUF `1.19%`
  - note: this long run was collected while the machine was already hot, so the absolute wall-clock numbers should be rechecked on a cool machine before using them as headline figures

## Remaining Hotspots

- `encoder_ffn` is still the dominant bucket at about `46.5%` of 100-clip GGUF transcribe time
- `encoder_self_attention` is next at about `19.8%`
- `encoder_conv` still costs about `12.4%`
- `decoder_ffn` still costs about `10.2%`
