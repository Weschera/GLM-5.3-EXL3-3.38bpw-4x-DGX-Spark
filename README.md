# GLM-5.3 (753B) 3.38bpw on 4× DGX Spark: 84.4 on Spark-Bench

GLM-5.3 full, served as one OpenAI-compatible endpoint across **four DGX Sparks** (TensorFold tensor-parallel TP4 over
RoCE, DFlash2 drafting), scored **84.4/100 TrueScore** on Spark-Bench v7.0: 80 tests, 2 repeats, thinking off,
uncapped, 0 transport errors.

**This is a deployment of [@ViC305](https://x.com/ViC305)'s recipe, not a new engine.** Victor Cruz built the SAGE
MixedK 3.38bpw quant and the four-Spark TensorFold recipe:
[vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe) @ `ed2f31d`
(MIT). Follow his README and AGENTS.md. This repo adds what we needed to make it run on our cluster, the benchmark
evidence, and the scripts we used.

## Score: up from 70.8

Our earlier four-Spark GLM-5.3 recipe ([keys 2.75bpw](https://github.com/Weschera/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark))
scored 70.8 on the same benchmark. Same model family and the same four Sparks; the quant and the engine changed.

| Category (quality) | **3.38bpw (this)** | 2.75bpw (old) |
|---|---:|---:|
| **TrueScore** | **84.4** | 70.8 |
| Long-output builds | **70.5** | 25.9 |
| Code | **82.5** | 28.5 |
| Robustness | **87.7** | 81.5 |
| Instructions | **97.0** | 92.2 |
| Agentic | **76.0** | 67.4 |
| Visual | **76.1** | 70.0 |
| Tool use | 78.4 | 78.4 |
| Planning | 86.5 | 88.8 |
| Structured output | 96.7 | 97.1 |
| Long context (2 tests) | 58.8 | 70.8 |

The old run's code and long-output scores were also hurt by an engine hang that zeroed two tests, so part of the jump
is that run's problem, not the 2.75 quant alone.

Where it sits on our board (same benchmark): RED-SNOW 2.49bpw on 1 Spark 85.1 · GLM-5.3-Flash 4bpw on 2 Sparks 84.8 ·
**GLM-5.3 3.38bpw on 4 Sparks 84.4**. A tie within 0.7 points. GLM-5.3 full wins long builds (70.5, best we have
measured), robustness and instructions; the Flash models win code, visuals and long-context retrieval, on less hardware.

Full report: [`evidence/spark-bench-v7-g338-thinking-off.md`](evidence/spark-bench-v7-g338-thinking-off.md)
(per-test scores and reasons; `.html` and `spark_bench.csv` alongside). Smoke test 21/21:
[`evidence/smoke-21-of-21.txt`](evidence/smoke-21-of-21.txt).

## Speed (ours, measured through the benchmark)

- 747,127 completion tokens over the run; 33.3 tok/s overall average including prefill and tool turns.
- Long single answers: 34–41 tok/s (e.g. 41.6K tokens at 41.3 tok/s, 127K tokens at 34.1 tok/s).
- One request at a time, so the full benchmark took ~6.2 hours of generation (vs 41 minutes for GLM-5.3-Flash on 2
  Sparks, which writes shorter answers and runs ~3× faster per token).
- Victor's own reference numbers (6 prompts, greedy, engine-reported) are in his README: 41.11 tok/s mean with DFlash2.
  Ours are client-side, sampled, on benchmark prompts; do not mix the two.

## What we changed from his recipe, and why

Every change was forced by our cluster; none changes the model, the engine or the sampling.

1. **Prebuild the CUDA extensions before `./glm53 up`** — [`scripts/prebuild-extensions.sh`](scripts/prebuild-extensions.sh).
   On first start each rank JIT-compiles three TensorFold extensions *after* the weights are loaded. One of our Sparks
   was at ~6 GiB free at that point; the compile took it under 2 GiB and the memory guard stopped all four ranks.
   Building them first, with nothing loaded, fixed it. The script also clears stale build locks left by a killed build
   (they make every later start wait), and stops with the `chown` line to run if an earlier root run left
   `~/.cache/torch_extensions` or `~/.triton` owned by root (that blocked our first build).
2. **One copy of the pack, shared over NFS** — [`scripts/share-pack-nfs.sh`](scripts/share-pack-nfs.sh). None of our
   Sparks had room for its own 319 GB copy, so the pack lives on one Spark and the other three mount it read-only at the
   same path over the fabric. Needs `nfs-kernel-server` on the serving Spark (ours was missing it).
3. **`hosts` written by hand.** All four of our Sparks were cloned from one image and share `/etc/machine-id`, so
   `./glm53 init` refuses (it is the documented case in his table). We wrote `tensorfold-four-spark-tp4/hosts` from
   `hosts.example` with addresses read from each Spark (`ip -br addr`). If your Sparks have distinct machine-ids, use
   `./glm53 init`.
4. **Swap off during the run.** His watchdog stops everything if swap grows at all. One of our Sparks swapped out 4 kB
   during load with 100+ GiB free (twice, at swappiness 1), which tripped it. We ran with `swapoff -a` on all four; the
   low-memory guard stayed on. Victor's guidance is not to allow swap growth; with swap off there is none, but this is
   a deviation from his tested setup, so we disclose it.
5. **Client reaches `/v1` through an ssh tunnel** (his default loopback bind kept): `ssh -N -L 18890:127.0.0.1:8890 <rank 0>`.

## Steps

1. His prerequisites and steps 0–2 from
   [AGENTS.md](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe/blob/ed2f31d/AGENTS.md): four Sparks on one
   QSFP switch, same user, passwordless ssh, HF access to the gated
   [vcruz305/GLM-5.3-EXL3-3.38bpw](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) pack (we pinned `cc64e77`).
2. If disk is short: download the pack once, then `scripts/share-pack-nfs.sh`, then
   `./glm53 setup --model-dir <that path>`.
3. If `./glm53 init` reports `same /etc/machine-id`: write `hosts` by hand (change 3).
4. `bash scripts/prebuild-extensions.sh`
5. `./glm53 up` (default profile `fast-160k`, 163,840 tokens), then `./glm53 smoke` → `ALL CHECKS PASSED: 21/21`.
6. Optional: `scripts/run-spark-bench.sh` to reproduce the score.

## Benchmark settings

Spark-Bench v7.0 (grader `509759f`, golden gate 73/73), `--thinking off --uncapped --repeats 2 --tier all`,
temperature 1.0, top_p 0.95 (GLM-5.3's recommended sampling), seeds 2001–2002, max_tokens = the remaining 163,840
window on every request. Our exact run script: [`evidence/our-run-script-as-used.sh`](evidence/our-run-script-as-used.sh).

## Known issue

Victor's server keeps generating after a client disconnects, and serves one request at a time. A client that times out
on a looping answer can leave the server busy until it reaches the context limit. It did not happen in this run.

## Credits

- **Victor Cruz ([@ViC305](https://x.com/ViC305))** — SAGE MixedK 3.38bpw quant and the four-Spark TensorFold recipe.
- **TensorFold** (ashhart) with Victor's pinned fork; **b12x** RoCE; **DFlash2** drafter
  [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) (CC BY-NC-ND 4.0, non-commercial).
- **Z.ai** — GLM-5.3.

Scripts in this repo: MIT.
