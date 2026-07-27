# GLM Revision / Experiment Plan

Source: NeurIPS 2026 submission #9257 reviews (Reviewer UJ9D, uuDg, Li8o), the Area Chair meta-review, and the automated PAT pre-submission feedback. Meta-review verdict: **reject**, with an explicit list of what would raise the acceptance probability on resubmission. This plan turns that list into concrete, codebase-grounded work items against this repo.

Every item below is tagged with who asked for it: `[UJ9D]` `[uuDg]` `[Li8o]` `[Meta]` `[PAT]`.

---

## P0 — Correctness fixes (no compute, do first, blocks nothing else)

These are text/notation bugs a reviewer will re-check immediately in any resubmission — fixing them costs nothing and their absence actively damages credibility.

- **SSIM claim is internally inconsistent and one version is mathematically impossible.** Abstract/Contributions say "+0.8 SSIM", Section 4.2/Table 1 report 0.07–0.08. SSIM is bounded in [0,1], so 0.8 cannot be right. Fix every occurrence to match the table. `[UJ9D][Meta][PAT]`
- **CNN receptive field claim is wrong.** Text claims 3 stacked 7×7 layers reach "3 rows away"; additive stacking actually gives ±9 rows. Either correct the claim, or redesign the CNN/GLM comparison to be receptive-field-matched (this also affects the "fair comparison" argument reviewers are skeptical of). `[UJ9D][PAT]`
- **Table 1 parameter accounting is ambiguous.** The ~7× reduction lines up suspiciously exactly with the 2D-vs-1D kernel ratio (49/7), implying `Γ` (the image-domain CNN) may be excluded from the count on both sides. State explicitly whether Table 1 reports sinogram-module-only or full end-to-end parameters, and report both. `[PAT]`
- **"16/24 layers" → "16/24 channels"** mislabeling in Section 4.2. `[PAT]`
- **Equation/notation fixes** (Eq. 5 missing edge weights `W_ij`; Eq. 6 aggregation term should scale as `O(|ε|·n·P·c_out)` not `|ε|n`; Eq. 16/17 unweighted Laplacian action needs `W_ij`; eigenvalue bound `[0,2]` applies to the *normalized* Laplacian, not the unnormalized one used in Eq. 16 — Eq. 27's own cycle-graph derivation gives max eigenvalue 4, contradicting the earlier bound; `U` defined transposed but used as if columns are eigenvectors; summation index `i` vs `k` mismatch in Eq. 19–20; `L̂` vs `L̃` typo; scalar `θ'_0` added to a matrix without `I`; adjacency notation drifts from `W` to `A` around Eq. 24). `[PAT]`
- **Missing Limitations section.** Checklist claims broader impacts are discussed "at the end of the limitations section" — no such section exists in the main text. Add one, explicitly scoping the claims to what's actually evaluated (single dataset, single mode, single pipeline, no error bars — until P1/P2 below fix that). `[UJ9D][Li8o][PAT]`
- **Checklist accuracy pass**: Q4/Q6/Q8 point to a nonexistent "Section 6.3" (should be 3.3); Q7's justification talks about "multiple benchmarks" when only one is evaluated; Q1 affirms the abstract accurately reflects results (untrue until the SSIM fix above lands). `[PAT]`
- **Reference/citation hygiene**: DVC BibTeX entry has uncleaned GitHub usernames; Bresson & Laurent is miscited as a graph-transformer example (it's a conv architecture); `\citet` used where `\citep` is grammatically required throughout the intro; typo pass (computationnaly, euclidian→Euclidean, measrued, etc.). `[PAT]`
- **Publish the code.** UJ9D explicitly flags "the promised code has not yet been provided" as a reproducibility concern. This repo should be made public with a usable README before resubmission (the current README/DVC pipeline is already close to sufficient — see P1 tooling notes below).

None of this requires touching `src/glm/` — it's a paper-text pass.

---

## P1 — Statistical rigor (this is the #1 cross-cutting complaint)

Every reviewer and the meta-review independently flag: single run, no seeds, no error bars — and the headline 3.14 dB result is config-dependent (drops to 0.62 dB / 0.02 SSIM at 16 channels, which could easily be noise). The checklist's justification ("computational cost beyond our means") doesn't hold up: the models are <100k params and train in ~40 epochs on a single A6000-class GPU — reviewers called this out directly. `[UJ9D][Li8o][Meta][PAT]`

**Plan:**
1. Add a `seed` axis to `params.yaml` (e.g. `pretrain_parameters.hyperparameters.seed`, `train_parameters.hyperparameters.seed`), and seed `torch`, `numpy`, and the `DistributedSampler`/`DataLoader` shuffling in [pretrain.py](src/glm/pretrain.py) / [train.py](src/glm/train.py).
2. Re-run the existing CNN-16/24 vs GLM-16/24 matrix with 5 seeds each (`dvc exp run --set-param pretrain_parameters.hyperparameters.seed=N` or a small driver script over `dvc exp run --queue`).
3. Add an aggregation step (new small script, e.g. `src/glm/aggregate_results.py`) that reads the per-seed `dvclive/metrics.json` outputs and reports mean ± std for PSNR/SSIM per config — this becomes the new Table 1.
4. Report explicitly whether the 16-channel gap survives across seeds. If it doesn't, say so — a paper that's honest about a null result on one config reads far better than one that gets caught overclaiming again.

This is the highest-leverage item: it's cheap (small models, short training) and directly defuses the meta-review's "no error bars provided" line.

---

## P1 — Real external baselines

Currently the *only* comparison is against an internal, architecture-matched CNN. All three reviewers and the meta-review say this isn't enough given how much the intro leans on critiquing LPD, FISTA-Net, FBPConvNet, and transformer sinogram denoisers. `[UJ9D][uuDg][Li8o][Meta][PAT]`

**Plan, roughly in order of cost/value:**
1. **FBP-only lower bound** — trivial, the `pseudo_inverse` module in [models/utils.py](src/glm/models/utils.py) already computes this; just skip the sinogram/image NNs and log the metric. Nearly free, and currently completely absent from the paper.
2. **Tune the CNN baseline properly.** PAT flagged a "grey background" artifact in Fig. 4a as a likely sign of an undertuned baseline. A quick LR/init sweep on `sinogram_CNN`/`image_CNN` before the seed runs above protects against "the baseline was just badly tuned" as a reviewer rebuttal.
3. **FBPConvNet** — a standard, well-known image-domain post-processing baseline; moderate implementation effort, high credibility payoff since it's the most commonly expected baseline in this literature.
4. **LPD (Learned Primal-Dual)** — unrolled baseline explicitly requested by `[uuDg]`, `[Li8o]`, and the meta-review. The `odl` `OperatorModule`/ray-transform plumbing already used in `load_pseudo_inverse` makes this reachable without new infra — this is also the natural vehicle for the "GLM inside a stronger pipeline" ask below, so it's worth doing once and reusing.
5. **A transformer-based sinogram denoiser** — PAT calls its complete absence "especially surprising" given how much the intro discusses transformers. A minimal ViT-style or axial-attention block swapped in place of the CNN sinogram module is enough; doesn't need to be SOTA, just present.
6. *(Lower priority, cite-not-run if time-constrained)* graph-based projection-domain competitors (GCUNET, sinogram-graph SPECT restoration) — can be handled as related-work discussion instead of a full re-implementation if compute/time runs out.

---

## P1 — GLM ablation study (asked for by literally every reviewer)

The core complaint: GLM differs from the CNN baseline in aggregation mechanism, parameter count (~7×), *and* kernel dimensionality simultaneously, so "message passing helps" is currently unfalsifiable. `[UJ9D][uuDg][Li8o][PAT]`

Most of this is already exposed as config in `params.yaml`'s `pretrain_parameters.models.GLM` block, so it's largely a sweep, not new code:

| Ablation | What to vary | Why | Asked by |
|---|---|---|---|
| Message passing on/off | Replace GCN aggregation with identity, isolating the 1D-conv reparameterization from the graph aggregation | Directly tests the paper's central claim | UJ9D |
| Distance metric | Geodesic (current) vs. Euclidean for edge weights | Paper asserts geodesic is "not appropriate" without ever testing it | Li8o, PAT |
| Heat kernel bandwidth `σ` | Sweep `σ`; **also test whether `σ` needs to scale with angular spacing under downsampling** | PAT raises a real technical concern: at fixed `σ=1`, edge weights `exp(-d_ij²/σ²)` decay toward zero as `d_ij` grows under sparse-view subsampling, which could silently disable message passing exactly in the zero-shot generalization setting the paper's headline claim depends on. **This should be checked before anything else in this table — if true, it undermines the paper's main result, not just an ablation.** | Li8o, PAT |
| Connectivity scheme | Immediate-neighbor only vs. wider/k-hop | uuDg asks directly; cheap to test via `create_graph_from_geometry` scheme options | uuDg |
| Depth | Number of stacked GLM modules (1/2/3/4) | uuDg | uuDg |
| Aggregation op | `aggr`: add/mean/max — already a param in `params.yaml` | Free sweep, already wired | (general rigor) |
| GCN flags | `normalize` / `improved` / `add_self_loops` / `cached` sensitivity | Already parameterized, cheap to sweep; note also — since the graph is invariant across the whole run, `cached: true` is a legitimate correctness-preserving speed win worth adopting regardless (see engineering note below) | (general rigor) |

**Action item, do this first:** before running the full ablation sweep, directly verify in code whether `σ` (or the geodesic distances feeding the heat kernel) are recomputed per-geometry at inference time or baked in from training-time spacing. If it's the latter, the zero-shot sparse-view result needs re-validation with a distance-aware (not fixed-bandwidth) kernel.

---

## P2 — Broaden the generalization evidence

All evaluation is 2DeteCT Mode2 only, 2D only, single distribution shift (angular undersampling). The paper's framing (clinical CT) doesn't match its evidence (materials-science dataset). `[UJ9D][uuDg][Li8o][Meta]`

1. **Run Mode1 and Mode3**, not just Mode2 — `preprocess_2detect.py` already takes `mode` as a CLI arg, so this is a re-run of the existing pipeline with `mode1`/`mode3`, not new code. Li8o asks this directly and ties it to a rating improvement.
2. **Add one dataset beyond 2DeteCT**, ideally one that supports the clinical-CT narrative (e.g. LoDoPaB-CT or the AAPM Low-Dose CT Grand Challenge data) — needed either to justify the clinical framing or to soften it explicitly.
3. **Test an additional distribution shift** beyond angular undersampling — noise level is the cheapest to add given the existing pipeline.
4. **3D/helical geometry**: Appendix A derives this theoretically but it's never run. Either instantiate a minimal 3D experiment, or explicitly scope the abstract/intro to 2D and move the 3D material to "future work" so the claims match the evidence.
5. Rewrite the intro to acknowledge 2DeteCT's materials-science origin and either justify transfer relevance to clinical CT or drop the clinical framing.

---

## P2 — Complete the efficiency story

Table 1 has params, Figure 6 has training time/memory — but no *inference* runtime or FLOPs, and fewer parameters doesn't imply faster inference once graph construction/message-passing overhead is counted. `[uuDg][PAT]`

- Measure wall-clock inference time for CNN vs. GLM at matched configs, including graph batching overhead (this is exactly the cost the [train.py](src/glm/train.py)/[pretrain.py](src/glm/pretrain.py) hoisting fix from this session removed from the *training* hot path — worth explicitly noting in the paper that graph batching is a one-time cost, not per-inference, if that's now true).
- Report FLOPs.
- State the batch size used in the Figure 6 memory plot caption; fix "Gb" → "GB".

---

## P2 — GLM inside a stronger reconstruction pipeline

Right now GLM is only tested as a drop-in for the sinogram-denoising stage of one fixed FBP + image-CNN pipeline. Reviewers want to know if the benefit survives inside a stronger pipeline. `[uuDg]`

- Reuse the LPD implementation from the baselines work above: swap its internal sinogram-domain block for a GLM module and compare against LPD-with-CNN-block. This single experiment answers both "is GLM better than LPD" (P1 baseline) and "does GLM help *inside* a strong pipeline" (this item) at once.

---

## P3 — Polish (do last, cheap, no compute)

- Improve Figure 3 (residual connection is drawn unintuitively) and Figure 4 (make baseline-vs-proposed visual differences legible). `[Li8o]`
- Add a paragraph contextualizing against zero-shot generative-prior sparse-view CT methods — discussion only, no new experiments needed. `[PAT]`
- Add a paragraph on graph-based projection-domain related work (GCUNET etc.) even where not run head-to-head. `[Meta][PAT]`

---

## Suggested execution order

1. **P0** (text/notation fixes) — do in parallel with everything else, zero compute cost.
2. **Check the `σ`/geodesic-distance scaling question** (flagged under GLM ablations) — this could invalidate the headline zero-shot claim, so resolve it before investing in the full seed/baseline sweep.
3. **P1 seeds + aggregation script** — cheap, and every other number you produce from here on should be reported with error bars from the start rather than retrofitted later.
4. **P1 baselines** (FBP lower bound → tuned CNN → FBPConvNet → LPD → transformer) — LPD unlocks the P2 "GLM inside a stronger pipeline" experiment for free.
5. **P1 ablations** — mostly `params.yaml` sweeps, run once the seed/logging infra from step 3 exists.
6. **P2 generalization breadth** (extra modes, extra dataset, extra shift) and **P2 efficiency numbers**.
7. **P3 polish**, final consistency pass, then re-verify every P0 item against the final numbers before resubmission.

## Engineering notes tied to this repo

- Seeds, baseline model variants (FBP-only / FBPConvNet / LPD / transformer), and most ablations fit naturally as new `params.yaml` axes and/or new `load_model`/`load_pseudo_inverse` branches in [models/utils.py](src/glm/models/utils.py) — the existing `active_model` / `active_pseudo_inverse` config-driven dispatch pattern already supports this without restructuring `train.py`/`pretrain.py`.
- Given the volume of new experiment configs, consider a `dvc exp run --queue` + `dvc exp run --run-all` sweep instead of hand-editing `params.yaml` per run, and a small results-aggregation script (mean/std across seeds) feeding directly into the revised Table 1.
- The `cached: true` GNN flag noted in the ablation table is safe to flip permanently for all runs now that the graph-batching hoist (this session's change to `train.py`/`pretrain.py`) guarantees a fixed graph/batch-size per run — free speed-up for the whole sweep above.
