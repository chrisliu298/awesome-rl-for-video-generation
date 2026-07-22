# Contributing

Contributions welcome! Open a PR to add papers, datasets, benchmarks, or tools related to reinforcement learning, preference optimization, and reward-driven post-training / alignment for video generation.

## Inclusion criteria

A resource must pass **one** of these two tests:

1. **The work defines, trains, or applies an RL / preference-optimization / reward-gradient method to align or post-train a video generation model.** This includes RLHF/PPO and RLAIF, DPO and its variants, GRPO / flow-GRPO / policy-gradient methods, reward backpropagation / differentiable-reward fine-tuning, and inference-time / test-time reward-guided alignment of a video generator.
2. **The work directly enables RL/alignment for video generation.** Preference/RL datasets, alignment benchmarks, and reward models / verifiers whose explicit purpose is to *drive* RL or preference optimization of a video generator, plus tooling that makes such training practical.

### What does NOT qualify

- **Pure supervised fine-tuning or distillation** with no reward or preference signal.
- **Generic video QA.** Video question-answering or captioning not tied to generation alignment.
- **Image-only RL methods** unless a direct conceptual foundation for video (e.g., DDPO, DPOK, Diffusion-DPO, ImageReward-driven alignment).
- **Pure generation methods** without a substantive RL / preference / reward-gradient component. A paper that mentions RL or DPO incidentally (e.g., a single Best-of-N line in the appendix) does not qualify.
- **Domain-specific applications** need a transferable methodological contribution beyond the domain.

### When in doubt

Read the actual paper. If the verdict depends on whether an RL / preference / reward-gradient method is a substantive part of the training, check the method section. Reward models get **light** treatment here (only rewards used to *drive* RL) — cross-link to the sibling [reward-model list](https://github.com/chrisliu298/awesome-rm-for-video-generation) for depth.

## Section placement

### Foundations and Optimization Paradigms

Essential background: PPO/RLHF/RLAIF, DPO, GRPO, Bradley–Terry preference modeling, and image-diffusion RL foundations (DDPO, DPOK, Diffusion-DPO). Rarely updated.

### Reinforcement Learning for Video Generation

Policy-gradient / GRPO / PPO methods that align a video generator. Subsections:

- **End-to-end / system-level RL alignment** — full-pipeline video RLHF
- **Process-level RL** — diffusion-sampling / trajectory-level intervention
- **Stability, efficiency, and structured constraints** — physics / identity / camera / geometry-constrained RL

### Preference Optimization (DPO-family)

Papers whose main contribution is a direct preference objective. Subsections:

- **Direct alignment objectives** — DPO adaptations, automatic preference-pair construction
- **Fine-grained / structured preference supervision** — segment-, timestep-, or hierarchy-level preferences
- **Stability / efficiency / hybrid preference–RL** — GRPO–DPO hybrids, diversity-aware, negative-preference

### Reward Backpropagation and Differentiable-Reward Alignment

Aligning by backpropagating a differentiable reward through denoising (e.g., VADER, T2V-Turbo, InstructVideo, IPRO).

### Reward Models and Verifiers for RL

Compact — rewards whose purpose is to *drive* RL/preference optimization (multi-dimensional, identity/consistency, physics-aware / verifiable, reward-hacking mitigation). Cross-link to the RM list rather than duplicating its full taxonomy.

### Inference-Time and Test-Time Alignment

Subsections:

- **Guidance-based alignment** — reward/CFG-style trajectory guidance and auxiliary-model steering
- **Iterative refinement / self-correction / test-time search** — test-time refinement, self-editing, search

### Datasets and Benchmarks

Table format. Subsections for RL/preference/physics-alignment datasets and alignment benchmarks / learned evaluators.

### Project Pages, Repos, and Useful Links

Links to repos, leaderboards, and adjacent collections (including the sibling reward-model list).

## Entry format

```
- [Full Paper Title](url) *(Year)* — One-line description.
```

The description should be a single terse sentence capturing what makes this paper distinctive. Look at existing entries for calibration.

## Batch additions

When adding multiple papers at once:

- **Flag section growth.** If a single update would double any subsection's size, consider whether the section needs splitting or whether some entries are marginal.
- **Prioritize gap-filling over completeness.** A paper that opens a new niche has higher priority than a fourth variant in a well-covered area.
- **Cap awareness.** Large batches dilute curation signal. Prefer 5-8 high-confidence additions over 12+ with several borderline entries.

## Taxonomy and reading path

- Update the taxonomy tables when a paper clearly fits an existing row. Skip taxonomy updates for papers that don't fit neatly.
- The Start Here reading path changes rarely. Only add a paper if it is the best introduction to a topic currently underrepresented in the path.

## Duplication

Duplication across sections is fine for exceptionally central papers, but prefer one primary location.
