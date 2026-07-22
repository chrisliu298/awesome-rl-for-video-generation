# 🎬 Awesome RL for Video Generation

<p align="center">
  <a href="https://awesome.re"><img src="https://img.shields.io/badge/Awesome-%F0%9F%8E%AC_RL_for_Video_Generation-000000?style=for-the-badge&labelColor=000000" alt="Awesome RL for Video Generation"></a>
</p>

<p align="center">
  <!-- entry-count-start --><a href="#contents"><img src="https://img.shields.io/badge/Entries-204-000000?style=for-the-badge&labelColor=000000" alt="Entries"></a><!-- entry-count-end -->
  <a href="https://github.com/chrisliu298/awesome-rl-for-video-generation/stargazers"><img src="https://img.shields.io/github/stars/chrisliu298/awesome-rl-for-video-generation?style=for-the-badge&logo=github&logoColor=white&label=Stars&labelColor=000000&color=000000" alt="GitHub Stars"></a>
  <a href="https://github.com/chrisliu298/awesome-rl-for-video-generation/network/members"><img src="https://img.shields.io/github/forks/chrisliu298/awesome-rl-for-video-generation?style=for-the-badge&logo=github&logoColor=white&label=Forks&labelColor=000000&color=000000" alt="GitHub Forks"></a>
  <a href="https://github.com/chrisliu298/awesome-rl-for-video-generation/commits"><img src="https://img.shields.io/github/last-commit/chrisliu298/awesome-rl-for-video-generation?style=for-the-badge&logo=github&logoColor=white&label=Last%20Commit&labelColor=000000&color=000000" alt="GitHub Last Commit"></a>
</p>

A curated list of **reinforcement learning, preference optimization, and reward-driven post-training and alignment methods for video generation**.

> Related: [Awesome Reward Models for Video Generation](https://github.com/chrisliu298/awesome-rm-for-video-generation), the sibling list for reward-model architectures, training data, judges, and evaluator taxonomies.

## Contents

- [Start Here](#start-here)
- [Concept Primer](#concept-primer)
- [Quick Start by Goal](#quick-start-by-goal)
- [Foundations and Optimization Paradigms](#foundations-and-optimization-paradigms)
- [Reinforcement Learning for Video Generation](#reinforcement-learning-for-video-generation)
  - [End-to-End and System-Level Post-Training](#end-to-end-and-system-level-post-training)
  - [Process, Trajectory, Stability, and Efficiency](#process-trajectory-stability-and-efficiency)
  - [Verifiable Rewards and Structured Constraints](#verifiable-rewards-and-structured-constraints)
- [Preference Optimization (DPO-family)](#preference-optimization-dpo-family)
  - [Direct Video Preference Objectives](#direct-video-preference-objectives)
  - [Fine-Grained, Structured, and Geometry-Aware Preferences](#fine-grained-structured-and-geometry-aware-preferences)
  - [Multi-Objective and Hybrid Preference–RL](#multi-objective-and-hybrid-preferencerl)
- [Reward Backpropagation and Differentiable-Reward Alignment](#reward-backpropagation-and-differentiable-reward-alignment)
- [Reward Models and Verifiers for RL](#reward-models-and-verifiers-for-rl)
- [Inference-Time and Test-Time Alignment](#inference-time-and-test-time-alignment)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Project Pages, Repos, and Useful Links](#project-pages-repos-and-useful-links)
- [Contributing](#contributing)
- [Citation](#citation)

## Start Here

A practical reading order:

1. **Learn the optimization vocabulary.** Read the [concept primer](#concept-primer), then PPO, DPO, and the GRPO section of DeepSeekMath.
2. **Follow the policy-gradient path.** Read DDPO and DPOK before DanceGRPO; then compare BranchGRPO, TAGRPO, DenseGRPO, OP-GRPO, and Flash-GRPO for rollout structure, credit assignment, and efficiency.
3. **Follow the preference path.** Read DPO and Diffusion-DPO before VideoDPO; continue with DenseDPO for temporal localization and Diffusion-APO for trajectory-aware preferences.
4. **Follow the differentiable-reward path.** Read AlignProp and DRaFT before VADER; Diffusion-DRF shows how dense VLM feedback can reduce new reward-data collection.
5. **Study test-time alignment separately.** Free²Guide, diffusion latent beam search, WMReward, and LatSearch change sampling rather than model weights.
6. **Evaluate the whole loop.** Pair a reusable judge such as VideoReward or VideoScore with VBench-2.0, VideoPhy-2, and VideoGen-RewardBench; never report only the optimized reward.

**Scope.** The object of curation is the optimization method. Included works train or steer a video generator with policy gradients, preferences, differentiable rewards, verifiers, or reward-guided search; directly enabling datasets and benchmarks are also included. Pure SFT, generic distillation, captioning, and video editing without a reward or preference objective are excluded. Image-only papers appear only when they are direct methodological foundations. Years denote the first public paper release, usually arXiv v1.

## Concept Primer

| Concept | Working intuition for video generation | Main failure mode to watch |
|---|---|---|
| **Policy gradient** | Treat the denoising, flow, or autoregressive sampling trajectory as a stochastic policy and raise the probability of trajectories with higher episodic reward. | Video rollouts are expensive; terminal rewards give noisy credit and can collapse diversity. |
| **GRPO** | Sample a group of videos for one prompt, normalize rewards within the group, and update from relative advantages without a learned value critic. | Ties, low within-group variance, or a saturated judge can make the learning signal vanish; easy reward shortcuts can dominate. |
| **DPO** | Increase the chosen video's likelihood relative to a rejected video and a frozen reference policy, avoiding explicit online RL. | Pair construction determines what is learned; coarse clip-level pairs can hide local or temporal defects. |
| **Reward backpropagation** | Differentiate a frozen reward through the decoder and some or all sampling steps, producing lower-variance gradients than score-function RL. | Memory cost, unstable long-horizon gradients, and direct exploitation of the differentiable surrogate. |
| **KL control** | Penalize movement from a reference model through explicit KL terms, clipped ratios, reference log-probabilities, or trust regions. | Too little control causes drift and reward hacking; too much prevents meaningful improvement. |
| **Bradley–Terry preference model** | Convert a score difference into the probability that one video is preferred over another; this underlies many pairwise reward models. | Inconsistent raters, ties, and multi-objective preferences violate a simple one-dimensional ranking assumption. |

For diffusion and flow models, “action” may mean a denoising transition, predicted clean sample, velocity, or sampled latent; papers are best compared by **where reward is measured**, **how credit reaches earlier steps**, **how the reference policy is enforced**, and **whether data are offline or generated online**.

## Quick Start by Goal

Start from the failure you need to fix. Training-time methods change the generator; evaluation/inference-time methods score or steer a frozen one.

| Goal | Training-time starting points | Evaluation / inference starting points |
|---|---|---|
| Complex prompt following and compositional control | [DanceGRPO](https://arxiv.org/abs/2505.07818), [Seedance 1.0](https://arxiv.org/abs/2506.09113), [VideoDPO](https://arxiv.org/abs/2412.14167), [OnlineVPO](https://arxiv.org/abs/2412.15159) | [VBench](https://arxiv.org/abs/2311.17982), [VideoScore2](https://arxiv.org/abs/2509.22799) |
| Temporal coherence and long-horizon stability | [DenseDPO](https://arxiv.org/abs/2506.03517), [InfLVG](https://arxiv.org/abs/2505.17574), [TAGRPO](https://arxiv.org/abs/2601.05729), [BranchGRPO](https://arxiv.org/abs/2509.06040) | [VBench](https://arxiv.org/abs/2311.17982), [LatSearch](https://arxiv.org/abs/2603.14526) |
| Subject and identity preservation | [ID-Crafter](https://arxiv.org/abs/2511.00511), [Identity-GRPO](https://arxiv.org/abs/2510.14256), [MagicID](https://arxiv.org/abs/2503.12689), [IPRO](https://arxiv.org/abs/2510.14255) | Identity similarity and temporal-consistency checks |
| Motion quality and physical plausibility | [Phys-AR](https://arxiv.org/abs/2504.15932), [PhysHPO](https://arxiv.org/abs/2508.10858), [PhysCorr](https://arxiv.org/abs/2511.03997), [PhysRVG](https://arxiv.org/abs/2601.11087) | [VideoPhy-2](https://arxiv.org/abs/2503.06800), [VBench-2.0](https://arxiv.org/abs/2503.21755), [WMReward](https://arxiv.org/abs/2601.10553) |
| Camera, geometry, and spatial relations | [CamVerse](https://arxiv.org/abs/2512.02870), [VGGRPO](https://arxiv.org/abs/2603.26599), [Epipolar-DPO](https://arxiv.org/abs/2510.21615), [SPATIALALIGN](https://arxiv.org/abs/2602.22745) | Camera-trajectory error, epipolar consistency, [VBench-2.0](https://arxiv.org/abs/2503.21755) |
| Aesthetics and output diversity | [DPP-GRPO](https://arxiv.org/abs/2511.20647), [SAGE-GRPO](https://arxiv.org/abs/2603.21872), [BPGO](https://arxiv.org/abs/2511.18919) | [VisionReward](https://arxiv.org/abs/2412.21059), [VideoScore](https://arxiv.org/abs/2406.15252), [VBench](https://arxiv.org/abs/2311.17982) |
| Safety and undesirable-output avoidance | [Diffusion-NPO](https://arxiv.org/abs/2505.11245), [VPO](https://arxiv.org/abs/2503.20491) | [SafeSora](https://arxiv.org/abs/2406.14477) and category-specific red-teaming |
| Lower-cost training or test-time scaling | [BranchGRPO](https://arxiv.org/abs/2509.06040), [PRFL](https://arxiv.org/abs/2511.21541), [DOLLAR](https://arxiv.org/abs/2412.15689) | [Diffusion Latent Beam Search](https://arxiv.org/abs/2501.19252), [Video-T1](https://arxiv.org/abs/2503.18942), [LatSearch](https://arxiv.org/abs/2603.14526) |

## Foundations and Optimization Paradigms

- [Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://www.jstor.org/stable/2334029) *(1952)* — Introduces the Bradley–Terry pairwise-comparison model underlying many learned preference rewards.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) *(2017)* — Defines clipped policy updates and trust-region-style control used by later visual RL methods.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) *(2022)* — Establishes the supervised-plus-reward-model-plus-PPO RLHF pipeline.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) *(2022)* — Provides the canonical RLAIF recipe for replacing part of human labeling with model-generated feedback.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) *(2023)* — Turns pairwise preferences into a reference-regularized classification loss without fitting an explicit reward model.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) *(2024)* — Introduces GRPO, whose group-relative advantages remove the learned critic later adapted to visual generation.
- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) *(2023)* — DDPO treats denoising as a multi-step stochastic policy and applies policy gradients to diffusion sampling.
- [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381) *(2023)* — Adds KL-regularized policy optimization for reward-aligned diffusion fine-tuning.
- [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) *(2023)* — Derives a diffusion-native DPO objective that became the principal bridge to VideoDPO-style methods.
- [Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739) *(2023)* — AlignProp differentiates reward through truncated diffusion sampling for low-variance alignment.
- [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) *(2023)* — DRaFT studies direct reward gradients, truncation, and LoRA for memory-efficient diffusion alignment.
- [Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model](https://arxiv.org/abs/2311.13231) *(2023)* — D3PO learns directly from pairwise feedback over diffusion trajectories without a separately trained reward model.
- [Aligning Text-to-Image Models using Human Feedback](https://arxiv.org/abs/2302.12192) *(2023)* — Early demonstration that reward fine-tuning on human feedback aligns text-to-image diffusion, a precursor to video reward alignment.
- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470) *(2025)* — Makes deterministic flow matching explorable through ODE-to-SDE conversion and brings online GRPO to flow-matching generators.

## Reinforcement Learning for Video Generation

### End-to-End and System-Level Post-Training

- [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) *(2025)* — Adapts critic-free group-relative policy optimization to diffusion and rectified-flow image and video generators.
- [Improving Dynamic Object Interactions in Text-to-Video Generation with AI Feedback](https://arxiv.org/abs/2412.02617) *(2024)* — Unifies offline RWR and DPO for video diffusion and finds binary VLM feedback especially effective on difficult object dynamics.
- [Seedance 1.0: Exploring the Boundaries of Video Generation Models](https://arxiv.org/abs/2506.09113) *(2025)* — Combines fine-grained SFT with video-specific RLHF and multidimensional rewards in a production-scale foundation model.
- [Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO](https://arxiv.org/abs/2511.16669) *(2025)* — VANS jointly optimizes a reasoning VLM and video diffusion model under one next-event reward.
- [Reinforcement Learning with Inverse Rewards for World Model Post-training](https://arxiv.org/abs/2509.23958) *(2025)* — RLIR derives training rewards from inverse dynamics to post-train action-conditioned video world models.
- [Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation](https://arxiv.org/abs/2511.14993) *(2025)* — Reports reinforcement-learning post-training across an open family of image and video foundation models.
- [A Systematic Post-Train Framework for Video Generation](https://arxiv.org/abs/2604.25427) *(2026)* — Organizes data refinement, SFT, reward modeling, and video-adapted Flow-GRPO into a staged pipeline.
- [LongCat-Video Technical Report](https://arxiv.org/abs/2510.22200) *(2025)* — Uses multi-reward GRPO with separate motion-quality and text-alignment critics for open video generation.
- [HunyuanVideo 1.5 Technical Report](https://arxiv.org/abs/2511.18870) *(2025)* — Pairs an in-house multidimensional video reward model with online RL and DPO post-training.
- [TeleBoost: A Systematic Alignment Framework for High-Fidelity, Controllable, and Robust Video Generation](https://arxiv.org/abs/2602.07595) *(2026)* — Stages supervised policy shaping, reward-driven RL, and preference refinement under stability constraints.
- [OmniNFT: Modality-wise Omni Diffusion Reinforcement for Joint Audio-Video Generation](https://arxiv.org/abs/2605.12480) *(2026)* — Applies modality-wise diffusion reinforcement to jointly optimize per-modality fidelity and cross-modal synchronization in audio-video generation.
- [Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model](https://arxiv.org/abs/2512.13507) *(2025)* — Extends Seedance's reward-feedback alignment to joint audio-video generation with three RLHF reward models (audio-video alignment, motion, aesthetics) maximized across T2V/T2VA/I2VA.
- [Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model](https://arxiv.org/abs/2504.08685) *(2025)* — Adds Video-DPO on annotator best/worst picks from four generations per prompt, plus a quality/artifact classifier for data filtering.
- [Kling-Omni Technical Report](https://arxiv.org/abs/2512.16776) *(2025)* — Applies multi-round DPO on human preference pairs over sampled video variants, targeting motion dynamics and visual integrity without trajectory sampling.

### Process, Trajectory, Stability, and Efficiency

- [Rethinking Reward Signals in Video GRPO: When Scores Become Targets](https://arxiv.org/abs/2511.19356) *(2025)* — TaRoS converts scalar scores into self-paced target-reaching rewards to preserve useful group variance.
- [BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models](https://arxiv.org/abs/2509.06040) *(2025)* — Shares early denoising prefixes and branches later rollouts to reduce cost while retaining relative comparisons.
- [Diverse Video Generation with Determinantal Point Process-Guided Policy Optimization](https://arxiv.org/abs/2511.20647) *(2025)* — DPP-GRPO rewards set-level diversity rather than optimizing independent samples toward one high-score mode.
- [TAGRPO: Boosting GRPO on Image-to-Video Generation with Direct Trajectory Alignment](https://arxiv.org/abs/2601.05729) *(2026)* — Adds direct trajectory alignment so group-relative updates respect the image-to-video denoising path.
- [OP-GRPO: Efficient Off-Policy GRPO for Flow-Matching Models](https://arxiv.org/abs/2604.04142) *(2026)* — Reuses stale flow trajectories with off-policy correction to improve rollout efficiency.
- [Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization](https://arxiv.org/abs/2605.15980) *(2026)* — Compresses policy optimization to one-step video rollouts for substantially cheaper alignment.
- [DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](https://arxiv.org/abs/2601.20218) *(2026)* — Converts terminal feedback into dense flow-time rewards for better credit assignment.
- [Learning to Credit the Right Steps: Objective-aware Process Optimization for Visual Generation](https://arxiv.org/abs/2604.19234) *(2026)* — Assigns process credit according to which generation steps can affect each objective.
- [Euphonium: Steering Video Flow Matching via Process Reward Gradient Guided Stochastic Dynamics](https://arxiv.org/abs/2602.04928) *(2026)* — Combines process-reward gradients with stochastic flow dynamics for trajectory-level steering.
- [Manifold-Aware Exploration for Reinforcement Learning in Video Generation](https://arxiv.org/abs/2603.21872) *(2026)* — SAGE-GRPO explores along the learned video manifold instead of injecting indiscriminate rollout noise.
- [CreFlow: Corrective Reflow for Sparse-Reward Embodied Video Diffusion RL](https://arxiv.org/abs/2605.14274) *(2026)* — Uses corrective reflow to propagate sparse embodied-task rewards through video diffusion trajectories.
- [AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization](https://arxiv.org/abs/2603.17461) *(2026)* — Introduces contrastive policy updates tailored to autoregressive video token generation.
- [RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO](https://arxiv.org/abs/2605.15190) *(2026)* — Applies GRPO to a consistency-trained autoregressive extrapolator while targeting real-time rollout.
- [KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration](https://arxiv.org/abs/2605.14278) *(2026)* — Explores semantic key–value states rather than pixels for ODE-native group-relative autoregressive video alignment.
- [Astrolabe: Steering Forward-Process Reinforcement Learning for Distilled Autoregressive Video Models](https://arxiv.org/abs/2603.17051) *(2026)* — Moves reinforcement learning into the forward process for efficient, preference-aligned updates of distilled autoregressive video models.
- [MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators](https://arxiv.org/abs/2607.15273) *(2026)* — Transfers forward-process DiffusionNFT reinforcement learning to few-step MeanFlow generators via an induced instantaneous-velocity predictor, with Wan 2.1 video experiments.
- [TempAct: Advancing Temporal Plausibility in Autoregressive Video Generation via Planner-Executor RL](https://arxiv.org/abs/2606.28016) *(2026)* — Jointly optimizes an LLM planner and autoregressive video executor with hierarchical exploration and plan-, video-, and transition-level rewards.
- [ReFree: Towards Realistic Co-Speech Video Generation via Reward-Free RL and Multilevel Speech Guidance](https://arxiv.org/abs/2606.13304) *(2026)* — Introduces reward-free reinforcement learning into flow-matching portrait-video training to suppress implausible head motion without human preferences or a learned reward.
- [Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models](https://arxiv.org/abs/2606.11025) *(2026)* — Replaces PPO-style ratio clipping with an exact Gaussian-KL divergence constraint for stable online RL of flow-based image and video generators.

### Verifiable Rewards and Structured Constraints

- [Reasoning Physical Video Generation with Diffusion Timestep Tokens via Reinforcement Learning](https://arxiv.org/abs/2504.15932) *(2025)* — Phys-AR exposes denoising timesteps as reasoning tokens and optimizes physical plausibility with RL.
- [PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop](https://arxiv.org/abs/2503.09595) *(2025)* — Studies small-data physics post-training, introduces a learned free-fall reward, and releases PisaBench.
- [PhysMaster: Mastering Physical Representation for Video Generation via Reinforcement Learning](https://arxiv.org/abs/2510.13809) *(2025)* — Optimizes internal physical representations rather than relying only on final-frame quality rewards.
- [Taming Camera-Controlled Video Generation with Verifiable Geometry Reward](https://arxiv.org/abs/2512.02870) *(2025)* — CamVerse uses dense segment-level camera-pose errors as an objective geometry reward.
- [Identity-GRPO: Optimizing Multi-Human Identity-preserving Video Generation via Reinforcement Learning](https://arxiv.org/abs/2510.14256) *(2025)* — Builds group-relative identity and quality rewards for multi-person video generation.
- [RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation](https://arxiv.org/abs/2509.16500) *(2025)* — Uses geometric consistency feedback to align driving-world video rollouts.
- [Real-Time Motion-Controllable Autoregressive Video Diffusion](https://arxiv.org/abs/2510.08131) *(2025)* — AR-Drag applies RL with a trajectory reward to few-step, motion-controlled autoregressive video diffusion.
- [ID-Crafter: VLM-Grounded Online RL for Compositional Multi-Subject Video Generation](https://arxiv.org/abs/2511.00511) *(2025)* — Uses VLM-grounded online rewards to preserve multiple identities and their semantic interactions.
- [Wan-R1: Verifiable-Reinforcement Learning for Video Reasoning](https://arxiv.org/abs/2603.27866) *(2026)* — Adapts GRPO to flow video models with executable maze and navigation rewards.
- [PhysRVG: Physics-Aware Unified Reinforcement Learning for Video Generative Models](https://arxiv.org/abs/2601.11087) *(2026)* — Unifies physics-sensitive rewards across video generation settings in one RL framework.
- [VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward](https://arxiv.org/abs/2603.26599) *(2026)* — Uses a 4D latent reward to reinforce scene persistence and world consistency.
- [World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](https://arxiv.org/abs/2604.24764) *(2026)* — Turns reconstructable 3D constraints into verifiable rewards for text-to-video GRPO.
- [Video Models Can Reason with Verifiable Rewards](https://arxiv.org/abs/2605.15458) *(2026)* — Shows that video generators can acquire task reasoning when optimized against objective outcome checks.
- [GeoFlow: Enforcing Implicit Geometric Consistency in Video Generation](https://arxiv.org/abs/2605.18365) *(2026)* — Reinforces implicit multi-frame geometry without requiring explicit camera labels at inference.
- [Geo-Align: Video Generation Alignment via Metric Geometry Reward](https://arxiv.org/abs/2605.23903) *(2026)* — Uses metric reconstruction errors as dense geometry-aware alignment feedback.
- [ReinDriveGen: Reinforcement Post-Training for Out-of-Distribution Driving Scene Generation](https://arxiv.org/abs/2604.01129) *(2026)* — Post-trains a driving generator for rare and distribution-shifted scenes with task-grounded rewards.
- [FlowPortrait: Reinforcement Learning for Audio-Driven Portrait Video Generation](https://arxiv.org/abs/2603.00159) *(2026)* — Optimizes lip synchronization, identity, and motion quality in flow-based portrait animation.
- [EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards](https://arxiv.org/abs/2603.17808) *(2026)* — Uses action recoverability as a verifier for robot-conditioned world-model videos.
- [Reward as An Agent for Embodied World Models](https://arxiv.org/abs/2606.19990) *(2026)* — Couples an agentic verifier with DynDiff-GRPO to broaden embodied rollouts while resisting reward hacking.
- [PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation](https://arxiv.org/abs/2605.14269) *(2026)* — Decomposes human motion into 3D structural constraints that provide interpretable RL feedback.
- [NEWTON: Agentic Planning for Physically Grounded Video Generation](https://arxiv.org/abs/2605.18396) *(2026)* — Casts physically grounded video generation as agentic planning with verifiable physics rewards, targeting VideoPhy-2 failure modes.
- [MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment](https://arxiv.org/abs/2512.06628) *(2025)* — Generates long-horizon robotic-manipulation video hierarchically, aligned with RL-based physical-plausibility rewards.
- [PhyPrompt: RL-based Prompt Refinement for Physically Plausible Text-to-Video Generation](https://arxiv.org/abs/2603.03505) *(2026)* — Refines a prompt LM with GRPO under a dynamic reward curriculum that shifts from semantic fidelity toward physical commonsense scored on generated videos.
- [Improving the Physics of Video Generation with VJEPA-2 Reward Signal](https://arxiv.org/abs/2510.21840) *(2025)* — Uses a VJEPA-2 world-model reward signal to reward-align the physical plausibility of generated video.
- [PAVXploreRL: Physical-Action-Visual World Model Reinforcement Learning with Action Exploration](https://arxiv.org/abs/2607.16602) *(2026)* — Post-trains an action-conditioned latent video world model with physical-plausibility, action-adherence, and visual-fidelity rewards plus noise-driven out-of-distribution action exploration.
- [WorldCompass: Reinforcement Learning for Long-Horizon World Models](https://arxiv.org/abs/2602.09022) *(2026)* — Aligns interactive autoregressive video world models via clip-level rollouts, interaction-following and visual-quality rewards, and negative-aware fine-tuning.

## Preference Optimization (DPO-family)

### Direct Video Preference Objectives

- [VideoDPO: Omni-Preference Alignment for Video Diffusion Generation](https://arxiv.org/abs/2412.14167) *(2024)* — Pioneers video diffusion DPO with automatically constructed pairs balancing visual quality and text alignment.
- [OnlineVPO: Align Video Diffusion Model with Online Video-Centric Preference Optimization](https://arxiv.org/abs/2412.15159) *(2024)* — Refreshes preference pairs online using video-centric rewards to reduce offline-data mismatch.
- [Bridging SFT and DPO for Diffusion Model Alignment with Self-Sampling Preference Optimization](https://arxiv.org/abs/2410.05255) *(2024)* — SePPO mixes self-sampled preferred outputs with supervised anchors to bridge imitation and preference learning.
- [SIPO: Stabilized and Improved Preference Optimization for Aligning Diffusion Models](https://arxiv.org/abs/2505.21893) *(2025)* — Reweights informative denoising timesteps to correct off-policy mismatch and stabilize diffusion preference updates.
- [Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models](https://arxiv.org/abs/2505.11245) *(2025)* — Learns from rejected visual samples directly, including video diffusion experiments, without requiring paired winners.
- [Discriminator-Free Direct Preference Optimization for Video Diffusion](https://arxiv.org/abs/2504.08542) *(2025)* — Constructs preference pairs through controlled corruption instead of training a discriminator or reward model.
- [HuViDPO: Enhancing Video Generation through Direct Preference Optimization for Human-Centric Alignment](https://arxiv.org/abs/2502.01690) *(2025)* — Targets human appearance and motion defects with dedicated preference data.
- [Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](https://arxiv.org/abs/2502.02088) *(2025)* — Alternates preference-data improvement and generator optimization to iteratively strengthen alignment.
- [RealDPO: Real or Not Real, that is the Preference](https://arxiv.org/abs/2510.14955) *(2025)* — Reward-model-free video DPO using real-action clips as chosen and model generations as rejected, with a tailored loss and the RealAction-5K dataset.
- [Hallo4: High-Fidelity Dynamic Portrait Animation via Direct Preference Optimization](https://arxiv.org/abs/2505.23525) *(2025)* — Direct preference optimization on curated human preferences to align lip-sync and expression naturalness in portrait animation.

### Fine-Grained, Structured, and Geometry-Aware Preferences

- [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](https://arxiv.org/abs/2506.03517) *(2025)* — Labels temporally aligned short segments to avoid whole-video motion bias and densify supervision.
- [AlignHuman: Improving Motion and Fidelity via Timestep-Segment Preference Optimization for Audio-Driven Human Animation](https://arxiv.org/abs/2506.11144) *(2025)* — Selects both denoising timesteps and temporal segments to localize animation defects.
- [MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](https://arxiv.org/abs/2503.12689) *(2025)* — Balances identity fidelity against motion preservation through hybrid preference objectives.
- [Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](https://arxiv.org/abs/2412.15156) *(2024)* — Trains an LLM to rewrite prompts using downstream video preferences rather than altering the generator alone.
- [VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](https://arxiv.org/abs/2503.20491) *(2025)* — Trains a DPO prompt rewriter from video-reward comparisons to improve quality and safety without updating the generator.
- [Hierarchical Fine-grained Preference Optimization for Physically Plausible Video Generation](https://arxiv.org/abs/2508.10858) *(2025)* — PhysHPO supplies hierarchical clip- and segment-level physical preferences.
- [RDPO: Real Data Preference Optimization for Physics Consistency Video Generation](https://arxiv.org/abs/2506.18655) *(2025)* — Uses real videos as preferred anchors to improve physical consistency.
- [Epipolar Geometry Improves Video Generation Models](https://arxiv.org/abs/2510.21615) *(2025)* — Epipolar-DPO converts cross-frame geometric agreement into preference supervision.
- [PhysCorr: Dual-Reward DPO for Physics-Constrained Text-to-Video Generation with Automated Preference Selection](https://arxiv.org/abs/2511.03997) *(2025)* — Combines semantic and physics rewards to select pairs and constrain DPO.
- [PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation](https://arxiv.org/abs/2512.24551) *(2025)* — Generalizes pairwise DPO to groupwise physics rankings and releases PhyVidGen-135K.
- [SPATIALALIGN: Aligning Dynamic Spatial Relationships in Video Generation](https://arxiv.org/abs/2602.22745) *(2026)* — Optimizes changing object relations with a geometry-derived dynamic-spatial score.
- [Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models](https://arxiv.org/abs/2601.04068) *(2026)* — LocalDPO confines preference updates to regions and intervals containing the defect.
- [McSc: Motion-Corrective Preference Alignment for Video Generation with Self-Critic Hierarchical Reasoning](https://arxiv.org/abs/2511.22974) *(2025)* — Uses a self-critic to diagnose motion errors and build corrective hierarchical preferences.
- [Beyond Reward Margin: Rethinking and Resolving Likelihood Displacement in Diffusion Models via Video Generation](https://arxiv.org/abs/2511.19049) *(2025)* — PG-DPO explicitly controls winner and loser likelihood movement instead of optimizing only their margin.
- [Diffusion-APO: Trajectory-Aware Direct Preference Alignment for Video Diffusion Transformers](https://arxiv.org/abs/2605.07503) *(2026)* — Aligns full diffusion trajectories rather than treating noisy states as independent preference examples.
- [VERTIGO: Visual Preference Optimization for Cinematic Camera Trajectory Generation](https://arxiv.org/abs/2604.02467) *(2026)* — DPO on a camera-trajectory generator using VLM preferences over rendered previews to improve camera-controlled video.
- [FantasyTalking2: Timestep-Layer Adaptive Preference Optimization for Audio-Driven Portrait Animation](https://arxiv.org/abs/2508.11255) *(2025)* — Pairs a Talking-Critic reward with timestep-layer adaptive preference optimization that fuses decoupled preference experts across denoising steps and layers.
- [When Physical Preferences Meet Semantic Constraints: Physical and Semantic Direct Preference Optimization for Text-to-Video Generation](https://arxiv.org/abs/2607.16947) *(2026)* — PSDPO weights preference pairs by physical–semantic agreement and stages DPO to improve physical plausibility while limiting semantic drift.
- [PhyWorld: Physics-Faithful World Model for Video Generation](https://arxiv.org/abs/2605.19242) *(2026)* — Applies a second-stage DPO over physics preference pairs to align video continuations with physical principles.
- [SyncDPO: Enhancing Temporal Synchronization in Video-Audio Joint Generation via Preference Learning](https://arxiv.org/abs/2605.12179) *(2026)* — Builds rule-based temporally misaligned video–audio negatives on the fly and applies curriculum DPO to improve fine-grained audiovisual synchronization.

### Multi-Objective and Hybrid Preference–RL

- [Calibrated Multi-Preference Optimization for Aligning Diffusion Models](https://arxiv.org/abs/2502.02588) *(2025)* — CaPO calibrates heterogeneous reward scales and selects Pareto-front preference pairs.
- [Multi-Objective Preference Optimization: Improving Human Alignment of Generative Models](https://arxiv.org/abs/2505.10892) *(2025)* — MOPO formalizes constrained optimization when alignment dimensions conflict.
- [Learning What to Trust: Bayesian Prior-Guided Optimization for Visual Generation](https://arxiv.org/abs/2511.18919) *(2025)* — BPGO down-weights uncertain reward dimensions with Bayesian priors and evaluates both image and video generation.
- [MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](https://arxiv.org/abs/2511.20629) *(2025)* — Trains preference-specialized adapters and merges them to move the multi-objective Pareto frontier, including video.
- [V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](https://arxiv.org/abs/2508.03254) *(2025)* — Alternates online preference collection with distillation to align and accelerate video diffusion.
- [Aligning Anime Video Generation with Human Feedback](https://arxiv.org/abs/2504.10044) *(2025)* — Introduces AnimeReward and gap-aware preference optimization for style-specific appearance and temporal consistency.
- [ViPO: Visual Preference Optimization at Scale](https://arxiv.org/abs/2604.24953) *(2026)* — Releases 300K high-resolution preference-labeled video pairs and proposes Poly-DPO for robust optimization on noisy visual preferences.

## Reward Backpropagation and Differentiable-Reward Alignment

- [InstructVideo: Instructing Video Diffusion Models with Human Feedback](https://arxiv.org/abs/2312.12490) *(2023)* — Optimizes text-to-video diffusion with segment-level image rewards and temporal regularization.
- [Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737) *(2024)* — VADER backpropagates dense pixel-space reward gradients through video diffusion rollouts.
- [T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback](https://arxiv.org/abs/2405.18750) *(2024)* — Adds differentiable image and video rewards directly to consistency distillation for fast high-quality sampling.
- [T2V-Turbo-v2: Enhancing Video Generation Model Post-Training through Data, Reward, and Conditional Guidance Design](https://arxiv.org/abs/2410.05677) *(2024)* — Extends mixed-reward consistency training with improved data and conditioning design.
- [Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](https://arxiv.org/abs/2510.14255) *(2025)* — IPRO differentiates identity and video-quality rewards to preserve a reference subject during animation.
- [CamPilot: Improving Camera Control in Video Diffusion Model with Efficient Camera Reward Feedback](https://arxiv.org/abs/2601.16214) *(2026)* — Decodes latents into 3D Gaussians and differentiates a view-consistency reward for camera alignment.
- [DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](https://arxiv.org/abs/2603.12257) *(2026)* — Learns a motion-aware latent identity reward that preserves multiple subjects without VAE decoding.
- [Your Data Manifold is Secretly a Reward Model: Shell-LCC for Text-to-Video Generation](https://arxiv.org/abs/2606.30248) *(2026)* — Turns the high-quality training-data manifold into a dense differentiable latent reward that preserves local detail.
- [MagicPrompt: Ultra-Lightweight Prompt Tuning for Video Generation](https://arxiv.org/abs/2607.14595) *(2026)* — Optimizes tiny attention-embedded soft prompts with dual-space rewards instead of full-model fine-tuning.
- [Diffusion-DRF: Free, Rich, and Differentiable Reward for Video Diffusion Fine-Tuning](https://arxiv.org/abs/2601.04153) *(2026)* — Derives dense differentiable VQA rewards from a frozen VLM without collecting reward-model training data.
- [PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards](https://arxiv.org/abs/2602.01624) *(2026)* — Uses optimal-transport matching to obtain dense annotation-free alignment rewards.
- [GigaVideo-1: Advancing Video Generation via Automatic Feedback with 4 GPU-Hours Fine-Tuning](https://arxiv.org/abs/2506.10639) *(2025)* — Performs lightweight reward-weighted fine-tuning with frozen VLM feedback and a realism constraint.
- [Reward-Forcing: Autoregressive Video Generation with Reward Feedback](https://arxiv.org/abs/2601.16933) *(2026)* — Injects reward feedback into autoregressive training rather than waiting for terminal policy updates.
- [Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation](https://arxiv.org/abs/2512.04678) *(2025)* — Adds reward-weighted distribution matching to distill a causal streaming generator.
- [Stream-R1: Reliability-Perplexity Aware Reward Distillation for Streaming Video Generation](https://arxiv.org/abs/2605.03849) *(2026)* — Balances judge reliability against model perplexity when distilling reward into a streaming generator.
- [Reward Lightning: Fast Video Generation via Homologous Preference Distillation](https://arxiv.org/abs/2607.03960) *(2026)* — Shares latent structure between a preference model and an adversarially distilled few-step generator.
- [Video Consistency Distance: Enhancing Temporal Consistency for Image-to-Video Generation via Reward-Based Fine-Tuning](https://arxiv.org/abs/2510.19193) *(2025)* — Backpropagates a frequency-domain consistency distance to suppress temporal flicker.
- [SHIFT: Motion Alignment in Video Diffusion Models with Adversarial Hybrid Fine-Tuning](https://arxiv.org/abs/2603.17426) *(2026)* — Combines supervised and advantage-weighted updates with pixel-motion rewards and adversarial anti-hacking control.
- [LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment](https://arxiv.org/abs/2412.04814) *(2024)* — Learns from human feedback with a lightweight reward-driven alignment stage for text-to-video generation.
- [On-Policy Adversarial Flow Distillation for Autoregressive Video Generation](https://arxiv.org/abs/2605.26105) *(2026)* — Trains a Bradley–Terry discriminator on paired teacher/student videos and converts its on-policy advantage into dense forward-process flow-matching updates.

## Reward Models and Verifiers for RL

Only rewards explicitly designed or demonstrated to train, steer, or verify a video generator are listed here. For the full reward-model taxonomy, datasets, judges, and evaluator literature, use [Awesome Reward Models for Video Generation](https://github.com/chrisliu298/awesome-rm-for-video-generation).

- [Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918) *(2025)* — Introduces VideoReward, VideoGen-RewardBench, Flow-DPO, Flow-RWR, and inference-time Flow-NRG.
- [VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation](https://arxiv.org/abs/2406.15252) *(2024)* — Learns multidimensional human video scores intended as an RLHF proxy reward.
- [VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation](https://arxiv.org/abs/2412.21059) *(2024)* — Produces interpretable dimension-level rewards for consistent multi-objective preference optimization.
- [VideoScore2: Think before You Score in Generative Video Evaluation](https://arxiv.org/abs/2509.22799) *(2025)* — Adds reasoned, GRPO-trained multidimensional judgments and supports reward-based best-of-N sampling.
- [RewardDance: Reward Scaling in Visual Generation](https://arxiv.org/abs/2509.08826) *(2025)* — Scales generative visual reward models while preserving reward variance and resisting policy reward hacking.
- [PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation](https://arxiv.org/abs/2411.17048) *(2024)* — Uses identity, dynamics, and semantic rewards to align personalized video generation.
- [Human detectors are surprisingly powerful reward models](https://arxiv.org/abs/2601.14037) *(2026)* — HuDA combines detection confidence and temporal prompt alignment to drive GRPO for complex human and animal motion.
- [What about gravity in video generation? Post-Training Newton's Laws with Verifiable Rewards](https://arxiv.org/abs/2512.00425) *(2025)* — NewtonRewards provides programmatic gravity checks for physics-focused post-training.
- [SoliReward: Mitigating Susceptibility to Reward Hacking and Annotation Noise in Video Generation Reward Models](https://arxiv.org/abs/2512.22170) *(2025)* — Trains a tie-aware video reward model designed to resist noisy labels and reward hacking.
- [Video Generation Models Are Good Latent Reward Models](https://arxiv.org/abs/2511.21541) *(2025)* — Extracts a process-aware latent reward from the generator itself and uses it for feedback learning.
- [AesRM: Improving Video Aesthetics with Expert-Level Feedback](https://arxiv.org/abs/2604.28078) *(2026)* — Builds expert-supervised aesthetic rewards and demonstrates SFT, GRPO, and process-reward use.
- [CaC: Advancing Video Reward Models via Hierarchical Spatiotemporal Concentrating](https://arxiv.org/abs/2605.11723) *(2026)* — Localizes temporal then spatial defects to produce anomaly-aware rewards.
- [Harness Local Rewards for Global Benefits: Effective Text-to-Video Generation Alignment with Patch-level Reward Models](https://arxiv.org/abs/2502.06812) *(2025)* — Supplies patch-level rewards for defects that global video scores average away.
- [Through the PRISM: Preference Representation in Intermediate States of Video Diffusion Models](https://arxiv.org/abs/2606.20310) *(2026)* — Trains a latent video reward model — a query head over a frozen diffusion backbone's noisy intermediate states — for noise-robust pre-decode Best-of-N selection.
- [Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling](https://arxiv.org/abs/2605.05922) *(2026)* — DeScore separates chain-of-thought video assessment from scalar reward prediction and uses dual-objective RL to improve reasoning and reward calibration.
- [GT-SVJ: Generative-Transformer-Based Self-Supervised Video Judge for Efficient Video Reward Modeling](https://arxiv.org/abs/2602.05202) *(2026)* — Reformulates a video generative transformer as an energy-based reward model trained with contrastive latent perturbations that expose temporal defects.

## Inference-Time and Test-Time Alignment

These methods optimize selection or guidance at sampling time rather than—or in addition to—changing generator weights. Generic editing or agent pipelines such as DragVideo, MotionAgent, RACCooN, and DFVEdit are intentionally excluded unless a reward, preference, or verifier drives the refinement.

- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) *(2022)* — Provides the standard conditional-versus-unconditional guidance baseline for post-training-free control.
- [Activation Steering of Video Generation Models via Reduced-Order Linear Optimal Control](https://arxiv.org/abs/2606.04775) *(2026)* — Applies closed-loop latent LQR interventions for minimally invasive concept and safety steering during sampling.
- [Free²Guide: Training-Free Text-to-Video Alignment using Image LVLM](https://arxiv.org/abs/2411.17041) *(2024)* — Uses path-integral control to steer diffusion with black-box LVLM rewards and no gradient or fine-tuning.
- [DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](https://arxiv.org/abs/2412.15689) *(2024)* — Combines few-step distillation with latent reward optimization to recover quality at low sampling cost.
- [Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search](https://arxiv.org/abs/2501.19252) *(2025)* — Expands and prunes latent denoising branches using reward scores during sampling.
- [ScalingNoise: Scaling Inference-Time Search for Generating Infinite Videos](https://arxiv.org/abs/2503.16400) *(2025)* — Searches noise candidates at each continuation boundary to reduce long-horizon drift.
- [Video-T1: Test-Time Scaling for Video Generation](https://arxiv.org/abs/2503.18942) *(2025)* — Allocates additional sampling and evaluation compute to select or refine stronger video candidates.
- [InfLVG: Reinforce Inference-Time Consistent Long Video Generation with GRPO](https://arxiv.org/abs/2505.17574) *(2025)* — Trains a GRPO context-selection policy that preserves relevant history across long autoregressive rollouts.
- [VISTA: A Test-Time Self-Improving Video Generation Agent](https://arxiv.org/abs/2510.15831) *(2025)* — Runs pairwise tournaments and specialist critiques to iteratively revise prompts and regenerate candidates.
- [Inference-time Physics Alignment of Video Generative Models with Latent World Models](https://arxiv.org/abs/2601.10553) *(2026)* — WMReward scores predicted dynamics in a latent world model and guides generation without retraining.
- [VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment](https://arxiv.org/abs/2603.16271) *(2026)* — Uses cross-frame 3D reprojection error as both an RL reward and a test-time path verifier.
- [LatSearch: Latent Reward-Guided Search for Faster Inference-Time Scaling in Video Diffusion](https://arxiv.org/abs/2603.14526) *(2026)* — Searches compact latent states with learned rewards to lower test-time scaling cost.
- [Proprio: Latent Self-Scoring and Inference-Time Refinement for Physically Plausible Video Generation](https://arxiv.org/abs/2605.28230) *(2026)* — Lets the generator score its own latent dynamics and iteratively repair implausible samples.
- [Self-Refining Video Sampling](https://arxiv.org/abs/2601.18577) *(2026)* — Feeds detected defects back into the sampler for iterative, training-free correction.
- [Bootstrapping Physics-Grounded Video Generation through VLM-Guided Iterative Self-Refinement](https://arxiv.org/abs/2511.20280) *(2025)* — Bootstraps physically grounded video generation with VLM-guided iterative self-refinement at inference.
- [Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation](https://arxiv.org/abs/2607.15849) *(2026)* — TANGO uses the diffusion model as a self-critic and optimizes a predicted-noise objective at test time to steer autoregressive rollouts away from terminal trajectories.
- [Plan-and-Verify Video Reward Reasoning with Spatio-Temporal Scene Graph Grounding](https://arxiv.org/abs/2606.11838) *(2026)* — SG-PVR verifies atomic prompt claims against a persistent spatiotemporal scene graph and uses the reward judgments to rerank text-to-video samples.
- [Stream-T1: Test-Time Scaling for Streaming Video Generation](https://arxiv.org/abs/2605.04461) *(2026)* — Performs streaming test-time search with reward-based candidate pruning and reward-guided memory updates for local quality and long-range coherence.
- [Improving Motion in Image-to-Video Models via Adaptive Low-Pass Guidance](https://arxiv.org/abs/2506.08456) *(2025)* — ALG adapts frequency-filtered guidance over time to increase motion without destabilizing appearance.
- [Causally Steered Diffusion for Automated Video Counterfactual Generation](https://arxiv.org/abs/2506.14404) *(2025)* — CSVC steers diffusion toward counterfactual outcomes using causal verification at inference.
- [Think Before You Diffuse: Infusing Physical Rules into Video Diffusion](https://arxiv.org/abs/2505.21653) *(2025)* — DiffPhy converts explicit physical rules into planning and guidance signals before denoising.

## Datasets and Benchmarks

### Datasets

| Name | Year | Alignment use | Links |
|---|---:|---|---|
| **SafeSora** | 2024 | Human safety preferences and harmlessness labels for text-to-video alignment. | [Paper](https://arxiv.org/abs/2406.14477) · [Dataset](https://huggingface.co/datasets/PKU-Alignment/SafeSora) |
| **WISA-80K** | 2025 | Human-curated videos spanning explicit laws in dynamics, thermodynamics, and optics. | [Paper](https://arxiv.org/abs/2503.08153) · [Dataset](https://huggingface.co/datasets/qihoo360/WISA-80K) |
| **PhyWorld** | 2024 | Physical-law prompts, videos, and annotations for training and testing world-consistent generators. | [Paper](https://arxiv.org/abs/2411.02385) · [Project](https://phyworld.github.io/) |
| **VideoDPO Preference Data** | 2024 | Automatically scored preferred/rejected video pairs over quality and prompt alignment. | [Paper / Data](https://arxiv.org/abs/2412.14167) |
| **VideoGen-HF** | 2025 | Multidimensional human video preferences used to train VideoReward and test flow alignment. | [Paper / Project](https://arxiv.org/abs/2501.13918) |
| **VideoFeedback** | 2024 | Fine-grained human ratings across visual quality, temporal quality, and text alignment. | [Paper](https://arxiv.org/abs/2406.15252) |
| **VideoFeedback2** | 2025 | Reasoning-oriented preference and score supervision for VideoScore2. | [Paper](https://arxiv.org/abs/2509.22799) |
| **OpenS2V-5M** | 2025 | Million-scale subject-to-video training pairs with subject and motion coverage. | [Paper / Project](https://arxiv.org/abs/2505.20292) |
| **GRADEO-Instruct** | 2025 | Instruction and reasoning traces for human-like generative-video evaluation. | [Paper](https://arxiv.org/abs/2503.02341) |
| **AnimeReward-30K** | 2025 | Human annotations of anime appearance and temporal consistency preferences. | [Paper / Data](https://arxiv.org/abs/2504.10044) |
| **VANS-Data-100K** | 2025 | Next-video-event reasoning and generation examples for Joint-GRPO. | [Paper / Code](https://arxiv.org/abs/2511.16669) |
| **PhyVidGen-135K** | 2025 | Groupwise physics-preference data released with PhyGDPO. | [Paper / Data](https://arxiv.org/abs/2512.24551) |

### Benchmarks and Learned Evaluators

| Name | Year | Alignment use | Links |
|---|---:|---|---|
| **VBench** | 2023 | General video quality and text-conditioned generation across multiple dimensions. | [Paper](https://arxiv.org/abs/2311.17982) · [Project](https://vchitect.github.io/VBench-project/) |
| **VBench-2.0** | 2025 | Intrinsic faithfulness, including physics, commonsense, and compositional consistency. | [Paper](https://arxiv.org/abs/2503.21755) |
| **VideoPhy** | 2024 | Physical commonsense evaluation for text-to-video generation. | [Paper](https://arxiv.org/abs/2406.03520) |
| **VideoPhy-2** | 2025 | Harder physics evaluation with broader phenomena and diagnostic scoring. | [Paper](https://arxiv.org/abs/2503.06800) |
| **VideoGen-RewardBench** | 2025 | Pairwise benchmark for video reward models used by alignment algorithms. | [Paper / Leaderboard](https://arxiv.org/abs/2501.13918) |
| **VideoScore-Bench** | 2024 | Fine-grained human-correlated scoring across video-generation dimensions. | [Paper](https://arxiv.org/abs/2406.15252) |
| **VideoScore2-Bench** | 2025 | Reasoning-based and interpretable evaluation of generated videos. | [Paper](https://arxiv.org/abs/2509.22799) |
| **WorldSimBench** | 2024 | Tests whether generated videos behave as usable world simulations. | [Paper](https://arxiv.org/abs/2410.18072) |
| **WorldModelBench** | 2025 | Judges video generators as world models using perceptual and action-aware criteria. | [Paper](https://arxiv.org/abs/2502.20694) |
| **EvalCrafter** | 2023 | Comprehensive automatic and human evaluation suite for text-to-video models. | [Paper](https://arxiv.org/abs/2310.11440) |
| **FETV** | 2023 | Fine-grained benchmark for text-to-video generation quality and text alignment. | [Paper](https://arxiv.org/abs/2311.01813) |
| **Video-Bench** | 2025 | Broad benchmark suite for diagnosing modern video generators and evaluators. | [Paper](https://arxiv.org/abs/2504.04907) |

A benchmark qualifies here when it directly measures alignment-relevant behavior or is commonly used as a reward, verifier, pair selector, or anti-reward-hacking check.

## Project Pages, Repos, and Useful Links

### Implementations and project pages

- [DanceGRPO](https://dancegrpo.github.io/) — project page, code, and checkpoints for GRPO on visual generation.
- [VADER](https://vader-vid.github.io/) — reward-gradient video alignment code and results.
- [VideoDPO](https://videodpo.github.io/) — project page and OmniScore preference-data pipeline.
- [VideoAlign / VideoReward](https://gongyeliu.github.io/videoalign/) — reward model, alignment algorithms, and benchmark.
- [DDPO PyTorch](https://github.com/kvablack/ddpo-pytorch) — reference implementation for diffusion policy optimization.
- [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) — official diffusion-DPO implementation.
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — open infrastructure with video diffusion training and post-training support.

### Adjacent collections

- [Awesome Reward Models for Video Generation](https://github.com/chrisliu298/awesome-rm-for-video-generation) — the sibling list with deeper reward-model, judge, and evaluator coverage.
- [Awesome Video Generation Post-Training](https://github.com/people-robots/Awesome-Video-Generation-Post-Training) — broad survey companion spanning SFT, distillation, preference optimization, and inference-time alignment.
- [Awesome RL for Video Generation (legacy)](https://github.com/wendell0218/Awesome-RL-for-Video-Generation) — useful recall source; verify its automated metadata against primary papers.
- [Awesome Video Diffusion](https://github.com/showlab/Awesome-Video-Diffusion) — broad video-diffusion bibliography.
- [Awesome Video Generation](https://github.com/AlonzoLeeeooo/awesome-video-generation) — general models, datasets, and evaluations.
- [Awesome Video Generation Tools](https://github.com/backblaze-labs/awesome-video-generation) — practitioner-oriented video-generation resources.

## Contributing

Contributions are welcome; read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

Inclusion criterion: the work must define, train, apply, or directly enable a reward-, preference-, verifier-, or RL-driven optimization method for video generation.

## Citation

```bibtex
@software{awesome-rl-for-video-generation,
  title = {{Awesome RL for Video Generation}},
  author = {Liu, Chris Yuhao and others},
  year = {2026},
  doi = {10.5281/zenodo.21483924},
  url = {https://github.com/chrisliu298/awesome-rl-for-video-generation},
  version = {v1.0.0}
}
```

---

*Repository last updated: 2026-07-22. Coverage: reinforcement learning, preference optimization, GRPO / flow-GRPO, differentiable-reward post-training, reward-guided inference, reward models and verifiers that drive RL, and enabling datasets and benchmarks for video generation. Years denote first public preprint release.*