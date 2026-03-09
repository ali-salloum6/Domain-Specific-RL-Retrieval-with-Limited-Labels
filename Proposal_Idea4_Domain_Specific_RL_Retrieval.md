# Research Proposal: Domain-Specific RL Retrieval with Limited Labels

**Joint RL + IR project.** Sample-efficient reinforcement learning for information retrieval in high-stakes domains (legal, medical) where labeled relevance is scarce.

---

## 1. Problem & Motivation

In **legal** and **medical** domains, retrieval quality directly affects decisions and risk: missing relevant precedents or misretrieving clinical evidence can have serious consequences. Three obstacles stand in the way:

- **Scarce labels:** Expert relevance judgments are expensive and slow to obtain.
- **Weak off-the-shelf performance:** General-purpose retrievers are not tuned for domain terminology, document structure, or task-specific success (e.g. citation correctness, downstream QA accuracy).
- **Non-differentiable objectives:** Metrics such as recall@k or end-task accuracy cannot be optimized with standard supervised learning alone.

**Evidence of the gap:** On **CLERC** (U.S. legal case retrieval), zero-shot IR reaches only **48.3% recall@1000** [1](https://arxiv.org/abs/2406.17186). **LegalBench-RAG** and **CLERC** are the first benchmarks that evaluate the *retrieval* component of RAG in law, stressing minimal relevant snippets rather than document-level retrieval [2](https://arxiv.org/abs/2408.10343), [3](https://reglab.github.io/legal-rag-benchmarks/). **D3LM** uses positive-unlabeled RL for legal *consultation* (diagnostic questions, court views) but does not optimize the *retrieval* module under limited labels [4](https://aclanthology.org/2024.findings-acl.918/). In medicine, **MIRAGE** shows that retrieval design strongly affects LLM accuracy, but RL for *retrieval* with limited labels remains underexplored [5](https://aclanthology.org/2024.findings-acl.372/).

We therefore focus on **sample-efficient RL for retrieval** in these high-stakes domains: adapting retrieval models to task-specific goals when relevance labels are limited, using public legal (and optionally medical) benchmarks.

---

## 2. Data

We will use public benchmarks to ensure reproducibility and comparability.

**Primary (legal):**

- **LegalBench-RAG** [2](https://arxiv.org/abs/2408.10343): 6,858 query–answer pairs over 79M+ characters, human-annotated by legal experts; evaluates precise retrieval of minimal relevant text. Includes LegalBench-RAG-mini for fast iteration.
  - [arXiv:2408.10343](https://arxiv.org/abs/2408.10343) | [GitHub](https://github.com/zeroentropy-ai/legalbenchrag)

- **CLERC** [1](https://arxiv.org/abs/2406.17186): U.S. legal case retrieval and retrieval-augmented analysis generation; supports IR (finding citations for analysis) and RAG (generating analyses from citations).
  - [arXiv:2406.17186](https://arxiv.org/abs/2406.17186) | [ACL Anthology](https://aclanthology.org/2025.findings-naacl.441/) | [GitHub](https://github.com/bohanhou14/CLERC)

**Optional (medical):** MIRAGE / MedRAG setting [5](https://aclanthology.org/2024.findings-acl.372/): 7,663 questions from five medical QA datasets; we would adopt the same retrieval evaluation protocol if scope allows.
  - [ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.372/)

**Data splits:** We will use official train/validation/test splits where provided, and for sample-efficiency experiments we will subsample training labels (e.g. 10%, 25%, 50%, 100%) with fixed validation/test sets.

---

## 3. Methods

**Baseline approach:** We adapt the **ICLR 2024** framework of [6](https://openreview.net/forum?id=xThb6APBoG): policy-gradient RL to optimize pretrained retrieval models for non-differentiable task-specific objectives (recall, MRR, or downstream RAG/QA accuracy). That work reduces the large action space to binary decisions per query–item pair and uses auxiliary retrievers for exploration; it is directly applicable to legal/medical IR by treating benchmark metrics as rewards.

**Our adaptations:**

- **Reward:** Reward is defined from retrieval metrics (e.g. recall@k, MRR, nDCG) and/or downstream task accuracy (QA, citation correctness) on the chosen benchmark. When full relevance labels are scarce, we will experiment with bandit-style or preference-based feedback (e.g. pairwise comparisons) to improve sample efficiency.

- **Retriever initialization:** We start from a pretrained retriever (general-purpose or domain-finetuned if available) and adapt it with RL on the target benchmark.

- **Sample-efficiency focus:** We will ablate over the fraction of labeled training data and report performance curves; we may use data augmentation or synthetic queries where consistent with the benchmark license.

**Joint IR+RL roles:** IR contributes corpus design, relevance criteria, and evaluation metrics; RL contributes policy optimization, reward design, and sample-efficient learning under scarce feedback.

---

## 4. Theory Link

- **RL:** The method rests on **policy gradient** optimization of a stochastic policy (retrieval rankings) with respect to a reward that is a function of task-specific metrics. Sample efficiency connects to **exploration–exploitation** (e.g. entropy regularization, auxiliary retrievers as in [6](https://openreview.net/forum?id=xThb6APBoG)) and to **credit assignment** in multi-step or listwise settings. If we use limited or noisy feedback, **contextual bandits** and **preference-based RL** (e.g. learning from pairwise comparisons) provide a theoretical basis for learning from partial information.

- **IR:** Relevance is formalized in **probability ranking principle** and **learning-to-rank** (LTR); our reward functions align with standard IR metrics (recall@k, MRR, nDCG). The project bridges LTR (supervised from relevance labels) and **reinforcement learning for ranking**, where the objective is non-differentiable or labels are scarce—a setting that has been studied in general retrieval [6](https://openreview.net/forum?id=xThb6APBoG) but not yet systematically in legal/medical domains.

---

## 5. Evaluation Plan

**Primary metrics:** Retrieval quality on held-out test sets: **recall@k** (k = 1, 5, 10, 100), **MRR**, **nDCG**. Where the benchmark defines a downstream task (QA, citation), we also report **downstream accuracy** (e.g. exact match, F1, or benchmark-specific metric).

**Baselines:** (1) Zero-shot pretrained retriever; (2) Supervised fine-tuning on full (or same subsampled) labels when applicable; (3) Existing legal/medical retrieval or RAG results reported on LegalBench-RAG and CLERC.

**Ablations:** (1) **Sample efficiency:** performance vs. fraction of training labels (e.g. 10%, 25%, 50%, 100%); (2) **Reward choice:** retrieval-only vs. retrieval + downstream reward; (3) **Initialization:** general vs. domain-adapted retriever.

**Protocol:** Fixed random seeds and data splits; report mean and standard deviation over multiple runs where feasible. We will release code and configs to reproduce results.

---

## 6. Risks

| Risk | Mitigation |
|------|-------------|
| **Too few labels to learn reliably** | Use LegalBench-RAG-mini and CLERC for iteration; focus on sample-efficiency ablations as a contribution; consider preference/bandit feedback. |
| **Reward design (e.g. gaming recall)** | Align reward with downstream task where possible; monitor for degenerate rankings; validate on held-out downstream metrics. |
| **Compute and runtime** | Start with LegalBench-RAG-mini and smaller retriever; scale to full benchmark once pipeline is stable. |
| **Domain shift (legal vs. medical)** | Primary scope is legal (LegalBench-RAG, CLERC); medical extension only if time permits, with clear scope limit. |
| **Reproducibility of baselines** | Use public code and checkpoints where available; document all data splits and hyperparameters. |

---

## 7. Work Plan

| Phase | Activities | Output |
|-------|------------|--------|
| **1. Setup (weeks 1–3)** | Acquire LegalBench-RAG and CLERC; set up retriever baseline and evaluation pipeline; reproduce zero-shot and (if available) supervised numbers. | Data loaded; baseline results on dev set. |
| **2. RL adaptation (weeks 4–8)** | Implement or adapt policy-gradient RL from [6](https://openreview.net/forum?id=xThb6APBoG); define reward from benchmark metrics; train on LegalBench-RAG-mini then full (or CLERC). | RL-adapted retriever; initial results. |
| **3. Sample efficiency & ablations (weeks 9–11)** | Subsample training labels (10%, 25%, 50%, 100%); run reward and initialization ablations; collect sample-efficiency curves. | Ablation tables and figures. |
| **4. Write-up & release (weeks 12–14)** | Finalize experiments; write report/paper; release code and configs. | Draft paper; public code. |

*Total: ~14 weeks. Medical benchmark (if in scope) can be added in parallel or as a short extension after Phase 3.*

---

## References (with Links)

[1](https://arxiv.org/abs/2406.17186) **CLERC: A Dataset for U.S. Legal Case Retrieval and Retrieval-Augmented Analysis Generation.** arXiv:2406.17186. NAACL 2025 Findings. [arXiv](https://arxiv.org/abs/2406.17186) | [ACL](https://aclanthology.org/2025.findings-naacl.441/) | [GitHub](https://github.com/bohanhou14/CLERC)

[2](https://arxiv.org/abs/2408.10343) **LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain.** arXiv:2408.10343. [arXiv](https://arxiv.org/abs/2408.10343) | [GitHub](https://github.com/zeroentropy-ai/legalbenchrag)

[3](https://reglab.github.io/legal-rag-benchmarks/) **A Reasoning-Focused Legal Retrieval Benchmark.** [RegLab](https://reglab.github.io/legal-rag-benchmarks/)

[4](https://aclanthology.org/2024.findings-acl.918/) **D3LM: Knowledge-Infused Legal Wisdom... Positive-Unlabeled Reinforcement Learning.** Wu et al. Findings of ACL 2024. [ACL](https://aclanthology.org/2024.findings-acl.918/)

[5](https://aclanthology.org/2024.findings-acl.372/) **Benchmarking Retrieval-Augmented Generation for Medicine (MIRAGE).** Xiong et al. Findings of ACL 2024. [ACL](https://aclanthology.org/2024.findings-acl.372/)

[6](https://openreview.net/forum?id=xThb6APBoG) **Adapting Retrieval Models to Task-Specific Goals using Reinforcement Learning.** ICLR 2024. [OpenReview](https://openreview.net/forum?id=xThb6APBoG) | [arXiv](https://arxiv.org/abs/2403.10131)
