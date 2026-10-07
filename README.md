# Llama3.1-Embedding

**An independent, open-source sentence/document embedding architecture built on
Meta's Llama 3.1 base model.**

This is **not** "Llama with the logits renamed". It is a real embedding stack:
mask-aware pooling, a learned projection head, L2 normalisation, an InfoNCE
contrastive training pipeline with parameter-efficient modes, a retrieval
evaluation suite, and a reproducible project structure.

> ### Current status — please read
>
> This repository currently contains **the architecture, training and evaluation
> framework**. It is *not* yet a trained model.
>
> * No trained checkpoint has been released.
> * No benchmark has been run. The table below says **NOT YET BENCHMARKED** and
>   it stays that way until real numbers exist.
> * No claim is made — and none should be inferred — that this outperforms
>   OpenAI, BGE, E5, GTE, Sentence-Transformers or any other embedding model.
> * Full-scale training of Llama 3.1 8B requires GPU hardware that a phone
>   cannot provide. See [Limitations](#limitations).

---

## Contents

- [What is Llama3.1-Embedding?](#what-is-llama31-embedding)
- [Architecture](#architecture)
- [Why use Llama hidden representations?](#why-use-llama-hidden-representations)
- [Pooling](#pooling)
- [Projection head](#projection-head)
- [Training modes](#training-modes)
- [Contrastive learning](#contrastive-learning)
- [Installation](#installation)
- [Model access (gated)](#model-access-gated)
- [Training](#training)
- [Encoding](#encoding)
- [Semantic search](#semantic-search)
- [Evaluation](#evaluation)
- [Baseline comparison](#baseline-comparison)
- [Benchmarks](#benchmarks)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License and attribution](#license-and-attribution)

---

## What is Llama3.1-Embedding?

A sentence/document encoder that turns text into fixed-size dense vectors for
semantic search, clustering, deduplication and retrieval-augmented generation.

```
texts -> [N, D] vectors,  D in {256, 384, 512, 768, 1024},  default 768
       with ||v||_2 == 1
```

```python
from llama_embedding import LlamaEmbeddingModel

model = LlamaEmbeddingModel.from_pretrained("meta-llama/Llama-3.1-8B")
vectors = model.encode([
    "A search and rescue drone",
    "An unmanned aircraft used for emergency response",
])
vectors.shape          # torch.Size([2, 768])
```

---

## Architecture

```
Input Text
   ↓
Tokenizer                     (Llama 3.1 BPE, used exactly as shipped)
   ↓
Llama 3.1 Base Transformer     (hidden states, NOT logits)
   ↓
Last Hidden States             [B, T, 4096]
   ↓
Attention-Mask-Aware Pooling   [B, 4096]
   ↓
Embedding Projection           Linear 4096 → D
   ↓
LayerNorm                      (configurable)
   ↓
L2 Normalisation               ‖v‖₂ = 1
   ↓
Final Dense Embedding          [B, D]
   ↓
Cosine Similarity / Vector Search
```

Full diagrams, including the training branch, are in
[`docs/architecture.md`](docs/architecture.md).

**The language-model head is never used.** The backbone is loaded with
`AutoModel` (not `AutoModelForCausalLM`) and the embedding is built from
`last_hidden_state`. The 128k-way vocabulary projection is skipped entirely,
which also removes a large activation cost from every forward pass.

---

## Why use Llama hidden representations?

* **Strong multilingual text priors.** Llama 3.1 was pretrained on a large
  multilingual corpus, so its contextual states encode far more than
  surface lexical overlap.
* **Bidirectional context within the window.** Unlike an embedding model trained
  only with a causal objective, every token in a sequence sees the full context
  at its position, which makes mean pooling well behaved.
* **Flexible dimensionality.** A projection head maps 4096 → any supported D,
  so you can trade accuracy for storage and latency per your index.
* **Parameter-efficient adaptation.** The backbone does not need to be
  fine-tuned to produce useful retrieval geometry (see
  [Training modes](#training-modes)).

The honest counterweight: a raw LLM backbone is **not** an embedding model. Its
hidden states are tuned for next-token prediction, so without pooling and a
trained projection the cosine geometry is poor. That gap is exactly what
[`scripts/benchmark.py`](#baseline-comparison) measures.

---

## Pooling

Pooling is one abstraction (`PoolingStrategy`) used everywhere; no module
duplicates the reduction logic.

| Strategy | Behaviour |
|---|---|
| `mean` *(default)* | `sum_t m_t·h_t / sum_t m_t` |
| `last_token` | Hidden state of the last non-padding token |
| `weighted_mean` | Per-token weighted mean; weights supplied by the caller |

Every strategy is **attention-mask aware**. Padding tokens are multiplied out
before summation, so they cannot influence the vector even if their hidden
states are arbitrary. This is asserted directly in
[`tests/test_pooling.py`](tests/test_pooling.py), which poisons padded slots
with `1e4` and checks the output is unchanged.

---

## Projection head

```
pooled [B, 4096]
   → Linear(4096 → D)
   → (optional activation)
   → LayerNorm          ← configurable
   → L2 normalise       ← configurable
```

`EmbeddingHead` is small (~3.1M parameters at `D=768`, about 0.04 % of the 8B
model) and completely independent of the backbone, so it can be trained alone,
exported alone, and swapped without touching Meta's weights.

Outputs satisfy `‖v‖₂ ≈ 1` — there is a unit test asserting exactly that.

---

## Training modes

Three modes, selected by `training.mode`:

| Mode | Backbone | Trainable | Trainable params (8B, D=768) | Memory |
|---|---|---|---|---|
| **`projection_only`** *(default)* | Frozen | head only | ~3.1 M (≈ 0.039 %) | lowest |
| **`lora`** | Frozen + LoRA on `q,k,v,o` | LoRA + head | ~0.1–0.2 % + head | moderate |
| `full` | Trainable | everything | ~8 B | extreme |

**Full fine-tuning is never the default and is never selected implicitly.** The
trainer raises an error if `projection_only` or `lora` ever leaves more than
50 % of parameters trainable.

Before training starts, the trainer prints:

```
====================================================================
Parameter efficiency
====================================================================
  mode                 : projection_only
  trainable parameters : 3,147,264
  total parameters     : 8,033,408,512
  percent trainable    : 0.0391%
  largest trainable modules:
    - head.linear: 3,145,728
    - head.layer_norm: 1,536
====================================================================
```

---

## Contrastive learning

InfoNCE over an in-batch similarity block:

```
L = -1/B · Σ_i log( exp(sim(q_i, d_i)/τ) / Σ_j exp(sim(q_i, d_j)/τ) )
```

* Similarity: **cosine** by default (`dot` and `euclidean` available).
* Negatives: all off-diagonal in-batch entries, plus explicit hard negatives
  from the dataset (`negative` / `hard_negatives` fields).
* Temperature `τ`: **default 0.02** — a documented software default (the value
  used as a default by widely deployed sentence-embedding implementations), and
  **tunable**, not claimed optimal.
* Symmetric (query→doc and doc→query) averaging is available via
  `training.symmetric_loss`.

Stability controls, all in `training/trainer.py`:

* gradient clipping (`max_grad_norm`)
* gradient accumulation
* mixed precision (`fp16` / `bf16` / off)
* gradient checkpointing
* checkpointing, resume, best-checkpoint tracking
* seeded data order and sampling
* **NaN/Inf detection that aborts the run** — training never silently continues
  with a corrupted state

---

## Installation

```bash
git clone https://github.com/burkiuze/Llama3.1-Embedding.git
cd Llama3.1-Embedding
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt          # torch, transformers
pip install -e .                         # install the package
pip install "peft>=0.11"                 # only for mode: lora
```

Optional extras: `numpy` (saving `.npy`), `faiss-cpu` / `qdrant-client` /
`chromadb` (vector stores), `mteb` (benchmark harness), `pyyaml`.

Run the tests (they never download the 8B model — they use a tiny stub
backbone and synthetic tensors):

```bash
pytest -q
python -m compileall .
```

---

## Model access (gated)

**Llama 3.1 weights are not included in this repository and cannot be
downloaded without Meta's approval.** This project does not bundle, mirror or
bypass that gate.

1. Request access: <https://www.llama.com/llama3_1/>
2. Accept the Llama 3.1 Community Licence.
3. Authenticate **locally**:

   ```bash
   huggingface-cli login          # interactive, stores a token in ~/.cache/huggingface
   # or
   export HF_TOKEN=hf_...         # keep it in your shell / secret manager, never in git
   ```

Use the **base** model, `meta-llama/Llama-3.1-8B` (or `-70B`), **not** the
Instruct variant.

Tokens are never committed: `.gitignore` excludes `.env`, `*.token`, `token`,
`hf_token` and the Hugging Face cache, and `EmbeddingConfig.to_dict()`
explicitly nulls the token before anything is serialised.

---

## Training

Your data is JSONL, one object per line:

```json
{"query": "how tall is everest", "positive": "Everest is 8849 m.", "negative": "Everest is in the Alps."}
{"anchor": "python list append", "positive": "list.append(x) adds an item."}
```

No dataset is bundled beyond 8 sample rows in `data/sample/` for smoke tests.
Point `--train-file` at your own data (MS MARCO, NLI, BEIR, synthetic
augmentations, …).

```bash
# projection-only (default, cheapest)
python -m training.train --config configs/projection.yaml \
    --train-file /path/to/train.jsonl \
    --output-dir runs/projection-768

# LoRA
pip install "peft>=0.11"
python -m training.train --config configs/lora.yaml --train-file /path/to/train.jsonl

# exercise the whole pipeline on CPU with no weights downloaded
python -m training.train --smoke-test
```

Dataset handling supports shuffle, seeding, batching, max length, a
deterministic validation split, hard negatives and duplicate filtering.

---

## Encoding

```python
from llama_embedding import LlamaEmbeddingModel

model = LlamaEmbeddingModel.from_pretrained(
    output_dir="runs/projection-768"     # trained head + config
    # backbone_name_or_path="meta-llama/Llama-3.1-8B"   # if no stored config
)

vectors = model.encode(
    ["first text", "second text"],
    batch_size=16,
    max_length=512,
    normalize=True,
)
vectors.shape                    # (2, 768)

model.encode("a single string").shape          # (1, 768)
model.encode_query("what is a rescue drone")   # query template
model.encode_document("An UAV used in SAR...") # document template
```

From the CLI:

```bash
python scripts/encode.py --checkpoint runs/projection-768 \
    --text "a rescue drone" "an emergency aircraft" --embedding-dim 768
```

Query/document prefixes are **configurable and language-agnostic** — no
English-only instruction is hard coded in the vector path.

---

## Semantic search

```python
from llama_embedding import LlamaEmbeddingModel, SemanticSearchIndex

model = LlamaEmbeddingModel.from_pretrained(output_dir="runs/projection-768")
index = SemanticSearchIndex.build(model, documents, batch_size=16)

hits = index.search(model.encode(["finding missing people"]), top_k=5)
for hit in hits:
    print(f"{hit.score:.4f}  {hit.text}")
```

Runnable example: [`examples/semantic_search.py`](examples/semantic_search.py).
Vector-database dependencies are entirely optional — the example uses plain
cosine similarity, and because the vectors are unit-norm, an inner-product
index (FAISS/Qdrant/Chroma) is a drop-in replacement.

---

## Evaluation

Implemented metrics: **cosine similarity**, **Recall@K**, **Precision@K**,
**HitRate@K**, **NDCG@K**, **MRR**, **MAP**.

```bash
python -m evaluation.evaluate \
    --checkpoint runs/projection-768 \
    --eval-file data/sample/eval.jsonl \
    --top-k 1 5 10
```

Evaluation is encoder-agnostic: it consumes an `encode` callable, so the same
code scores the trained model, the raw baseline, or vectors from any external
system. Optional hooks exist for MTEB / sentence-transformers style harnesses;
they are integrations, not bundled results.

---

## Baseline comparison

To answer "did training actually help?", compare the **raw pooled backbone**
(no projection head) against the **trained model** — same backbone, same
evaluation code, only the head differs:

```bash
python scripts/benchmark.py \
    --checkpoint runs/projection-768 \
    --eval-file /path/to/eval.jsonl
```

This prints a side-by-side table of measured metrics. It reports only what it
measures on the data you provide.

---

## Benchmarks

| Model | Dim | MTEB avg | BEIR avg | Notes |
|---|---|---|---|---|
| Llama3.1-Embedding | 768 | NOT YET BENCHMARKED | NOT YET BENCHMARKED | no trained checkpoint |
| Raw Llama 3.1 pooled (baseline) | 4096 | NOT YET BENCHMARKED | NOT YET BENCHMARKED | untrained reference point |

These cells are placeholders and must stay that way until someone publishes
reproducible measurements with a specific checkpoint. Do not fill them with
numbers copied from other projects.

---

## Limitations

* **No trained checkpoint.** This is a framework, not a released model.
* **No benchmark numbers.** Nothing is measured yet.
* **Llama 3.1 8B is heavy.** ~16 GB in bf16 before activations. Expect a
  multi-GPU setup (or a smaller backbone) for real training. The code checks
  device availability and dtype selection rather than assuming a GPU.
* **English-centric defaults.** The symmetric (prefix-free) configuration is the
  neutral default; asymmetric templates improve retrieval but must be tuned per
  language/domain. Nothing here has been tuned for any language.
* **Context length vs pooling.** Mean pooling over very long documents dilutes
  salient content; chunk documents before encoding.
* **Matryoshka is unproven.** The hooks and the objective exist and are tested,
  but prefix quality is only meaningful after a Matryoshka-trained run.

---

## Roadmap

1. Train and publish a first `projection_only` checkpoint at D=768.
2. Run MTEB (retrieval subset) and BEIR; fill in the benchmark table with
   reproducible numbers and the exact commit + data recipe.
3. LoRA recipe with a larger, hard-negative-mined corpus.
4. Validate Matryoshka truncation at 1024/768/512/384/256 and document the
   accuracy curve.
5. Multilingual instruction templates behind the existing config hooks.
6. Optional ONNX / GGUF export of the projection head.
7. MTEB adapter so the model plugs into the standard harness.

---

## License and attribution

**This is an independent open-source project. It is not affiliated with,
endorsed by, or a product of Meta.**

* The **code** in this repository is released under the terms in
  [`LICENSE`](LICENSE) (Apache License 2.0).
* It builds on **Meta Llama 3.1**, which is licensed separately by Meta under
  the Llama 3.1 Community Licence.
* **Original Llama weights are not distributed by this repository.** They are
  downloaded from Hugging Face by `transformers` after you have obtained and
  accepted Meta's terms.
* You must comply with the applicable Llama licence and access terms when using
  the Llama weights or any model derived from them.
* Nothing here relicenses Meta's weights. The Apache-2.0 grant covers this
  project's source code only.

---

## Project layout

```
Llama3.1-Embedding/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/              base.yaml, projection.yaml, lora.yaml
├── llama_embedding/      model, pooling, projection, tokenizer, config,
│                         losses, similarity, inference, lora
├── training/             dataset, collator, trainer, train
├── evaluation/           metrics, retrieval, evaluate
├── scripts/              encode.py, train_embedding.py, benchmark.py
├── examples/             encode_text.py, semantic_search.py
├── tests/                pooling, projection, similarity, loss, shapes,
│                         dataset, metrics, trainer
├── data/sample/          8-row smoke fixtures only
└── docs/architecture.md
```

## Tests

```bash
pytest -q                  # unit tests, CPU, no model download
python -m training.train --smoke-test    # end-to-end training on the stub backbone
```

The suite covers mask-aware pooling, padding exclusion, last-token pooling,
projection shapes, unit-norm output, cosine similarity, contrastive loss,
batch/dimension handling, invalid dimensions, empty input, and seeded
determinism.
