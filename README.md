# 🌍 Exploiting Word Embeddings for Machine Translation

This project investigates **word-level machine translation without parallel sentences** by leveraging **monolingual word embeddings**. It implements and compares:

- Supervised embedding alignment methods
- Unsupervised adversarial mapping (GAN-based translator)

The goal is to learn a mapping between French and English embedding spaces and evaluate translation quality in both settings.

---

## Problem Statement

Given two monolingual embedding spaces:

- 🇫🇷 French word embeddings
- 🇬🇧 English word embeddings

We aim to learn a transformation:

\[
W: \mathbb{R}^d_{fr} \rightarrow \mathbb{R}^d_{en}
\]

such that:

- French embeddings mapped through \( W \) align with English embeddings
- Nearest neighbors in embedding space correspond to correct translations

---

## Methodology

## 1. Supervised Learning (Baseline)

We use a bilingual dictionary to learn a direct mapping between embeddings.

### Approaches:
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent (MGD)
- Simple linear regression mapping

### Objective:
Minimize distance between aligned word pairs:

\[
\| W x_{fr} - x_{en} \|
\]

---

## 2. Unsupervised GAN-Based Translator

We model translation as an adversarial game.

### Generator (Mapping Network)

Maps French embeddings → English space:

```python
self.l1 = nn.Linear(dim, dim, bias=False)

### Discriminator

Classifies real vs generated embeddings:

Classifies real vs generated embeddings:

```Linear(dim → 2048 → 2048 → 1)
LeakyReLU + Dropout + Sigmoid
