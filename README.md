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

#### Generator

Maps French embeddings into English space:

```python
nn.Linear(dim, dim, bias=False)
```

---

#### Discriminator

Distinguishes real English embeddings from mapped ones:

```
Linear(dim → 2048 → 2048 → 1)
LeakyReLU
Dropout
Sigmoid
```

---

## Training

The model uses adversarial training:

- Discriminator learns to separate real vs generated embeddings  
- Generator learns to fool the discriminator  

### Stabilization tricks:

- Label smoothing (0.8 real / 0.2 fake)
- Orthogonality constraint on mapping matrix
- SGD optimizer

---

## Data

### Embeddings

```
wiki.fr.vec
wiki.en.vec
```

- 50,000 words
- 300-dimensional vectors

---

### Dictionary

```
Train: fr-en.0-5000.txt
Test: fr-en.5000-6500.txt
```

---

## Evaluation

- Nearest neighbor search in embedding space
- Cosine similarity
- Translation accuracy on test dictionary

---

## Tech Stack

```
Python
PyTorch
NumPy
scikit-learn
SciPy
Matplotlib
```

---

## Summary

This project shows that word embeddings contain enough geometric structure to enable cross-lingual translation through:

- Supervised linear alignment
- Unsupervised adversarial learning (GAN)
