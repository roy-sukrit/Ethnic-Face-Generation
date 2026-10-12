# Face Generation Based on Ethnicity

**Privacy-Preserving Synthetic Face Generation for Ethnic Groups — a fairness evaluation using soft-biometrics**


---

## Overview

This repository extends the [FLUXSynID framework](https://github.com/Raul2718/FLUXSynID) (Melzi et al., ICCV 2025 Workshops) with ethnicity-specific attribute files, batch prompt generation ,ethnic LoRA fine-tuning, and a periocular biometric evaluation pipeline.

Existing datasets (FairFace, VGGFace2, CelebA) are heavily skewed toward Caucasian and East Asian populations, leaving groups such as Inuit, Polynesian, and Latin American severely underrepresented. This work generates **~14,000 synthetic periocular images** across **7 ethnic groups** using FLUX.1-dev with ethnic LoRA fine-tuning — with zero real biometric data — and evaluates whether a KNN classifier trained on synthetic embeddings can transfer to real images.

---

## Dataset

<div align="center">
  <img width="500" height="500" alt="Sample synthetic identities" src="" />
</div>

<br/>

The synthetic face dataset (~2,000 images per ethnicity group, 7 groups) is available on request.

📧 Contact: [Sukrit](mailto:a00325746@student.tus.ie)

---

## Pipeline

```
Ethnicity Taxonomy
       │
       ▼
Attribute Sampling  ──►  Qwen2.5 LLM  ──►  FLUX.1-dev + Ethnic LoRA
(origins, clothing,       prompt expansion    (fp8, DPM++2M, 20 steps,
 gender, clash filter)                         1024×1024 → face crop)
                                                      │
                                                      ▼
                                           Periocular Crop (MediaPipe)
                                                      │
                                                      ▼
                                        DINOv2 ViT-L/14 Embeddings
                                                      │
                                                      ▼
                                           KNN Classifier (k=7, cosine)
                                                      │
                                                      ▼
                                        Evaluated on real FairFace crops
```

---

## Repository Structure

```
attributes/
├── east_asian/
│   ├── origins.txt               # Geographic/cultural origins
│   ├── genders.txt               # Gender distribution with sampling weights
│   ├── clothing_male.txt         # Male clothing descriptors
│   ├── clothing_female.txt       # Female clothing descriptors
│   ├── attribute_clashes.json    # Incompatible attribute combinations
│   └── file_probabilities.json   # Sampling probabilities per attribute file
├── latin_america/
├── north_africa_middle_eastern/
├── south_east_asian/
├── Inuit/
├── Northern_Asian/
└── Polynesian/

scripts/
└── ...                           # Modified generation and evaluation scripts
```

---

## Ethnic Groups

| Group | Folder |
|---|---|
| East Asian | `east_asian/` |
| Latin American | `latin_america/` |
| Middle Eastern / North African | `north_africa_middle_eastern/` |
| Southeast Asian | `south_east_asian/` |
| Inuit | `Inuit/` |
| Northern Asian | `Northern_Asian/` |
| Polynesian | `Polynesian/` |

---

## How the Attribute System Works

Each group folder defines the visual attributes sampled during generation:

- **`origins.txt`** — Cultural and geographic descriptors used in prompt construction (e.g. "Japanese", "Korean")
- **`genders.txt`** — Gender labels with sampling weights
- **`clothing_male.txt` / `clothing_female.txt`** — Clothing descriptors appropriate to the group
- **`file_probabilities.json`** — Controls how often each attribute file is sampled, allowing realistic demographic distributions
- **`attribute_clashes.json`** — Filters out visually inconsistent attribute combinations before generation

Sampled attributes are passed to **Qwen2.5** for natural language prompt expansion, then fed to **FLUX.1-dev** with an ethnic LoRA for image synthesis.

---

## Key Results

| Model | Best Accuracy | Dataset |
|---|---|---|
| DINOv2 ViT-L/14 (synth-trained KNN) | **85.7%** (East Asian) | N=56 real images |
| DINOv2 ViT-L/14 | within **4.4pp** of supervised baseline | N=300 FairFace |
| ResNet-50 (synth-trained KNN) | 58.9% (Latin American) | N=56 real images |

A KNN classifier trained **exclusively on synthetic embeddings** achieves competitive accuracy on unseen real FairFace images — confirming that synthetic data captures genuine ethnic morphological structure.

---

## Installation

This repo extends FLUXSynID — set up the base framework first:

```bash
# 1. Set up the base FLUXSynID framework
git clone https://github.com/Raul2718/FLUXSynID.git
cd FLUXSynID
# Follow FLUXSynID's own installation instructions

# 2. Clone this repo for ethnic attribute files and evaluation scripts
cd ..
git clone https://github.com/roy-sukrit/FluxSynID-Ethnic.git
pip install -r FluxSynID-Ethnic/requirements.txt
```


---

## License

MIT License (for the ethnic extension files and evaluation scripts in this repo).  

