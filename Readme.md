# Multimodal Mutual-Information Guided Diffusion

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Project-orange" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> **Status:** 🚧 Ongoing research project — active development  
> **Metrics:** Quantitative evaluation (FID/KID, CLIP/AudioCLIP/ImageBind similarity, alignment metrics) **coming soon**

## 📋 Overview

This repository explores **training-free guidance** for diffusion models using **learned mutual-information (MI) signals** between paired modalities. We train a **density-ratio estimator** at multiple noise levels $t$ to approximate the pointwise MI term:

$$\log \frac{p(x_t,y_t)}{p(x_t)\,p(y_t)}$$

At sampling time, we add its gradient to the base score so the sampler prefers **corresponding pairs**.

### 🎯 Key Features

- **Training-free guidance** for multimodal diffusion models
- **Multiple density-ratio objectives** (Discriminator, DV, uLSIF, RuLSIF, KLIEP)
- **Two modality settings**:
  - Night → Day image translation (DDPM)
  - Audio → Image generation (AudioLDM2 + Stable Diffusion)
- **Latent space guidance** for pretrained models

## 🖼️ Supported Modalities

### Night → Day Image Translation
<p align="center">
  <img src="figures/night2day_dataset.png" alt="Night to Day Dataset Examples" width="80%">
  <br>
  <em>Examples of night-to-day image pairs used for training the density ratio estimator</em>
</p>

Two separately trained DDPMs (night/day) with a ratio model that learns correspondence across noise levels. The gradient nudges the "day" sampler toward the night input.

### Audio → Image Generation
<p align="center">
  <img src="figures/ave_dataset.png" alt="AVE Dataset Examples" width="80%">
  <br>
  <em>Audio-Visual Event (AVE) dataset samples showing paired audio spectrograms and corresponding images</em>
</p>

Guidance in **latent space** using pretrained VAEs (AudioLDM2, SD). The ratio model is trained on AVE pairs; its gradient steers Stable Diffusion toward the audio input.

## 🔬 Technical Approach

### Density Ratio Learning

Let $q(x_t,y_t)=p(x_t,y_t)$ (joint) and $r(x_t,y_t)=p(x_t)p(y_t)$ (product of marginals), both at the **same** noise level $t$.

We learn a time-conditioned estimator $\hat{\ell}_\theta(x_t,y_t,t)$ such that:

$$\hat{\ell}_\theta \approx \log\frac{q}{r}$$

### Guided Sampling

During sampling for $x_t\mid y_t$, the guidance is:

$$\nabla_{x_t}\log p(x_t\mid y_t) \approx \underbrace{s_\phi(x_t,t)}_{\text{unconditional score}} + \lambda\,\nabla_{x_t}\hat{\ell}_\theta(x_t,y_t,t)$$

with an optional schedule $\lambda=\lambda(t)$ (e.g., scaled by $\sigma_t$).

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.7+ (for GPU support)
- 16GB+ GPU memory recommended

### Environment Setup

```bash
# Create conda environment
conda create -n mm_guidance python=3.9 -y
conda activate mm_guidance

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install einops tqdm soundfile librosa scikit-learn
pip install diffusers transformers accelerate safetensors
```

## 🚀 Quick Start

### Training

Train density ratio estimators for different modality pairs:

```bash
# Night↔Day with discriminator loss
python src/train.py ddpm_night2day --loss_type disc --epochs 20

# Audio→Image with RuLSIF (latent space)
python src/train.py sd_audioldm --loss_type rulsif --batch_size 16
```

### Inference

Generate guided samples:

```bash
# Guided night→day pairs (DDPM)
python src/inference.py ddpm_night2day \
    --loss_type disc \
    --guidance_scale 2.0 \
    --sampling_steps 100

# Audio-conditioned images (SD + AudioLDM2)
python src/inference.py sd_audioldm \
    --loss_type rulsif \
    --guidance_scale 1.5
```

## 📊 Implemented Density-Ratio Objectives

We parameterize a small network $T_\theta(x_t,y_t,t)$ and map it to a ratio or log-ratio:

| Method | Objective | Guidance Signal |
|--------|-----------|-----------------|
| **Discriminator** | Logistic classification: $\mathbb{E}_q[-\log\sigma(T)] + \mathbb{E}_r[-\log(1-\sigma(T))]$ | $\nabla_{x_t} T_\theta$ |
| **Donsker–Varadhan** | MI lower bound: $\max_\theta\; \mathbb{E}_q[T] - \log \mathbb{E}_r[e^{T}]$ | $\nabla_{x_t} \hat{\ell}_\theta$ |
| **uLSIF** | Direct ratio fitting: $\tfrac{1}{2} \mathbb{E}_r[w^2] - \mathbb{E}_q[w]$ | $\nabla_{x_t} \log w$ |
| **RuLSIF** | Relative ratio: $w_\alpha = \frac{q}{\alpha q + (1-\alpha) r}$ | $\nabla_{x_t} \log\left(\frac{w_\alpha}{1-\alpha+\alpha w_\alpha}\right)$ |
| **KLIEP** | KL-based: $\max_\theta\; \mathbb{E}_q[\log w] \;\text{s.t.}\; \mathbb{E}_r[w]=1$ | $\nabla_{x_t} \log w$ |

## 📁 Project Structure

```
src/
├── dataloaders/
│   ├── ave.py                      # AVE (audio–image) pairs
│   └── night2day.py                # Night/Day image pairs
├── models/
│   └── mi_models.py                # Ratio/MI estimators
├── utils/
│   ├── density_ratio_losses.py     # Loss implementations
│   ├── diffusion_schedule.py       # Schedules & time embeddings
│   ├── score_computation.py        # Gradient computation
│   ├── guided_sampler_native.py    # Native DDPM guidance
│   └── diffusion_model_loader.py   # Model loaders
├── ddpm/
│   └── denoising_diffusion_pytorch/  # Local DDPM implementation
├── train.py                         # Training script
└── inference.py                     # Inference script

checkpoints/                         # Saved models
├── {loss_type}/
│   └── {model_type}_{loss_type}_mi_model_best.pt

figures/                             # Dataset visualizations
├── night2day_dataset.png
└── ave_dataset.png
```

## 💾 Model Checkpoints

- **DDPM models:** `src/ddpm/results/<domain>/<cfg>/<run_timestamp>/model-<k>.pt`
- **MI models:** `checkpoints/{loss_type}/{model_type}_{loss_type}_mi_model_best.pt`

## 📈 Datasets

### Night/Day Dataset
- Paired night and day images for image-to-image translation
- Curated correspondences between different lighting conditions

### AVE (Audio-Visual Event) Dataset
- Synchronized audio-video pairs
- Mel-spectrograms paired with aligned video frames
- Product-of-marginals negatives via in-batch shuffling

## 🗺️ Roadmap

- [ ] **Quantitative Metrics**
  - [ ] FID/KID scores
  - [ ] CLIP/AudioCLIP/ImageBind similarity
  - [ ] Cross-modal alignment metrics
- [ ] **Ablation Studies**
  - [ ] Loss function comparison
  - [ ] Guidance scale analysis
  - [ ] Schedule ablations
- [ ] **Extended Modalities**
  - [ ] Text → Image guidance
  - [ ] Video → Audio generation
  - [ ] Cross-domain translation
- [ ] **Model Improvements**
  - [ ] Adaptive guidance scheduling
  - [ ] Multi-scale density ratios
  - [ ] Efficiency optimizations

## 📝 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inprogress{mm_guidance_2025,
  title   = {Mutual-Information Guided Sampling for Multimodal Diffusion},
  author  = {Oubari, Fouad and Collaborators},
  year    = {2025},
  note    = {Work in progress}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AVE dataset creators
- Authors of diffusers, AudioLDM2, and Stable Diffusion
- Density ratio estimation literature

---

<p align="center">
  <em>For questions or collaborations, please open an issue or contact the authors.</em>
</p>