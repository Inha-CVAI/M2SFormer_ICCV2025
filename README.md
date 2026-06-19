# M2SFormer [ICCV2025 Highlight]
The reproduction code of M2SFormer which is accepted in ICCV 2025

## 💡 We also have other Image Forensics Localization projects that may interest you ✨
> [**SARIF: Segment Anything for Robust Image Forensics [ECCV 2026]**](https://github.com/Inha-CVAI/SARIF_ECCV2026)  
> Dong-Hyun Moon, Ju-Hyeon Nam, Sang-Chul Lee  
> [![Github](https://img.shields.io/badge/-Github-black?style=flat&logo=github&logoColor=white)](https://github.com/Inha-CVAI/SARIF_ECCV2026)
> [![Stars](https://img.shields.io/github/stars/Inha-CVAI/SARIF_ECCV2026?style=social)](https://github.com/Inha-CVAI/SARIF_ECCV2026/stargazers)
> [![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey.svg)](#)


## 🔔 Latest News
- [2026-06-17]: 🎉 We are excited to announce that our new image forgery localization paper, **SARIF**, was accepted to **ECCV 2026**! ✨
- [2026-01-07]: We uploaded ICCV 2025 Inference Code!
- [2025-09-04]: We uploaded ICCV 2025 Poster!
- [2025-07-24]: Our paper was selected as _**Highlight**_ paper in ICCV 2025. 🎉🎉
- [2025-06-26]: We are happy to announce that M2SFormer was accepted at [ICCV 2025](https://iccv.thecvf.com/virtual/2025/poster/881). 🥳🥳

## Poster
<img width="3150" height="1772" alt="ICCV2025 Poster-1" src="https://github.com/user-attachments/assets/450f6743-61e6-43ac-bea9-fdae044b08fa" />

## Video

## Abstract
Image editing techniques have rapidly advanced, facilitating both innovative use cases and malicious manipulation of digital images. Deep learning-based methods have recently achieved high accuracy in pixel-level forgery localization, yet they frequently struggle with computational overhead and limited representation power, particularly for subtle or complex tampering. In this paper, we propose M2SFormer, a novel Transformer encoder-based framework designed to overcome these challenges. Unlike approaches that process spatial and frequency cues separately, M2SFormer unifies multi-frequency and multi-scale attentions in the skip connection, harnessing global context to better capture diverse forgery artifacts. Additionally, our framework addresses the loss of fine detail during upsampling by utilizing a global prior map—a curvature metric indicating the difficulty of forgery localization—which then guides a difficulty-guided attention module to preserve subtle manipulations more effectively. Extensive experiments on multiple benchmark datasets demonstrate that M2SFormer outperforms existing state-of-the-art models, offering superior generalization in detecting and localizing forgeries across unseen domains.

## Overall Architecture of M2SFormer
<img width="8665" height="3697" alt="M2SFormer" src="https://github.com/user-attachments/assets/d85f07db-5f8d-4a32-9e73-291f7e1df151" />

### 🔑 Key Insights of the Overall Architecture
- _**Unified Multi-Spectral and Multi-Scale Attention**_:
  Instead of processing spatial and frequency cues separately, M2SFormer introduces an M2S Attention Block in the skip connections. This block fuses frequency-domain features (via 2D DCT) and multi-scale spatial cues (SIFT-inspired pyramids), enabling the model to capture both subtle tampering artifacts and diverse forgery patterns across different scales
  
- _**Edge-Aware Difficulty-Guided Decoder**_:
  The decoder integrates a curvature-based global prior map to quantify the “difficulty” of localization. By identifying high-curvature edge regions as perceptually harder, the model transforms this difficulty into a text-based representation and applies Difficulty-Guided Attention (DGA), focusing the Transformer decoder on challenging regions while preserving fine details
  
- _**Balanced Efficiency and Generalization**_:
  By embedding frequency attention within feature maps (not raw inputs) and leveraging a lightweight PVT-v2 backbone, M2SFormer achieves a favorable trade-off between accuracy and computation. This design ensures robust cross-domain generalization while keeping the parameter count (~27M) and FLOPs (~14G) efficient compared to prior frequency- or Transformer-based approaches
  
## Experiment Results

### 🔑 Key Findings from Experiment Results
- _**Superior Cross-Domain Generalization**_:
  M2SFormer consistently outperforms state-of-the-art methods on unseen datasets (e.g., Columbia, IMD2020, CoMoFoD, In-the-Wild, MISD). For instance, when trained on CASIAv2, it achieves DSC 58.8 / mIoU 50.8, and when trained on DIS25k, it reaches DSC 87.7 / mIoU 81.5, demonstrating strong robustness across manipulation types and domains
  
- _**Efficiency with Higher Accuracy**_:
  Despite achieving better accuracy, M2SFormer maintains relatively low computational cost—~27.4M parameters and ~14.2 GFLOPs at 256×256 resolution. Compared with models like FBINet and EITLNet, which rely heavily on frequency or dual encoders, M2SFormer strikes a better balance between efficiency and performance
  
- _**Clear Gains from Ablation Studies**_:
  Ablation studies confirm that both multi-spectral + multi-scale fusion and the Edge-Aware DGA decoder are crucial. Removing either significantly degrades unseen-domain performance (e.g., DSC drops from 43.0 → 32.3 without DGA). This validates that the joint integration of spectral-spatial attention and difficulty-guided decoding is the main driver of generalization improvements

- _**Real-World Corruption Robustness**_:
  In addition to domain transfer, M2SFormer shows greater resilience to real-world corruptions such as JPEG compression, blurring, and noise, which are common in online tampered images. By leveraging frequency-domain cues (DCT) and curvature-based difficulty guidance, the model preserves boundary fidelity and subtle traces even under degraded conditions, outperforming conventional CNN or single-domain approaches

### Segmentation results on CASIAv2 training scheme
<img width="2360" height="688" alt="image" src="https://github.com/user-attachments/assets/ee8a294c-9da2-4db1-9539-1607057bf531" />

<img width="6209" height="3965" alt="QualitativeResults" src="https://github.com/user-attachments/assets/7493003c-ab06-4b9d-92e5-ab6de5c97c7c" />

### Segmentation results on DIS25k training scheme
<img width="2384" height="694" alt="image" src="https://github.com/user-attachments/assets/a1f2ab53-c1d9-45d2-a3d1-ddb720fbfb69" />


<img width="6209" height="3965" alt="SupQualitativeResults" src="https://github.com/user-attachments/assets/07225969-1e26-43ce-b80d-bb8235e18cde" />

### Segmentatoin results on real-world scenario
<img width="1373" height="467" alt="image" src="https://github.com/user-attachments/assets/71694016-ca21-43f1-8154-89ddc7880bc4" />


### Efficiency Analysis
<img width="4603" height="4338" alt="EfficiencyGraph" src="https://github.com/user-attachments/assets/5035cc54-5cbb-4d47-9dd5-f0da89d31b74" />
  
# Code Usage

# Bibtex
