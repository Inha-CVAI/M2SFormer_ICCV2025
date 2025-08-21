# M2SFormer
The reproduction code of M2SFOrmer which is accepted in ICCV 2025

## Latest News
Our paper was selected as Highlight paper in ICCV 2025!!

## Abstract
Image editing techniques have rapidly advanced, facilitating both innovative use cases and malicious manipulation of digital images. Deep learning-based methods have recently achieved high accuracy in pixel-level forgery localization, yet they frequently struggle with computational overhead and limited representation power, particularly for subtle or complex tampering. In this paper, we propose M2SFormer, a novel Transformer encoder-based framework designed to overcome these challenges. Unlike approaches that process spatial and frequency cues separately, M2SFormer unifies multi-frequency and multi-scale attentions in the skip connection, harnessing global context to better capture diverse forgery artifacts. Additionally, our framework addresses the loss of fine detail during upsampling by utilizing a global prior map—a curvature metric indicating the difficulty of forgery localization—which then guides a difficulty-guided attention module to preserve subtle manipulations more effectively. Extensive experiments on multiple benchmark datasets demonstrate that M2SFormer outperforms existing state-of-the-art models, offering superior generalization in detecting and localizing forgeries across unseen domains.

## Overall Architecture of M2SFormer
<img width="8665" height="3697" alt="M2SFormer" src="https://github.com/user-attachments/assets/d85f07db-5f8d-4a32-9e73-291f7e1df151" />

### Difficulty-Guided Attention
<img width="3165" height="2156" alt="TGADecoder" src="https://github.com/user-attachments/assets/afc10cd7-bbd0-4648-95c8-8e7f0ce086d3" />

## Experiment Results

### Segmentation results on CASIAv2 training scheme
<img width="6209" height="3965" alt="QualitativeResults" src="https://github.com/user-attachments/assets/7493003c-ab06-4b9d-92e5-ab6de5c97c7c" />

### Segmentation results on DIS25k training scheme
<img width="6209" height="3965" alt="SupQualitativeResults" src="https://github.com/user-attachments/assets/07225969-1e26-43ce-b80d-bb8235e18cde" />

### Efficiency Analysis
<img width="4603" height="4338" alt="EfficiencyGraph" src="https://github.com/user-attachments/assets/5035cc54-5cbb-4d47-9dd5-f0da89d31b74" />

# Code Usage

# Bibtex
```
@inproceedings{nam2025m2sformer,
  title = {M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization},
  author = {Nam, Ju-Hyeon and Moon, Dong-Hyun and Lee, Sang-Chul},
  booktitle = {Proceedings of the IEEE/CVF international conference on computer vision},
  year = {2025},
  keywords = {international}
}
```
