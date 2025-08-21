# M2SFormer
The reproduction code of M2SFOrmer which is accepted in ICCV 2025

## Abstract
Image editing techniques have rapidly advanced, facilitating both innovative use cases and malicious manipulation of digital images. Deep learning-based methods have recently achieved high accuracy in pixel-level forgery localization, yet they frequently struggle with computational overhead and limited representation power, particularly for subtle or complex tampering. In this paper, we propose M2SFormer, a novel Transformer encoder-based framework designed to overcome these challenges. Unlike approaches that process spatial and frequency cues separately, M2SFormer unifies multi-frequency and multi-scale attentions in the skip connection, harnessing global context to better capture diverse forgery artifacts. Additionally, our framework addresses the loss of fine detail during upsampling by utilizing a global prior map—a curvature metric indicating the difficulty of forgery localization—which then guides a difficulty-guided attention module to preserve subtle manipulations more effectively. Extensive experiments on multiple benchmark datasets demonstrate that M2SFormer outperforms existing state-of-the-art models, offering superior generalization in detecting and localizing forgeries across unseen domains.

## Overall Architecture of M2SFormer

## Experiment Results

### Segmentation results on CASIAv2 training scheme

### Segmentation results on DIS25k training scheme

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
