# SPSS Reproduction

This repository contains the reproduction of the paper **"Striving for Simplicity: Simple Yet Effective Prior-Aware Pseudo-Labeling for Semi-Supervised Ultrasound Image Segmentation"**. The paper introduces a pseudo-labeling technique for semi-supervised ultrasound image segmentation, leveraging an encoder-twin-decoder network with an adversarially learned shape prior to improve segmentation accuracy.

## Paper Details
- **Paper Code**: 4867
- **Paper Link**: [MICCAI 2024 Paper](https://papers.miccai.org/miccai-2024/735-Paper2948.html)
- **Reproduction Level**: 3 (25 points)
- **Github Link**: [SPSS-Reproduction](https://github.com/prachuryanath/SPSS-Reproduction)

## Overview
The proposed method addresses the challenges of limited labeled data and anatomical inaccuracies in ultrasound image segmentation. By balancing labeled and unlabeled data, the approach enhances the precision and usability of automated ultrasound analysis.

## Hardware Requirements
- **Graphics Used**: A100 (40 GB)
- **Training Time GAN**: 19 hrs 52 mins
- **Training Time CNN Model**: 6 hrs 57 mins


## Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv .spss
   source spss/bin/activate
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Conclusion
This reproduction is classified as Level 3 due to the need for moderate changes to the code and repository structure. Despite the challenges, the results are consistent with the original paper, demonstrating the effectiveness of the proposed method.

---

**By**: Prachurya Nath  
**Github**: [prachuryanath](https://github.com/prachuryanath)
