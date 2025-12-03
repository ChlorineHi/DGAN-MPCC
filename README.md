# DGAN-MPCC: Dual-GAN Enhanced Multi-Positive Contrastive Clustering Method

![Model Architecture](picture.png)

## Abstract

AI-driven clustering methods have significantly enhanced the capacity of researchers to explore the heterogeneity inherent in single-cell omics data, which is a crucial aspect of understanding complex biological systems in healthcare. Despite advancements, most existing methods still face challenges, such as (1) inherent sparsity and noise in cell data, which frequently lead to overfitting in networks. To address this, some researchers have proposed using Generative Adversarial Networks (GANs), however, the conventional single GAN architecture primarily focuses on simple data enhancement and lacks the capacity to infer complex biological data, thus leading to suboptimal clustering performance. (2) Contrastive learning has been proposed to obtain high-quality clustering structures; however, existing methods predominantly rely on a single positive pair, which prevents them from modeling and learning continuous transitions in cell states and thus hinders the establishment of feature representations sensitive to cell types. 

To address these issues, we propose a novel **Dual-GAN Enhanced Multi-Positive Contrastive Clustering Method (DGAN-MPCC)**, tailored for low-quality single-cell data. Specifically, we propose using two independent GANs to simultaneously enhance the quality of both the input and bottleneck layers, thereby refining the generated cell embedding. Additionally, we have developed a multi-positive contrastive clustering framework that adaptively defines a multi-positive set from clustering structures, enabling each sample to establish positive relationships with all samples within the same cluster, thereby diversifying supervisory signals within the same class. Extensive experiments on several real-world single-cell datasets demonstrate that DGAN-MPCC surpasses current methods across multiple scenarios, providing a more robust and efficient tool for AI-driven decision-making in healthcare.

## Architecture

The DGAN-MPCC framework consists of:

1. **Head GAN Network**: Enhances input data quality before encoding
2. **ZINB-based Autoencoder**: Core feature extraction and dimensionality reduction module
3. **Tail GAN Network**: Enhances bottleneck layer quality after encoding
4. **Multi-Positive Contrastive Learning Module**: Adaptive clustering optimization using MultiPositive InfoNCE loss

## Key Features

- **Dual GAN Architecture**: Two independent GANs for enhanced data quality at both input and bottleneck layers
- **Multi-Positive Contrastive Learning**: Adaptive multi-positive set definition from clustering structures
- **ZINB Loss**: Zero-Inflated Negative Binomial loss for handling sparse single-cell data
- **Robust Clustering**: Improved performance on low-quality single-cell omics data

## File Structure

- `mymodel_new_v2_formal.py`: Main model implementation
- `gan_models.py`: Generator and Discriminator architectures
- `contrastive_loss.py`: Multi-positive contrastive loss implementation
- `zinb_loss.py`: ZINB loss implementation
- `layers.py`: Custom neural network layers
- `preprocess.py`: Data preprocessing utilities
- `evaluation.py`: Clustering evaluation metrics
- `klug/`: Sample datasets

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- scanpy
- pandas

## Usage

[Add usage instructions here]

## Citation

[Add citation information here]

## License

[Add license information here]

