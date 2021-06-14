# synthetic-TIS-prediction-model
The continuation of a project that can be found in the [TIS-synthetic-data repository](https://github.com/YunseolPark/BAthesis-TIS-prediction-using-synthetic-data).
This project is on the generation of a TIS prediction model and the analysis of said model through the use of synthetic datasets and transfer learning.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Feature Analysis](#feature-analysis)
5. [Noise Analysis](#noise-analysis)

## Introduction

Prediction of translation initiation sites (TISs) can give insight into translation and the proteins synthesized by certain mRNAs. Furthermore, by interpreting the prediction model, it may even aid in uncovering new translation mechanisms or give emphasis to an existing one.
However, a lot of real-world datasets contain noise and they are extremely complex, which makes it difficult to find the features that influence the decision of the model.
Synthetic data can be used to solve this problem. In particular, synthetic data can be used to give insight into the features of the model and thus into the real-world data. They are suitable for this purpose since they are constructed by selecting and incorporating some of the complex features of the real-world dataset. The outcome of the synthetic model can then be compared to the real model to find the features that contribute most to the prediction of TIS. Furthermore, the effect of noise on datasets can also be investigated to see how the model performs with noisy data.

## Dataset

| Dataset | Source | Size |
| --- | --- | --- |
| Synthetic | n/a | 27102 |
| Real | Magana-Mora et al. (2013) | 27102 |
| Combined | n/a | 27102 |
| Human |  | 1500 |
| Human test |  | 200 |

### Synthetic Dataset
The synthetic dataset, generated via the Python code in file GenerateTIS.py, contains 27102 sequences that are 300 nucleotides long. Each TIS is centrally located in the sequence, on the 150th – 152nd nucleotide. They were generated with the same structure as the real dataset. The dataset is generated with 5 features, adding each feature in a different step (Figure 1). The following sections will discuss each step in detail.
