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
However, a lot of real-world datasets contain noise and many genome annotations like TIS prediction are high-risk problems. Furthermore, real-world data are extremely complex, which makes it difficult to find the features that influence the decision of the model.
Synthetic data can be used to solve this problem. In particular, synthetic data can be used to give insight into the features of the model and thus into the real-world data. They are suitable for this purpose since they are constructed by selecting and incorporating some of the complex features of the real-world dataset. The outcome of the synthetic model can then be compared to the real model to find the features that contribute most to the prediction of TIS. Furthermore, the effect of noise on datasets can also be investigated to see how the model performs with noisy data.
