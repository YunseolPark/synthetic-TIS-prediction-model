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

%Figure

## Dataset

| Dataset | Source | Size |
| --- | --- | --- |
| Synthetic | n/a | 27102 |
| Real | [Magana-Mora et al. (2013)] | 27102 |
| Combined | n/a | 27102 |
| Human |  | 1500 |
| Human test |  | 200 |

### Synthetic Dataset
The synthetic dataset, generated via the Python code in file GenerateTIS.py, contains 27102 sequences that are 300 nucleotides long. Each TIS is centrally located in the sequence, on the 150th – 152nd nucleotide. They were generated with the same structure as the real dataset. The dataset is generated with 5 features, adding each feature in a different step (Figure 1). The following sections will discuss each step in detail.

The consensus sequence was obtained from the real data and inserted only for the positive dataset (the dataset with true TISs). The 10 nucleotides each directly upstream and downstream to TIS were taken and a position weight matrix (PWM) was made using `ConsensusSequence.py` as seen in Table. The consensus sequence was determined using the 50/70% rule, a modification of Cavener 50/75% rule ([Cavener, 1987]). This consensus sequence was similar to the consensus sequence of higher plants, indicating that it is not only a representation of the real data but the representation for *A. thaliana* in general.
The upstream ATG was also added to the positive dataset despite the evidence of it being a negatively influencing factor. This was done to see if the model would be able to learn leaky scanning.
The downstream stop was taken by selecting a random stop codon from TAA, TAG, and TGA and was added to the negative dataset (the dataset with false TISs).
The donor splice site was obtained from [Kim et al. (2019)] (Table ) and added upstream for the negative dataset and downstream for the positive dataset.
Then the rest of the sequence was filled with nucleotide frequency, also obtained from the real data as a PWM (Table ) using `NucleotideFrequency.py`.

|           | -10 | -9 | -8 | -7 | -6 | -5 | -4 |  -3 |  -2 | -1 |
|:---------:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:--:|
|     A%    |  35 | 32 | 34 | 35 | 36 | 33 | 45 |  50 |  42 | 44 |
|     C%    |  16 | 17 | 20 | 17 | 16 | 24 | 14 |  11 |  29 | 19 |
|     G%    |  20 | 21 | 18 | 20 | 22 | 17 | 21 |  24 |  9  | 23 |
|     T%    |  29 | 30 | 27 | 28 | 27 | 26 | 19 |  15 |  20 | 14 |
| Consensus |  a  |  a |  a |  a |  a |  a |  a | A/G | A/C |  a |
<br/>
|           | -10 | -9 | -8 | -7 | -6 | -5 | -4 |  -3 |  -2 | -1 |
|:---------:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:--:|
|     A%    |  35 | 32 | 34 | 35 | 36 | 33 | 45 |  50 |  42 | 44 |
|     C%    |  16 | 17 | 20 | 17 | 16 | 24 | 14 |  11 |  29 | 19 |
|     G%    |  20 | 21 | 18 | 20 | 22 | 17 | 21 |  24 |  9  | 23 |
|     T%    |  29 | 30 | 27 | 28 | 27 | 26 | 19 |  15 |  20 | 14 |
| Consensus |  a  |  a |  a |  a |  a |  a |  a | A/G | A/C |  a |

<br/><br/>

|  Position |  -3 |  -2 |  -1 |
|:---------:|:---:|:---:|:---:|
|     A     | 352 | 651 |  85 |
|     C     | 361 | 131 |  45 |
|     G     | 154 |  87 | 777 |
|     T     | 132 | 159 |  91 |
| Consensus |  C  |  A  |  G  |

<br/><br/>

| Class |  A%  |  C%  |  G%  |  T%  |
|:-----:|:----:|:----:|:----:|:----:|
|  UTR  | 31.1 | 19.6 | 15.5 | 33.8 |
|  ORF  | 26.1 | 23.0 | 20.4 | 29.7 |
<br/>
|    Class   |  A%  |  C%  |  G%  |  T%  |
|:----------:|:----:|:----:|:----:|:----:|
|  Upstream  | 31.4 | 18.4 | 18.5 | 31.7 |
| Downstream | 31.3 | 18.1 | 18.5 | 31.4 |

### Combined dataset
The combined dataset is generated by combining the real and synthetic datasets in a 1:1 ratio while maintaining the size to be the same as that of the real and synthetic datasets.

### Human dataset
The human dataset has its own test set, indicated in Table as *Human test set*, unlike the other datasets, which required partitioning to gain test sets. The human test set was taken from the same source and from the same database.

## Model Training
We train a synthetic black-box model (SBBM), real black-box model (RBBM), and combined black-box model (CBBM) using the synthetic, real, and combined datasets, respectively. All three datasets were partitioned in a way that 90% of the full dataset was used for the training and validation sets (further partitioned into 7/8th and 1/8th respectively), and 10% for the test set.

The metrics used were recall, precision, accuracy, and f1 score, but only accuracy and f1 score will be noted here.

## Feature analysis
###x
