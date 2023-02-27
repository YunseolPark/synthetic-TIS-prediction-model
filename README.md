# synthetic-TIS-prediction-model

The continuation of a bahcelor's project that can be found in the [TIS-synthetic-data repository](https://github.com/YunseolPark/BAthesis-TIS-prediction-using-synthetic-data).
This project is on the generation of a translation initiation site (TIS) prediction model and the analysis of said model through the use of synthetic datasets and transfer learning.

The deep learning model is generated with PyTorch. To run the models, the files `main_original.py`, `main_occlusion.py`, `main_transferlearning.py`, and  `main_percentage.py` can be used.

Synthetic datasets are used with occlusion of different features to obtain interpretability of the model. To generate the synthetic datasets, the file `class_generateTIS.py` can be used and for occlusion, the file `cls_remove_feature5.py`. The files `func_CodonUsage.py`, `func_ConsensusSequence.py`, `func_NuleotideFrequency.py` should be used prior to generation of the synthetic dataset to obtain the necessary information from the real dataset. The file `check_TIS.py` is used to validate the generated synthetic data.

All code has been written in Python v3.8.3.

The Arabidopsis thaliana dataset is from [Magana-Mora et al. (2013)](https://academic.oup.com/bioinformatics/article/29/1/117/272605) and the human dataset is from [Chen et al. (2014)](https://www.sciencedirect.com/science/article/pii/S0003269714002814).
