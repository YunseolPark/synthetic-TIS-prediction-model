from cls_TISarchitecture import Net
from func_modeltraining import run_model
import sys
import argparse

if __name__ == '__main__':
    '''
    Usage:
    main_original.py
    main_original.py --models <list of model names to use for training; default: all models> --n_epochs <number of epochs; default=20> --trials <number of trials; default=5>
    '''
    # Paths to the models to train
    model_dict = {'SBBM': ['partitioned_data/arabTIS_train_pos.txt', 'partitioned_data/arabTIS_train_neg.txt', 'partitioned_data/arabTIS_val_pos.txt', 'partitioned_data/arabTIS_val_neg.txt', 'partitioned_data/arabTIS_test_pos.txt', 'partitioned_data/arabTIS_test_neg.txt'],
                  'RBBM': ['partitioned_data/at_train_pos.txt', 'partitioned_data/at_train_neg.txt', 'partitioned_data/at_val_pos.txt', 'partitioned_data/at_val_neg.txt', 'partitioned_data/at_test_pos.txt', 'partitioned_data/at_test_neg.txt'],
                  'CBBM': ['partitioned_data/combined_train_pos.txt', 'partitioned_data/combined_train_neg.txt', 'partitioned_data/combined_val_pos.txt', 'partitioned_data/combined_val_neg.txt', 'partitioned_data/combined_test_pos.txt', 'partitioned_data/combined_test_neg.txt'],
                  'RBBM_half': ['partitioned_data/at_half_train_pos.txt', 'partitioned_data/at_half_train_neg.txt', 'partitioned_data/at_half_val_pos.txt', 'partitioned_data/at_half_val_neg.txt', 'partitioned_data/at_half_test_pos.txt', 'partitioned_data/at_half_test_neg.txt'],
                  'SBBM_percentage': ['partitioned_data/arabTIS_percentage_train_pos.txt', 'partitioned_data/arabTIS_percentage_train_neg.txt', 'partitioned_data/arabTIS_percentage_val_pos.txt', 'partitioned_data/arabTIS_percentage_val_neg.txt', 'partitioned_data/arabTIS_percentage_test_pos.txt', 'partitioned_data/arabTIS_percentage_test_neg.txt'],
                  'CBBM_percentage': ['partitioned_data/combined_percentage_train_pos.txt', 'partitioned_data/combined_percentage_train_neg.txt', 'partitioned_data/combined_percentage_val_pos.txt', 'partitioned_data/combined_percentage_val_neg.txt', 'partitioned_data/combined_percentage_test_pos.txt', 'partitioned_data/combined_percentage_test_neg.txt']
                  }
    parser = argparse.ArgumentParser(description="Trains the original models.")
    parser.add_argument('--models', type=str, nargs='+', default=list(model_dict.keys()))
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()
    print('List of models to be trained: ' + ', '.join(args.models))
    print('Number of epochs: ' + str(args.n_epoch))
    print('Number of trials to repeat training: ' + str(args.trials))
    # Log file name to save training and testing stages
    his_file = 'original_models'
    for model_name in args.models:
        data = model_dict[model_name]
        model = Net()
        print(model)
        print(his_file)
        print(model_name)
        print('\n' + '='*50 + '\n')
        for i in range(args.trials):
            print("Trial number: " + str(i+1))
            run_model(data, args.n_epoch, model, model_name, his_file)