import torch
from cls_TISarchitecture import Net
from func_modeltraining import run_model
import sys

if __name__ == '__main__':
    '''
    Usage:
    main_transferlearning.py
    main_transferlearning.py <name of pretrained model> <list of model names to use for training> <list of layers to freeze> <number of trials>
    main_transferlearning.py <name of pretrained model> <list of model names to use for training> <list of layers to freeze>
    '''
    # Dictionary with paths to the files
    model_dict = {'SBBM': ['partitioned_data/arabTIS_train_pos.txt', 'partitioned_data/arabTIS_train_neg.txt', 'partitioned_data/arabTIS_val_pos.txt', 'partitioned_data/arabTIS_val_neg.txt', 'partitioned_data/arabTIS_test_pos.txt', 'partitioned_data/arabTIS_test_neg.txt'],
                  'RBBM': ['partitioned_data/at_train_pos.txt', 'partitioned_data/at_train_neg.txt', 'partitioned_data/at_val_pos.txt', 'partitioned_data/at_val_neg.txt', 'partitioned_data/at_test_pos.txt', 'partitioned_data/at_test_neg.txt'],
                  'CBBM': ['partitioned_data/combined_train_pos.txt', 'partitioned_data/combined_train_neg.txt', 'partitioned_data/combined_val_pos.txt', 'partitioned_data/combined_val_neg.txt', 'partitioned_data/combined_test_pos.txt', 'partitioned_data/combined_test_neg.txt'],
                  'RBBM_half': ['partitioned_data/at_half_train_pos.txt', 'partitioned_data/at_half_train_neg.txt', 'partitioned_data/at_half_val_pos.txt', 'partitioned_data/at_half_val_neg.txt', 'partitioned_data/at_half_test_pos.txt', 'partitioned_data/at_half_test_neg.txt'],
                  'SBBM_percentage': ['partitioned_data/arabTIS_percentage_train_pos.txt', 'partitioned_data/arabTIS_percentage_train_neg.txt', 'partitioned_data/arabTIS_percentage_val_pos.txt', 'partitioned_data/arabTIS_percentage_val_neg.txt', 'partitioned_data/arabTIS_percentage_test_pos.txt', 'partitioned_data/arabTIS_percentage_test_neg.txt'],
                  'CBBM_percentage': ['partitioned_data/combined_percentage_train_pos.txt', 'partitioned_data/combined_percentage_train_neg.txt', 'partitioned_data/combined_percentage_val_pos.txt', 'partitioned_data/combined_percentage_val_neg.txt', 'partitioned_data/combined_percentage_test_pos.txt', 'partitioned_data/combined_percentage_test_neg.txt']}
    pretrained_model = sys.argv[1]
    model_list = sys.argv[2]
    layer_list = sys.argv[3]
    #layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'out']
    # Default layer of trials
    if len(sys.argv) == 4:
        trials = 5
    else:
        trials = sys.argv[4]
    # Log file name to save the history of training
    his_file = 'transfer_learning'
    for model_name in model_list:
        data = model_dict[model_name]
        model = Net()
        model.load_state_dict(torch.load('models/' + pretrained_model + '.pth'))
        #code.interact(local = dict(globals(), **locals()))
        model_name = 'transfer_' + pretrained_model + 'to' + model_name
        print(model)
        print(model_name)
        print('\n' + '='*50 + '\n')
        for i in range(trials):
            run_model(data, 20, model, model_name, his_file, layer_list)