from cls_TISarchitecture import Net
from func_modeltraining import run_model
import sys

if __name__ == '__main__':
    '''
    Usage:
    main_occlusion.py
    main_occlusion.py <list of model names to use for training> <number of trials - default: 5>
    main_occlusion.py <list of model names to use for training>
    main_occlusion.py <number of trials>
    '''
    # Dictionary with paths to the files
    model_dict = {'SBBM_only_consensus': ['partitioned_data/arabTIS_only_consensus_train_pos.txt', 'partitioned_data/arabTIS_only_consensus_train_neg.txt', 'partitioned_data/arabTIS_only_consensus_val_pos.txt', 'partitioned_data/arabTIS_only_consensus_val_neg.txt', 'partitioned_data/arabTIS_only_consensus_test_pos.txt', 'partitioned_data/arabTIS_only_consensus_test_neg.txt'],
                  'SBBM_except_consensus': ['partitioned_data/arabTIS_except_consensus_train_pos.txt', 'partitioned_data/arabTIS_except_consensus_train_neg.txt', 'partitioned_data/arabTIS_except_consensus_val_pos.txt', 'partitioned_data/arabTIS_except_consensus_val_neg.txt', 'partitioned_data/arabTIS_except_consensus_test_pos.txt', 'partitioned_data/arabTIS_except_consensus_test_neg.txt'],
                  'SBBM_only_upstream': ['partitioned_data/arabTIS_only_upstream_train_pos.txt', 'partitioned_data/arabTIS_only_upstream_train_neg.txt', 'partitioned_data/arabTIS_only_upstream_val_pos.txt', 'partitioned_data/arabTIS_only_upstream_val_neg.txt', 'partitioned_data/arabTIS_only_upstream_test_pos.txt', 'partitioned_data/arabTIS_only_upstream_test_neg.txt'],
                  'SBBM_except_upstream': ['partitioned_data/arabTIS_except_upstream_train_pos.txt', 'partitioned_data/arabTIS_except_upstream_train_neg.txt', 'partitioned_data/arabTIS_except_upstream_val_pos.txt', 'partitioned_data/arabTIS_except_upstream_val_neg.txt', 'partitioned_data/arabTIS_except_upstream_test_pos.txt', 'partitioned_data/arabTIS_except_upstream_test_neg.txt'],
                  'SBBM_only_downstream': ['partitioned_data/arabTIS_only_downstream_train_pos.txt', 'partitioned_data/arabTIS_only_downstream_train_neg.txt', 'partitioned_data/arabTIS_only_downstream_val_pos.txt', 'partitioned_data/arabTIS_only_downstream_val_neg.txt', 'partitioned_data/arabTIS_only_downstream_test_pos.txt', 'partitioned_data/arabTIS_only_downstream_test_neg.txt'],
                  'SBBM_except_downstream': ['partitioned_data/arabTIS_except_downstream_train_pos.txt', 'partitioned_data/arabTIS_except_downstream_train_neg.txt', 'partitioned_data/arabTIS_except_downstream_val_pos.txt', 'partitioned_data/arabTIS_except_downstream_val_neg.txt', 'partitioned_data/arabTIS_except_downstream_test_pos.txt', 'partitioned_data/arabTIS_except_downstream_test_neg.txt'],
                  'SBBM_only_splice': ['partitioned_data/arabTIS_only_splice_train_pos.txt', 'partitioned_data/arabTIS_only_splice_train_neg.txt', 'partitioned_data/arabTIS_only_splice_val_pos.txt', 'partitioned_data/arabTIS_only_splice_val_neg.txt', 'partitioned_data/arabTIS_only_splice_test_pos.txt', 'partitioned_data/arabTIS_only_splice_test_neg.txt'],
                  'SBBM_except_splice': ['partitioned_data/arabTIS_except_splice_train_pos.txt', 'partitioned_data/arabTIS_except_splice_train_neg.txt', 'partitioned_data/arabTIS_except_splice_val_pos.txt', 'partitioned_data/arabTIS_except_splice_val_neg.txt', 'partitioned_data/arabTIS_except_splice_test_pos.txt', 'partitioned_data/arabTIS_except_splice_test_neg.txt'],
                  'SBBM_only_codonfreq': ['partitioned_data/arabTIS_only_codonfreq_train_pos.txt', 'partitioned_data/arabTIS_only_codonfreq_train_neg.txt', 'partitioned_data/arabTIS_only_codonfreq_val_pos.txt', 'partitioned_data/arabTIS_only_codonfreq_val_neg.txt', 'partitioned_data/arabTIS_only_codonfreq_test_pos.txt', 'partitioned_data/arabTIS_only_codonfreq_test_neg.txt'],
                  'SBBM_except_codonfreq': ['partitioned_data/arabTIS_except_codonfreq_train_pos.txt', 'partitioned_data/arabTIS_except_codonfreq_train_neg.txt', 'partitioned_data/arabTIS_except_codonfreq_val_pos.txt', 'partitioned_data/arabTIS_except_codonfreq_val_neg.txt', 'partitioned_data/arabTIS_except_codonfreq_test_pos.txt', 'partitioned_data/arabTIS_except_codonfreq_test_neg.txt']}
    # Default number of trials
    if len(sys.argv) == 2 and type(sys.argv[1]) == list:
        model_list = sys.argv[1]
        trials = 5
    # Use default files if not specified
    elif len(sys.argv) == 2 and type(sys.argv[1]) == int:
        model_list = list(model_dict.keys())
        trials = sys.argv[1]
    # Default files and number of trials
    elif len(sys.argv) == 1:
        model_list = list(model_dict.keys())
        trials = 5
    else:
        model_list = sys.argv[1]
        trials = sys.argv[2]
    # Log file name to save training and testing stages
    his_file = 'occluded models'
    for model_name in model_list:
        data = model_dict[model_list]
        model = Net()
        print(model)
        print(his_file)
        print(model_name)
        print('\n' + '='*50 + '\n')
        for i in range(trials):
            print("Trial number: " + str(i+1))
            run_model(data, 20, model, model_name, his_file)