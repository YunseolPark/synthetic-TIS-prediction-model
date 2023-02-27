import code
import random
    
def data_partitioning(pos_file, neg_file, filename, ratio=[80,10], shuffle=False):
    pos_data = open(pos_file, 'r').readlines()
    neg_data = open(neg_file, 'r').readlines()
    if shuffle:
        random.shuffle(pos_data)
        random.shuffle(neg_data)
    train = round((ratio[0] / 100) * len(pos_data))
    test = round((1 - (ratio[1] / 100)) * len(pos_data))
    #code.interact(local = dict(globals(), **locals()))
    train_pos = open(filename+'_train_pos.txt', 'w')
    train_neg = open(filename+'_train_neg.txt', 'w')
    val_pos = open(filename+'_val_pos.txt', 'w')
    val_neg = open(filename+'_val_neg.txt', 'w')
    test_pos = open(filename+'_test_pos.txt', 'w')
    test_neg = open(filename+'_test_neg.txt', 'w')
    train_pos.write(''.join(i for i in pos_data[:train]))
    train_neg.write(''.join(i for i in neg_data[:train]))
    val_pos.write(''.join(i for i in pos_data[train:test]))
    val_neg.write(''.join(i for i in neg_data[train:test]))
    test_pos.write(''.join(i for i in pos_data[test:]))
    test_neg.write(''.join(i for i in neg_data[test:]))

def combine_dataset(syn_data, real_data, filename, ratio=(1,1), half='partitioned_data/at', shuffle=False):
    """
    Combine two datasets in the given ratio
    """
    syn_train_pos = open(syn_data[0], 'r').readlines()
    syn_val_pos = open(syn_data[2], 'r').readlines()
    syn_train_neg = open(syn_data[1], 'r').readlines()
    syn_val_neg = open(syn_data[3], 'r').readlines()
    real_train_pos = open(real_data[0], 'r').readlines()
    real_val_pos = open(real_data[2], 'r').readlines()
    real_train_neg = open(real_data[1], 'r').readlines()
    real_val_neg = open(real_data[3], 'r').readlines()
    save_train_pos = open(filename+'_train_pos.txt', 'w')
    save_train_neg = open(filename+'_train_neg.txt', 'w')
    save_val_pos = open(filename+'_val_pos.txt', 'w')
    save_val_neg = open(filename+'_val_neg.txt', 'w')
    
    synthetic = [syn_train_pos, syn_train_neg, syn_val_pos, syn_val_neg]
    real = [real_train_pos, real_train_neg, real_val_pos, real_val_neg]
    if shuffle:
        for syn in synthetic:
            random.shuffle(syn)
        for r in real:
            random.shuffle(real)
    if half:
        half_train_pos = open(half+'_half_train_pos.txt', 'w')
        half_train_neg = open(half+'_half_train_neg.txt', 'w')
        half_val_pos = open(half+'_half_val_pos.txt', 'w')
        half_val_neg = open(half+'_half_val_neg.txt', 'w')
        half_save = [half_train_pos, half_train_neg, half_val_pos, half_val_neg]
    else:
        half_save = ['','','','']
    save = [save_train_pos, save_train_neg, save_val_pos, save_val_neg]

    divisor = ratio[0] + ratio[1]
    for syn, r, s, hs in zip(synthetic, real, save, half_save):
        syn_len = ratio[0]*(len(syn)//divisor)
        real_len = len(r) - syn_len
        #code.interact(local = dict(globals(), **locals()))
        s.write(''.join(i for i in syn[:syn_len]))
        s.write(''.join(i for i in r[:real_len]))
        s.close()
        if half:
            hs.write(''.join(i for i in r[:real_len]))
            hs.close()

def half_data(real_data, filename, ratio=(1,2)):
    """
    Select only half of the data for partitioning (same half as the combined one)
    """
    real_train_pos = open(real_data[0], 'r').readlines()
    real_val_pos = open(real_data[2], 'r').readlines()
    real_train_neg = open(real_data[1], 'r').readlines()
    real_val_neg = open(real_data[3], 'r').readlines()
    save_train_pos = open(filename+'_half_train_pos.txt', 'w')
    save_train_neg = open(filename+'_half_train_neg.txt', 'w')
    save_val_pos = open(filename+'_half_val_pos.txt', 'w')
    save_val_neg = open(filename+'_half_val_neg.txt', 'w')
    save_train_pos.write(''.join(i for i in real_train_pos[:ratio[0]*len(real_train_pos)//ratio[1]]))
    save_train_neg.write(''.join(i for i in real_train_neg[:ratio[0]*len(real_train_neg)//ratio[1]]))
    save_val_pos.write(''.join(i for i in real_val_pos[:ratio[0]*len(real_val_pos)//ratio[1]]))
    save_val_neg.write(''.join(i for i in real_val_neg[:ratio[0]*len(real_val_neg)//ratio[1]]))


if __name__ == '__main__':
    '''data_partitioning('data/arabTIS_pos.txt', 'data/arabTIS.neg', 'data/at_pos_dic2013.txt', 'data/at_neg_dic2013.txt', ['data/combined_train_pos.txt', 'data/combined_train_neg.txt', 'data/combined_val_pos.txt', 'data/combined_val_neg.txt', 'data/combined_test_pos.txt', 'data/combined_test_neg.txt', ])'''
    #data_partitioning('data/arabTIS.pos', 'data/arabTIS.neg', 'test_data')
    #combine_dataset(['test_datatrain_pos.txt', 'test_datatrain_neg.txt', 'test_dataval_pos.txt', 'test_dataval_neg.txt', 'test_datatest_pos.txt', 'test_datatest_neg.txt'], ['test_datatrain_pos.txt', 'test_datatrain_neg.txt', 'test_dataval_pos.txt', 'test_dataval_neg.txt', 'test_datatest_pos.txt', 'test_datatest_neg.txt'], 'test_combine')
    #data_partitioning('data/arabTIS.pos', 'data/arabTIS.neg', 'new_data/arabTIS')
    #data_partitioning('data/at_pos_dic2013.txt', 'data/at_neg_dic2013.txt', 'new_data/at')
    #combine_dataset(['new_data/arabTIS_train_pos.txt', 'new_data/arabTIS_train_neg.txt', 'new_data/arabTIS_val_pos.txt', 'new_data/arabTIS_val_neg.txt', 'new_data/arabTIS_test_pos.txt', 'new_data/arabTIS_test_neg.txt'], 
    #                ['new_data/at_train_pos.txt', 'new_data/at_train_neg.txt', 'new_data/at_val_pos.txt', 'new_data/at_val_neg.txt', 'new_data/at_test_pos.txt', 'new_data/at_test_neg.txt'],
    #                'new_data/combined')
    #half_data(['partitioned_data/at_train_pos.txt', 'partitioned_data/at_train_neg.txt', 'partitioned_data/at_val_pos.txt', 'partitioned_data/at_val_neg.txt', 'partitioned_data/at_test_pos.txt', 'partitioned_data/at_test_neg.txt'], 'partitioned_data/at')
    '''combine_dataset(['data/arabTIS_train.pos', 'data/arabTIS_train.neg', 'data/arabTIS_val.pos', 'data/arabTIS_val.neg'], 
                    ['data/at_pos_train.txt', 'data/at_neg_train.txt', 'data/at_pos_val.txt', 'data/at_neg_val.txt'],
                    'data/1to2_combined', ratio=(2,1))'''
    #half_data(['data/at_pos_train.txt', 'data/at_neg_train.txt', 'data/at_pos_val.txt', 'data/at_neg_val.txt'], 'data/1to2_at', (1,3))
    #data_partitioning('data_generate/data/at_pos_dic2013.txt', 'data_generate/data/at_neg_dic2013.txt', 'partitioned_data/at')
    data_partitioning('data_generate/data/arabTIS_pos.txt', 'data_generate/data/arabTIS_neg.txt', 'partitioned_data/arabTIS')
    data_partitioning('data_generate/data/arabTIS_percentage_pos.txt', 'data_generate/data/arabTIS_percentage_neg.txt', 'partitioned_data/arabTIS_percentage')
    combine_dataset(['partitioned_data/arabTIS_percentage_train_pos.txt', 'partitioned_data/arabTIS_percentage_train_neg.txt', 'partitioned_data/arabTIS_percentage_val_pos.txt', 'partitioned_data/arabTIS_percentage_val_neg.txt', 'partitioned_data/arabTIS_percentage_test_pos.txt', 'partitioned_data/arabTIS_percentage_test_neg.txt'], 
                    ['partitioned_data/at_train_pos.txt', 'partitioned_data/at_train_neg.txt', 'partitioned_data/at_val_pos.txt', 'partitioned_data/at_val_neg.txt', 'partitioned_data/at_test_pos.txt', 'partitioned_data/at_test_neg.txt'],
                    'partitioned_data/combined_percentage')
    combine_dataset(['partitioned_data/arabTIS_train_pos.txt', 'partitioned_data/arabTIS_train_neg.txt', 'partitioned_data/arabTIS_val_pos.txt', 'partitioned_data/arabTIS_val_neg.txt', 'partitioned_data/arabTIS_test_pos.txt', 'partitioned_data/arabTIS_test_neg.txt'], 
                    ['partitioned_data/at_train_pos.txt', 'partitioned_data/at_train_neg.txt', 'partitioned_data/at_val_pos.txt', 'partitioned_data/at_val_neg.txt', 'partitioned_data/at_test_pos.txt', 'partitioned_data/at_test_neg.txt'],
                    'partitioned_data/combined')
    '''data_partitioning('data_generate/data/arabTIS_pos_codons.txt', 'data_generate/data/arabTIS_neg_codons.txt', 'partitioned_data/arabTIS_codons')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_only_upstream.txt', 'data_generate/data/missing_features/arabTIS_neg_only_upstream.txt', 'partitioned_data/arabTIS_only_upstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_only_consensus.txt', 'data_generate/data/missing_features/arabTIS_neg_only_consensus.txt', 'partitioned_data/arabTIS_only_consensus')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_only_splice.txt', 'data_generate/data/missing_features/arabTIS_neg_only_splice.txt', 'partitioned_data/arabTIS_only_splice')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_only_downstream.txt', 'data_generate/data/missing_features/arabTIS_neg_only_downstream.txt', 'partitioned_data/arabTIS_only_downstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_only_codonfreq.txt', 'data_generate/data/missing_features/arabTIS_neg_only_codonfreq.txt', 'partitioned_data/arabTIS_only_codonfreq')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_except_upstream.txt', 'data_generate/data/missing_features/arabTIS_neg_except_upstream.txt', 'partitioned_data/arabTIS_except_upstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_except_consensus.txt', 'data_generate/data/missing_features/arabTIS_neg_except_consensus.txt', 'partitioned_data/arabTIS_except_consensus')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_except_splice.txt', 'data_generate/data/missing_features/arabTIS_neg_except_splice.txt', 'partitioned_data/arabTIS_except_splice')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_except_downstream.txt', 'data_generate/data/missing_features/arabTIS_neg_except_downstream.txt', 'partitioned_data/arabTIS_except_downstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_pos_except_codonfreq.txt', 'data_generate/data/missing_features/arabTIS_neg_except_codonfreq.txt', 'partitioned_data/arabTIS_except_codonfreq')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_only_upstream.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_only_upstream.txt', 'partitioned_data/arabTIS_percentage_only_upstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_only_consensus.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_only_consensus.txt', 'partitioned_data/arabTIS_percentage_only_consensus')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_only_splice.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_only_splice.txt', 'partitioned_data/arabTIS_percentage_only_splice')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_only_downstream.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_only_downstream.txt', 'partitioned_data/arabTIS_percentage_only_downstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_only_codonfreq.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_only_codonfreq.txt', 'partitioned_data/arabTIS_percentage_only_codonfreq')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_except_upstream.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_except_upstream.txt', 'partitioned_data/arabTIS_percentage_except_upstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_except_consensus.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_except_consensus.txt', 'partitioned_data/arabTIS_percentage_except_consensus')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_except_splice.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_except_splice.txt', 'partitioned_data/arabTIS_percentage_except_splice')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_except_downstream.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_except_downstream.txt', 'partitioned_data/arabTIS_percentage_except_downstream')
    data_partitioning('data_generate/data/missing_features/arabTIS_percentage_pos_except_codonfreq.txt', 'data_generate/data/missing_features/arabTIS_percentage_neg_except_codonfreq.txt', 'partitioned_data/arabTIS_percentage_except_codonfreq')
    '''
