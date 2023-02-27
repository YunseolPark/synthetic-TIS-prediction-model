def Assign(probs, label, proportion):
    """
    Function to assign the TP/TN/FP/FN from the given prediction and label

    Args:
        probs: takes in the probability of prediction of the sequence
        label: takes in the label of the sequence
        proportion: takes in a list that has the proportion of TP, TN, FP, FN
                    [TP, TN, FP, FN]
    return:
        The new proportion list
    """
    for p, l in zip(probs.data, label.data):
        true_prob = p[1]
        lab = l
        if true_prob >= 0.5 and lab == 1:
            proportion[0] += 1      # TP
        elif true_prob < 0.5 and lab == 0:
            proportion[1] += 1      # TN
        elif true_prob >= 0.5 and lab == 0:
            proportion[2] += 1      # FP
        elif true_prob < 0.5 and lab == 1:
            proportion[3] += 1      # FN
    return proportion

def Recall(proportion):
    """
    Function to calculate the recall

    Args:
        proportion: takes in a list that contains the number of TP, TN, FP, FN in that order
    return:
        the recall
    """
    TP, TN, FP, FN = proportion
    recall = TP / (TP + FN)
    return recall

def Precision(proportion):
    """
    Function to calculate the precision

    Args:
        proportion: takes in a list that contains the number of TP, TN, FP, FN in that order
    return:
        the precision
    """
    TP, TN, FP, FN = proportion
    precision = TP / (TP + FP)
    return precision

def F1(proportion):
    """
    Function to calculate the F1 score

    Args:
        proportion: takes in a list that contains the number of TP, TN, FP, FN in that order
    return:
        the F1 score
    """
    recall = Recall(proportion)
    precision = Precision(proportion)
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


if __name__ == '__main__':
    """
    def Accuracy(preds, label, batch_size):
        preds = np.array(preds)
        label = np.array(label)
        acc = 0
        for index in range(batch_size):
            one_pred = preds[index]
            one_label = label[index]
            check = np.equal(one_pred, one_label)
            add = np.sum(check)
            acc += add
        print('other', acc)
        acc = acc / batch_size
        return acc
    """
    Assign()