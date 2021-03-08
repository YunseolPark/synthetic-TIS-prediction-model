'''
2020.10.20
Yunseol Park
'''

import torch
import torch.nn as nn
from func_TISmetrics import Assign
import code

def train_model(model, train_data, optimizer, criterion):
    """
    Function to train one epoch.

    Args:
        model: takes in the model to train
        train_data: takes in the data for training
        optimizer: optimizer
        criterion: takes in a torch function for loss calculation
    """
    model.train()
    for data, label in train_data:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        #print(loss)


def train_calc(model, train_data, criterion):
    """
    Calculate the loss and accuracy of the training model.

    Args:
        model: takes in the model to train
        train_data: takes in the data for training
        criterion: takes in a torch function for loss calculation
    Return:
        The accuracy and loss of the training model
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0
    size = 0
    with torch.no_grad():
        for data, label in train_data:
            output = model(data)
            loss = criterion(output, label)
            # Use softmax for getting prediction
            probs = nn.functional.softmax(output, 1)
            preds = torch.argmax(probs.data, 1)
            # Calculate total accuracy and loss
            total_acc += (preds == label).sum().item()
            total_loss += loss.item()
            size += label.size(0)
    # Get the average accuracy and loss
    accuracy = total_acc / size
    loss = total_loss / size
    return accuracy, loss


def test_model(model, test_data, criterion, saliency=None):
    """
    Function to test a model for one epoch

    Args:
        model: takes in a trained model for evaluation
        test_data: takes in the test data
        criterion: takes in a torch function for loss calculation
    return:
        the loss, accuracy, and a list containing TP, TN, FP, FN obtained from evaluation
    """
    model.eval()
    total_acc = 0
    total_loss = 0.0
    size = 0
    proportion = [0, 0, 0, 0]     # List to save TP, TN, FP, FN (in that order)
    if saliency != None:
        filename = open(saliency, 'w')
    with torch.no_grad():
        for data, label in test_data:
            output = model(data)
            loss = criterion(output, label)
            # Use softmax for getting prediction
            probs = nn.functional.softmax(output, 1)
            preds = torch.argmax(probs.data, 1)
            if saliency != None:
                code.interact(local = dict(globals(), **locals()))
                prob_write = [i[1] for i in probs.numpy()]
                label_write = [i for i in label]
                data_write = [i.numpy() for i in data]
                for i in range(len(prob_write)):
                    print(str(prob_write[i])+','+str(label_write[i]), file=filename)
                    print(data)
            # Calculate the total accuracy and loss
            total_acc += (preds == label).sum().item()
            #total_acc = total_acc / label.size(0)
            total_loss += loss.item()
            # Add TP/TN/FP/FN accordingly to list
            proportion = Assign(probs, label, proportion)
            size += label.size(0)
    # Calculate average accuracy and loss
    loss = total_loss / len(test_data)
    accuracy = total_acc / size
    #proportion = [3, 3, 2, 2]   # For testing only
    return loss, accuracy, proportion


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from cls_TISmodel import Net
    from cls_TISdataset import TISDataset

    pos_data = 'data/pos_cleaned.txt'
    neg_data = 'data/neg_cleaned.txt'
    data = TISDataset(pos_data, neg_data)
    train_set = data
    train_data = DataLoader(train_set, batch_size=1, shuffle=True)  # batch size: 256
    test_data = DataLoader(train_set, batch_size=3, shuffle=True)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = FL
    alpha = 0.25
    gamma = 2


    train_model(model, train_data, optimizer, criterion, gamma, alpha)
    loss, acc, propor = test_model(model, test_data, criterion, gamma, alpha)
    #loss, acc = train_acc_loss(model, train_data, criterion)
    #print(loss, acc)