import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cls_TISarchitecture import Net
from cls_TISdataset import TISDataset
from func_TISTrainEval import train_model, test_model, train_calc
from func_TISmetrics import Recall, Precision, F1
from func_TISsave import saveHistory, saveModel
#from func_focal_loss import FL
import code
from tqdm import tqdm
import os
import pandas as pd

"""
Main file that runs all the files 
"""

def freeze_layers(model, layer_list):
    #freeze layers that are only in the list
    for name, param in model.named_parameters():
        name_string = name.split('.')[0]
        if name_string not in layer_list:
            param.requires_grad = False
    #check that everything is done correctly
    print('Layers | Requires_grad')
    for name, param in model.named_parameters():
        print(name + ' | ' + str(param.requires_grad))
    print('\n' + '='*50 + '\n')
    #code.interact(local = dict(globals(), **locals()))

def initialize(data, model, layer_list=None):
    # Load train, validation, test data
    postrain_data, negtrain_data, posval_data, negval_data, postest_data, negtest_data = data
    train_set = TISDataset(postrain_data, negtrain_data)
    val_set = TISDataset(posval_data, negval_data)
    test_set = TISDataset(postest_data, negtest_data)
    batch_size = 256
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    # Freeze certain layers if using pre-trained model
    if layer_list:
        freeze_layers(model, layer_list)
    # Set parameters
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay = 0.5, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)	#0.00001
    criterion = nn.CrossEntropyLoss()
    return batch_size, train_data, val_data, test_data, optimizer, criterion, model

def training(batch_size, train_data, val_data, optimizer, criterion, n_epoch, model, model_name, his_file):
    print("Initializing training")
    best = 99999    # Best score; compare to validation loss to select model
    # Parameters for saving log file
    history_dict = {'epoch': [], 'train cost': [], 'validation cost': [], 'recall': [], 'precision': [], 'accuracy': [], 'F1': [], 'model selection': []}
    #header = ['epoch', 'train cost', 'validation cost', 'recall', 'precision', 'accuracy', 'F1', 'model selection']
    his_path = 'history'

    for epoch in range(1, n_epoch+1):
        #value = []      # List for saving value to save in log
        with tqdm(train_data, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            train_model(model, tepoch, optimizer, criterion, batch_size)
            # Calculate the training accuracy and loss
            accuracy, loss = train_calc(model, tepoch, criterion)
            # Validate model
            val_loss, val_acc, proportion = test_model(model, val_data, criterion)
            tepoch.set_postfix(loss=val_loss, accuracy=val_acc)
        # Calculate metrics
        recall = Recall(proportion)
        precision = Precision(proportion)
        f1 = F1(proportion)
        # Select model if validation loss is lower than best score
        if val_loss < best:
            tepoch.write("model selected [loss: {}, accuracy: {}]".format(val_loss, val_acc))
            best = val_loss
            saveModel(model, 'models/tmp', model_name)   # Save model to a temp folder
            model_selector = '-> model selected'
        else:
            model_selector = ''
        # Add values to add to history file
        history_dict = {key:value+[item] for (key, value), item in zip(history_dict.items(), [epoch, loss, val_loss, recall, precision, val_acc, f1, model_selector])}
        #code.interact(local = dict(globals(), **locals()))
    saveHistory(history_dict, his_path, his_file, model_name, 'train')   # Save history

    # Load and save model (the last model saved to the temp file)
    model = Net()
    model.load_state_dict(torch.load('models/tmp/'+model_name+'.pth'))
    #code.interact(local = dict(globals(), **locals()))
    saveModel(model, 'models', model_name)
    os.remove('models/tmp/'+model_name+'.pth')
    return model

def testing(test_data, criterion, model, model_name, his_file):
    # Start testing the model
    print('Testing model')
    # Set parameters for saving history file
    his_path = 'history'
    test_file = his_file + '_result'
    result_dict = {'recall': [], 'precision': [], 'accuracy': [], 'F1': [], 'TP': [], 'TN': [], 'FP': [], 'FN': []}
    # Test model
    test_loss, test_acc, test_proportion = test_model(model, test_data, criterion)
    # Calculate metrics
    test_recall = Recall(test_proportion)
    test_precision = Precision(test_proportion)
    test_f1 = F1(test_proportion)
    tp, tn, fp, fn = test_proportion
    # Save history
    result_dict = {key:value+[item] for (key, value), item in zip(result_dict.items(), [test_recall, test_precision, test_acc, test_f1, tp, tn, fp, fn])}
    saveHistory(result_dict, his_path, test_file, model_name, 'test')
    print('TP:{:^5d}, TN:{:^5d}, FP:{:^5d}, FN:{:^5d}\nrecall {:^2.8f} | precision {:^2.8f} | accuracy {:^2.8f} | F1 {:^2.8f}'
        .format(tp, tn, fp, fn, test_recall, test_precision, test_acc, test_f1))
    
def run_model(data, n_epoch, model, model_name, his_file, layer_list=None):
    batch_size, train_data, val_data, test_data, optimizer, criterion, model = initialize(data, model, layer_list)
    model = training(batch_size, train_data, val_data, optimizer, criterion, n_epoch, model, model_name, his_file)
    testing(test_data, criterion, model, model_name, his_file)