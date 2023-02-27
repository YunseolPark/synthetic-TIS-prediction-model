import os
import csv
import torch
import sys
import pandas as pd
import code

def query_Y_N(question, default='yes'):
    """
    Function to ask a yes or no question

    Args:
        question: takes in a string that indicates the question/query
        default: takes in a string or None that indicates the default answer of query (default: yes)
    Return:
        A boolean that either indicates yes (True) or no (False) to the given query
    """
    query = {"yes": True, "y": True, "no": False, "n": False}
    # Set default
    if default is None:
        prompt = "[y/n]"
    elif default == "yes":
        prompt = "[Y/n]"
    elif default == "no":
        prompt = "[y/N]"
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)     # Ask question
        choice = input().lower()    # Get answer/input
        # When there is no choice but there is default value, then return the default value
        if default is not None and choice == '':
            return query[default]
        # When the input is in the query dictionary, then simply return the item of query
        elif choice in query:
            return query[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def rename(filename):
    """
    Function to rename a file name

    Args:
        filename: the original filename
    return:
        The new name for the file
    """
    # Isolate actual file name and extension
    #file = filename.split('.')
    # Check if there is already a file in the path
    check = os.path.isfile(filename)
    counter = 1
    while check:
        # Generate new name by adding (1) to the end of filename
        new_name = filename + '(' + str(counter) + ')'
        # + file[1]
        # Check if the new file name already exists in path
        check = os.path.isfile(new_name)
        counter += 1
    return new_name

#def saveHistory(header, value, path, filename, model, begin):
def saveHistory(history_dict, path, filename, model_name, save_type):
    """
    Function to save the history of training a model

    Args:
        header: takes in a list of strings indicating the header of the table
        value: takes in a list of values that are to be saved in the history file
        path: takes in a string indicating the directory that the file will be saved to
        filename: takes in a string indicating the file name that the file will be saved as
        model_name: takes in the name of the model of which the history will be saved
        begin: takes in a boolean that indicates whether it is the beginning of training (True) or not (False)
    """
    # If the directory does not exist, make the directory
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + '/' + filename + '.xlsx'
    # Check if file already exists in path
    #check = os.path.isfile(filename)
    history_dict = {key:value+[''] for key, value in history_dict.items()}
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    history_df = pd.DataFrame(history_dict)
    #code.interact(local = dict(globals(), **locals()))
    if model_name in writer.sheets:
        start_row = writer.sheets[model_name].max_row
    else:
        start_row = 0
    if save_type == 'train':
        history_df = history_df.set_index('epoch')
        history_df.to_excel(writer, sheet_name=model_name, startrow=start_row)
    else:
        history_df.to_excel(writer, sheet_name=model_name, startrow=start_row, index=False)
    writer.save()
    writer.close()

def saveModel(model, path, name):
    """
    Function to save the model to a file

    Args:
        model: takes in the name of the model to be saved
        path: takes in a string indicating the directory that the file will be saved to
        name: takes in a string indicating the file name that the file will be saved as
    """
    # If the directory does not exist, make the directory
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + '/' + name + '.pth'
    #+ '.pth'
    # Check if file already exists in path
    check = os.path.isfile(filename)
    if not check or 'tmp' in path:
        # When the file does not alredy exist or if it is a temp file, save to filename
        torch.save(model.state_dict(), filename)
    else:
        # Rename file and save it to the new filename
        filename = rename(filename)
        torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    saveHistory('history', 'test')