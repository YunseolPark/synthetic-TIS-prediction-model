'''
2020.10.09
Yunseol Park
'''

import torch
import torch.nn as nn
import code

class Net(nn.Module):
    """
    Class that creates the network architecture for the model
    """

    def convolution(self, in_shape, out_shape, kernel_size, maxpool, **kwargs):
        """
        Makes a sequential of a convolution block + maxpool

        Args:
            in_shape: in_channels for Conv1d
            out_shape: out_channels for Conv1d
            kernel_size: kernel size for Conv1d
            padding_size: size for padding
            maxpool_size: kernel size for Maxpool1d
            dropout_size: kernel size for Dropout
            kwargs: any other argumets for Conv1d
        Returns:
            the output of running the convolution block
        """
        output = nn.Sequential(
            nn.Conv1d(in_shape, out_shape, kernel_size=kernel_size, padding=kernel_size//2, **kwargs),
            nn.ReLU(),
            nn.MaxPool1d(maxpool)
            )
        return output

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = self.convolution(4, 100, kernel_size=9, maxpool=3)
        self.conv2 = self.convolution(100, 150, kernel_size=7, maxpool=3)
        self.conv3 = self.convolution(150, 200, kernel_size=7, maxpool=3)
        self.conv4 = self.convolution(200, 250, kernel_size=7, maxpool=3)
        self.conv5 = self.convolution(250, 300, kernel_size=5, maxpool=3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 512)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(512, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flat(x)
        x = self.fc1(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    #### the best so far is the above, without removing fc2. Now, I removed fc2 to se what happens #####
    #### the best is the above, 4 conv layers and one fc. ####
    from cls_TISdataset import TISDataset

    pos_data_loc = 'data/arabTIS.pos'
    neg_data_loc = 'data/arabTIS.neg'
    my_dataset = TISDataset(pos_data_loc, neg_data_loc)

    data_index = 2
    single_dna_data, label = my_dataset.__getitem__(data_index)
    single_dna_data.unsqueeze_(0)
    print(single_dna_data.size())

    my_network = Net()
    logits = my_network(single_dna_data)
    probs = nn.functional.softmax(logits, 1)
    preds = torch.argmax(probs, 1)
    ce = nn.CrossEntropyLoss()
    #print(label, preds)
    #loss = ce(logits, label)
