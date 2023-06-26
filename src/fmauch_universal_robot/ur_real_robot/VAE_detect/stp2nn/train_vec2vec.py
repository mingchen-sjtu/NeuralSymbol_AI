import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import math

class VectorsDataset(Dataset):
    
    # run once when instantiating the Dataset object
    def __init__(self, init_vectors, next_vectors, labels):
        self.init_vectors = torch.tensor(init_vectors, dtype=torch.float32)
        self.next_vectors = torch.tensor(next_vectors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    # the number of samples in the dataset
    def __len__(self):

        return self.init_vectors.shape[0]

    # loads and returns a sample from the dataset at the given idx
    def __getitem__(self, idx):

        return self.init_vectors[idx], self.next_vectors[idx], self.labels[idx] 



def load_dataloader(train_data, test_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_dataloader, test_dataloader


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.action = nn.Sequential(nn.Flatten(),
                                    nn.Linear(64, 256),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(256, 128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, 64))

        self.object = nn.Sequential(nn.Flatten(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(32, 16),
                                    nn.Linear(16, 1),
                                    nn.Sigmoid())

    def forward(self, x, y):
        y_pred = self.action(y - x)
        object_score = self.object(y - x)
        return y_pred, object_score


class Train_Test():
    # key code
    def __init__(self, net, train_iter, test_iter, num_epochs):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.train_num_batches = len(train_iter)
        self.test_num_batches = len(test_iter)
        self.train_size = len(train_iter.dataset)
        self.test_size = len(test_iter.dataset)

        self.model = net
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.object_criteria = nn.BCELoss()
        self.num_epochs = num_epochs

        self.train_loss = 0
        self.test_loss = 0
        self.train_loss_list = []
        self.test_loss_list = []

    def action_criteria(self, y_pred, y, label):
        action_loss = torch.norm(y_pred - y) * label
        action_loss = torch.mean(action_loss, dim=-1)
        return action_loss


    def train(self):

        # iterate train dataloader
        for batch, (x, y, label) in enumerate(self.train_iter):
            self.model.train()
            self.optimizer.zero_grad()

            # loss backward
            y_pred, object_score = self.model(x, y)
            object_score = object_score.squeeze(-1)
            self.object_loss = self.object_criteria(object_score, label)
            self.action_loss = self.action_criteria(y_pred, y, label)

            # loss = self.object_loss
            loss = self.object_loss * 1 + self.action_loss
            loss.backward()

            # update parameters
            self.optimizer.step()

            # loss
            self.train_loss += loss.item()

        self.train_loss /= self.train_num_batches
        self.train_loss_list.append(self.train_loss)
        print(f"Train Avg Loss: {self.train_loss:>8f} \n")

    def test(self):
        with torch.no_grad():
            for test_x, test_y, test_label in self.test_iter:
                test_y_pred, test_object_score = self.model(test_x, test_y)
                test_object_score = test_object_score.squeeze(-1)
                self.test_object_loss = self.object_criteria(test_object_score, test_label)
                self.test_action_loss = self.action_criteria(test_y_pred, test_y, test_label)
                
                # test_loss = self.test_object_loss
                test_loss = self.test_object_loss * 1 + self.test_action_loss

                self.test_loss += test_loss.item()

        self.test_loss /= self.test_num_batches
        self.test_loss_list.append(self.test_loss)
        print(f"Test Avg Loss: {self.test_loss:>8f} \n")

    def train_test_loop(self):
        # iterate epochs
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")

            self.train()
            self.test()

    def plot_2d_line(self):
        '''
        line 2d plot
        '''

        fig = plt.figure(figsize=(6, 6))

        p1, = plt.plot(range(self.num_epochs), self.train_loss_list, color='blue', linestyle='-', linewidth=2, alpha=0.8)
        p2, = plt.plot(range(self.num_epochs), self.test_loss_list, color='red', linestyle='--', linewidth=2, alpha=0.8)

        # plt.axhline(23.5, ls="-.", lw=1, c='orange')
        plt.grid(color='grey')

        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend([p1, p2], ['train_loss', 'test_loss'], loc='upper right')

        # plt.savefig(save_path)
        plt.show()
        plt.close()


if __name__ == '__main__':

    all_vectors = np.load('vectors.npy') # (7328, 64)
    all_labels = np.load('labels.npy')

    # def. 4 types of vectors
    clear_aim_vectors = all_vectors[np.where(all_labels==0)]
    no_clear_aim_vectors = all_vectors[np.where(all_labels==1)]
    no_clear_no_aim_vectors = all_vectors[np.where(all_labels==2)]
    clear_no_aim_vectors = all_vectors[np.where(all_labels==3)]

    # prepare data for 'push' model
    data = []
    data_nums = 4000
    pos_ratio = 0.8

    # def. init_vectors
    init_idx = np.random.choice(no_clear_no_aim_vectors.shape[0], size=data_nums, replace=True)
    init_vectors = no_clear_no_aim_vectors[init_idx] # (1000, 64)

    # def. next_vectors
    pos_idx = np.random.choice(clear_no_aim_vectors.shape[0], size=round(data_nums*pos_ratio), replace=True)
    pos_vectors = clear_no_aim_vectors[pos_idx]

    neg_storage = np.vstack([clear_aim_vectors, no_clear_aim_vectors, no_clear_no_aim_vectors])
    neg_idx = np.random.choice(neg_storage.shape[0], size=round(data_nums*(1-pos_ratio)), replace=True)
    neg_vectors = neg_storage[neg_idx]

    next_vectors = np.vstack([pos_vectors, neg_vectors])
    labels = np.hstack([np.ones(round(data_nums*pos_ratio)), np.zeros(round(data_nums*(1-pos_ratio)))])

    # shuffle
    shuffle_idx = list(range(data_nums))
    random.shuffle(shuffle_idx)
    init_vectors = init_vectors[shuffle_idx]
    next_vectors = next_vectors[shuffle_idx]
    labels = labels[shuffle_idx]

    # Dataset
    train_data = VectorsDataset(init_vectors[ : int(data_nums * 0.7)], next_vectors[ : int(data_nums * 0.7)], labels[ : int(data_nums * 0.7)])
    test_data = VectorsDataset(init_vectors[int(data_nums * 0.7) : data_nums], \
                                next_vectors[int(data_nums * 0.7) : data_nums], labels[int(data_nums * 0.7) : data_nums])
    train_dataloader, test_dataloader = load_dataloader(train_data, test_data, batch_size=32)

    net = net()

    train_test = Train_Test(net, train_dataloader, test_dataloader, num_epochs=100)
    train_test.train_test_loop()
    train_test.plot_2d_line()