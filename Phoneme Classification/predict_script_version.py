import os
import torch
import numpy as np
import csv
from torch.nn import Module
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

# Search with small map - getitem() O(1) - memory O(n), but storing only end indices
class TestDataset(Dataset):
    def __init__(self, X_path):
        self.X = np.load(X_path, allow_pickle=True)
        
        self.height = CONTEXT*2+1
        
        self.sample_index = []
        length=0
        for i in range(len(self.X)):
            length+=self.X[i].shape[0]
            self.sample_index.append(length-1)
        self.length = length
        
    def __len__(self):
        return self.length
    
    def search(self, index):
        start = 0
        end = len(self.X)-1
        i = (start+end)//2
        while end - start >= 0 :
            # END CASE
            if i>0:
                if index <= self.sample_index[i] and index > self.sample_index[i-1]:
                    in_sample_index = index - self.sample_index[i-1] - 1
                    sample = i
                    return sample, in_sample_index
            else:
                if index <= self.sample_index[i]:
                    in_sample_index = index
                    sample = i
                    return sample, in_sample_index

            # CONTINUE SEARCH
            if index > self.sample_index[i]:
                start = i+1
                i = (start + end)//2
            elif index < self.sample_index[i]:
                end = i
                i = (start + end)//2
            else:
                raise Exception("unaccounted case")

    # keep this simple/quick because run many times
    def __getitem__(self, index):
        sample, i = self.search(index)
        
        if i<CONTEXT:
            X = self.X[sample][:i+CONTEXT+1]
            X = np.pad(X, pad_width=((self.height-X.shape[0],0),(0,0)), mode='constant', constant_values=0.)
        elif i >= self.X[sample].shape[0] - CONTEXT:
            X = self.X[sample][i-CONTEXT:]
            X = np.pad(X, pad_width=((0,self.height-X.shape[0]),(0,0)), mode='constant', constant_values=0.)
        else:
            X = self.X[sample][i-CONTEXT:i+CONTEXT+1]

        X=torch.flatten(torch.Tensor(X))
        
        return X

class Model(Module): 
    def __init__(self):
        super().__init__()

        layers = [
                  Linear(IN_OUT, int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), int(IN_OUT*1.4)),
                  ReLU(),
                  BatchNorm1d(int(IN_OUT*1.4)),
                  Linear(int(IN_OUT*1.4), 346)
        ]
        
        self.layers = Sequential(*layers)

    def forward(self,x):
        return self.layers(x)

# Code was transferred from a python notebook so not modularized
# Many parts of this code are based on the recitation code.
if __name__ == "__main__":
    
    cuda=True # TODO: Make sure you turn on/off CUDA depending on needs


    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 4 if cuda else 0
    batchsize = 256 if cuda else 64
    CONTEXT = 40
    IN_OUT = (CONTEXT*2+1)*13

    # Model Load
    model_path = 'hw1p2_models/model_7_4/model.pt'
    model = torch.load(model_path)
    print(model)

    num_workers = 4 if cuda else 0
    batchsize = 256 if cuda else 64

    print("Loading Test Set")
    # Test Set
    test_X_path = './test.npy'
    test_dataset = TestDataset(test_X_path)

    test_loader_args = dict(shuffle=False, batch_size=batchsize, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batchsize)

    test_loader = DataLoader(test_dataset, **test_loader_args)
    print("Test Set Loaded")

    sub_name = './submission5.csv' #<====================CHANGE THIS EVERY TIME=============<<<<<<<<

    def predict(model, test_loader, sub_name):
        submission = csv.writer(open(sub_name, "w"))
        submission.writerow(['id','label'])
        
        total = 0
        
        model.eval()
        for i, (x) in enumerate(test_loader):
            x = x.to(device)

            out = model(x)

            out = out.to("cpu")
            
            batch = torch.argmax(out, dim=1)
            for item in range(batch.shape[0]):
                total+=1
                submission.writerow([str(i*batchsize+item),str(batch[item].item())])
            
            if i%1000==0:
                print("Saved {} predicitons".format((i+1)*batchsize), end='\r')
        return total

    T=predict(model, test_loader, sub_name)
    print("{} Predictions COMPLETE".format(T))