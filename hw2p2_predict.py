import os
import numpy as np
import torch
from PIL import Image
import csv
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.nn import Module, CosineSimilarity, Linear, BatchNorm1d, BatchNorm2d, Sequential, Dropout, ReLU, LeakyReLU, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, CrossEntropyLoss


class VerificationMyDataset(Dataset):
    def __init__(self, pair_path, test=False):
        pairs = open(pair_path, "r").readlines(-1)
        self.test = test
        self.x=[]
        if not self.test:
            self.y=[]
        self.to_tensor = ToTensor()
        self.length=len(pairs)
        for i in range(self.length):
            line = pairs[i].strip().split(' ')
            image1 = self.to_tensor(Image.open(line[0]))
            image2 = self.to_tensor(Image.open(line[1]))
            self.x.append((image1,image2))
            if not self.test:
                self.y.append(int(line[2]))
        
    def __len__(self):
        return self.length

    # keep this simple/quick because run many times
    def __getitem__(self, index):
        if not self.test:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

class BasicBlock(Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=False, drop_rate=0.3): # groups=1, dilation=1, norm_layer=None
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size-2, stride=stride, bias=False),  # , padding=kernel_size-1
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size-2, stride=1, bias=False),  # , padding=kernel_size-1
            BatchNorm2d(out_channels),
            ReLU()
        ]
        
        if dropout:
            layers.append(Dropout(drop_rate))
            
        self.block = Sequential(*layers)

    def forward(self,x):
        out = self.block(x)
         
        return out

class SkipDownSample(Module):
    def __init__(self, in_channels, out_channels, stride=1): # groups=1, dilation=1, norm_layer=None
        super().__init__()
    
        self.skip = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    
    def forward(self, x):
        return self.skip(x)
    
# Model_26_ 
class Model(Module):
    def __init__(self):
        super().__init__()
        
        # HEAD
        head = [
            Conv2d(3, 64, kernel_size=3, stride=1, bias=False), # 64x64
            MaxPool2d(kernel_size=3, stride=2, padding=1) # 32x32
        ]
        self.head = Sequential(*head)
        
        # Embedder
        self.block1 = BasicBlock(64,64)
        self.block2 = BasicBlock(64,64)
        
        self.block3 = Sequential(BasicBlock(64,128), MaxPool2d(kernel_size=3, stride=2, padding=1))#, stride=2) # 16x16
        self.skipdown1 = SkipDownSample(64,128, stride=2)             # MAYBE SHIFT THIS DOWN 1 SET OF BLOCKS
        self.block4 = BasicBlock(128,128)
        
        self.block5 = BasicBlock(128,256) # , stride=2) # 16x16
        self.skipdown2 = SkipDownSample(128,256) # , stride=2)
        self.block6 = BasicBlock(256,256)
        
        self.block7 = BasicBlock(256,512, stride=2) # 8x8
        self.skipdown3 = SkipDownSample(256,512, stride=2)            # MAYBE SHIFT THIS DOWN 1 SET OF BLOCKS
        self.block8 = BasicBlock(512,512)
        
        # I added
        self.block9 = BasicBlock(512, 1024) # 8x8
        self.skipdown4 = SkipDownSample(512,1024)
        
        self.block10 = BasicBlock(1024,2048) # 8x8
        self.skipdown5 = SkipDownSample(1024,2048)

        # EMBEDDING
        self.avgpool = AvgPool2d(kernel_size=8, stride=1) # kernel size 4 -> 8
        
        # CLASSIFICATION
        self.classification = Linear(2048, 4000)
    
    
    def forward(self, x):
        # HEAD
        # 64x64
        out = self.head(x)
        # 32x32
        # EMBEDDING
        out = self.block1(out) + out
        out = self.block2(out) + out
        out = self.block3(out) + self.skipdown1(out) # 16x16
        out = self.block4(out) + out
        out = self.block5(out) + self.skipdown2(out)# 8x8
        out = self.block6(out) + out
        out = self.block7(out) + self.skipdown3(out)# 4x4
        out = self.block8(out) + out
        out = self.block9(out) + self.skipdown4(out)# 2x2
        out = self.block10(out) + self.skipdown5(out)
        
        out = self.avgpool(out)
        embedding = torch.flatten(out,1)
        
        classification = self.classification(embedding)
                
        return embedding#, classification

def verify(model, data_loader, sub_name, test=False):
    model.eval()
    
    cos_sim = CosineSimilarity(dim=1)
    cosine_similarity = torch.Tensor([])
    true_similarity = torch.Tensor([])
    if not test:
        for i, (x,y) in enumerate(data_loader):
            img1 = x[0].to(device)
            img2 = x[1].to(device)
            y = y.to("cpu")
            
            out1 = model(img1).to("cpu")
            out2 = model(img2).to("cpu")
            
            cosine_similarity = torch.cat((cosine_similarity.detach(), cos_sim(out1,out2).detach()), 0)
            true_similarity = torch.cat((true_similarity, y), 0)

            del x, y, img1, img2, out1, out2
            torch.cuda.empty_cache()
            if i%10==0:
                print("Verification on validation set:", i*batchsize, end='\r')
            
        AUC = roc_auc_score(true_similarity, cosine_similarity.detach().numpy())
        return AUC
    else:
        for i, (x) in enumerate(data_loader):
            img1 = x[0].to(device)
            img2 = x[1].to(device)
            
            out1 = model(img1).to("cpu")
            out2 = model(img2).to("cpu")
            
            cosine_similarity = torch.cat((cosine_similarity.detach(), cos_sim(out1,out2).detach()), 0)
            if i%10==0:
                print("Verification on test set:", i*batchsize, end='\r')
        return write_submission(sub_name, cosine_similarity)

def write_submission(sub_name, cosine_similarity):
    pair_path = "verification_pairs_test.txt"
    pairs = open(pair_path, "r").readlines(-1)
    
    submission = csv.writer(open(sub_name, "w"))
    submission.writerow(['Id','Category'])

    self_length=len(pairs)
    for i in range(self_length):
        submission.writerow([pairs[i].strip(),cosine_similarity[i].item()])
        if i%10==0:
            print("Saved {} predicitons".format((i+1)*batchsize), end='\r')

    print("--- Submission File COMPLETE ---")
    return self_length

# Code was transferred from a python notebook so not modularized
# Many parts of this code are based on the recitation code.
if __name__ == "__main__":
    cuda = True
    batchsize = 128 if cuda else 64
    num_workers = 4 if cuda else 0
    device = torch.device("cuda" if cuda else "cpu")

    # pairs_path = "verification_pairs_val.txt"
    # ver_val_dataset = VerificationMyDataset(pairs_path)
    # ver_val_loader_args = dict(batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batchsize)
    # ver_val_loader = DataLoader(ver_val_dataset, **ver_val_loader_args)
    # print("Verification Validation Set Loaded")

    model_path = 'hw2p2_models/model_2_7/model.pt'
    model = torch.load(model_path).to(device)

    sub_name = './submission.csv'

    pairs_path = "verification_pairs_test.txt"

    test_dataset = VerificationMyDataset(pairs_path, test=True)
    test_loader_args = dict(batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batchsize)
    test_loader = DataLoader(test_dataset, **test_loader_args)
    print("Test Data Loaded")

    T = verify(model, test_loader, sub_name, test=True)
    print("{} Predictions COMPLETE".format(T))