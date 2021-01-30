import os
import numpy as np
import torch
from PIL import Image
import csv
from time import time
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.nn import Module
from torch.nn import Linear, CosineSimilarity, BatchNorm1d, BatchNorm2d, Sequential, Dropout, ReLU, LeakyReLU, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, CrossEntropyLoss

class VerificationMyDataset(Dataset):
    def __init__(self, pair_path, test=False):
        pairs = open(pair_path, "r").readlines(-1)
        self.test = test
        self.x=[]
        if not self.test:
            self.y=[]
        self.to_tensor = transforms.ToTensor()
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

# Based on ResNet 18 with modifications
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
        self.skipdown1 = SkipDownSample(64,128, stride=2)
        self.block4 = BasicBlock(128,128)
        
        self.block5 = BasicBlock(128,256) # , stride=2) # 16x16
        self.skipdown2 = SkipDownSample(128,256) # , stride=2)
        self.block6 = BasicBlock(256,256)
        
        self.block7 = BasicBlock(256,512, stride=2) # 8x8
        self.skipdown3 = SkipDownSample(256,512, stride=2)
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
                
        return embedding, classification

def validate(model, val_loader, val_dataset):
    model.eval()
    total = len(val_dataset)
    num_correct = 0
    for i, (x,y) in enumerate(val_loader):
        x = x.to(device)
        y = y.reshape(-1).to(device)

        out = model(x)[1]

        out = out.to("cpu")
        y = y.to("cpu")

        batch_predictions = np.argmax(out.data, axis=1)
        num_correct += (batch_predictions == y).sum()
    
    accuracy = num_correct.item() / total
        
    # Deallocate memory in GPU
    torch.cuda.empty_cache()
    del x
    del y
    
    model.train()
    return accuracy

def save_state(AUC, accuracy, running_max, model_number, model, train_loader_args, device, NUM_EPOCHS, learning_rate, optimizer, criterion):
    path = './hw2p2_models/model_' + str(RUN_NUMBER) + '_'+str(model_number)
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model, path+'/model.pt')
    # write parameter tracking file
    parameter_file = open(path+'/hyperparameters.txt', 'w')
    parameter_file.write('AUC:\n' + str(AUC))
    parameter_file.write('Accuracy:\n' + str(accuracy))
    parameter_file.write('Running Max:\n' + str(running_max[0]) + "  " + str(running_max[1]))
    parameter_file.write('\nModel:\n' + str(model))
    parameter_file.write('\ntrain_loader_args:\n' + str(train_loader_args))
    parameter_file.write('\nDevice:\n' + str(device))
    parameter_file.write('\nNUM_EPOCHS:\n' + str(NUM_EPOCHS))
    parameter_file.write('\nLearning Rate:\n' + str(learning_rate))
    parameter_file.write('\nOptimizer:\n' + str(optimizer))
    parameter_file.write('\nCriterion:\n' + str(criterion))
    parameter_file.close()
    

sub_name = './submission6.csv'
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
            
            out1 = model(img1)[0].to("cpu")
            out2 = model(img2)[0].to("cpu")
            
            cosine_similarity = torch.cat((cosine_similarity.detach(), cos_sim(out1,out2).detach()), 0)
            true_similarity = torch.cat((true_similarity, y), 0)

            del x, y, img1, img2, out1, out2
            torch.cuda.empty_cache()
        model.train()
        try:  
            AUC = roc_auc_score(true_similarity.type(torch.DoubleTensor), cosine_similarity.type(torch.DoubleTensor).detach().numpy())
            return AUC
        except Exception as e:
            print(e)
            return -1
    else:
        for i, (x) in enumerate(data_loader):
            img1 = x[0].to(device)
            img2 = x[1].to(device)
            
            out1 = model(img1)[0].to("cpu")
            out2 = model(img2)[0].to("cpu")
            
            cosine_similarity = torch.cat((cosine_similarity.detach(), cos_sim(out1,out2).detach()), 0)
            if i%1000==0:
                print("Verification", i, end='\r')
        model.train()
        return write_submission(sub_name, cosine_similarity)

def write_submission(sub_name, cosine_similarity):
    pair_path = "verification_pairs_test.txt"
    pairs = open(pair_path, "r").readlines(-1)
    
    submission = csv.writer(open(sub_name, "w"))
    submission.writerow(['Id','Category'])

    self_length=len(pairs)
    for i in range(self_length):
        submission.writerow([pairs[i].strip(),cosine_similarity[i].item()])
        if i%1000==0:
            print("Saved {} predicitons".format((i+1)*batchsize), end='\r')

    print("Submission File COMPLETE")
    return self_length

# Code was transferred from a python notebook so not modularized
# Many parts of this code are based on the recitation code.
if __name__ == "__main__":
    cuda = True
    batchsize = 256 if cuda else 64
    num_workers = 4 if cuda else 0
    # from Recitation 5
    # not doing horizontal flip as features can be side dependent, not doing RandomGrayScale to start
    transformations = transforms.Compose([
    #     transforms.Resize((68,68)),
    #     transforms.RandomCrop((64,64)), # this should help with translation variants
    #     transforms.ColorJitter(brightness=0.09, contrast=0.09, saturation=0.09),  
    #     transforms.RandomRotation(degrees=5), #20
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root='classification_data/train_data/', transform=transformations)#ToTensor())
    droplast=False
    if len(train_dataset)%batchsize==1:
        print("Dropping last because last batch of size:", len(train_dataset)%batchsize)
        droplast=True
    train_loader_args = dict(batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=droplast) if cuda else dict(shuffle=True, batch_size=batchsize, drop_last=droplast)
    train_loader = DataLoader(train_dataset, **train_loader_args)
    print("Classification Training Set Loaded")

    val_dataset = ImageFolder(root='classification_data/val_data/', transform=transforms.ToTensor())
    val_loader_args = dict(batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batchsize)
    val_loader = DataLoader(val_dataset, **val_loader_args)
    print("Classification Validation Set Loaded")

    pairs_path = "verification_pairs_val.txt"
    ver_val_dataset = VerificationMyDataset(pairs_path)
    ver_val_loader_args = dict(batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batchsize)
    ver_val_loader = DataLoader(ver_val_dataset, **ver_val_loader_args)
    print("Verification Validation Set Loaded")

    RUN_NUMBER = 1

    device = torch.device("cuda" if cuda else "cpu")
    model = Model()
    model.to(device)
    print(model)
    NUM_EPOCHS = 6
    learning_rate = 1e-1
    mile_stones = [4]
    gamma = 0.2
    optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-5, lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stones, gamma=gamma)
    criterion = CrossEntropyLoss()

    print("Start Training")
    accuracy = 0.
    model_number = 0 
    prev_acc = 0
    running_max = ['',0.]
    for epoch in range(NUM_EPOCHS):
        ti = time()
        model.train()
        for i, (x,y) in enumerate(train_loader):
            
            optimizer.zero_grad()

            x = x.to(device)
            y = y.reshape(-1).to(device) # need to turn to row

            output = model(x)[1]
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            # progress
            if i%10==0:
                print('Epoch:', epoch, '| Iteration:', i, end='\r')

        
        # Deallocate memory in GPU
        torch.cuda.empty_cache()
        del x
        del y

        # validation
        accuracy = validate(model, val_loader, val_dataset)
        AUC = verify(model, ver_val_loader, sub_name)

        print("Epoch", epoch, "Accuracy:", accuracy, "AUC:", AUC)
        if prev_acc == 0:
            print("\tImprovement:", accuracy-prev_acc)
        else:
            print("\tImprovement:", accuracy-prev_acc, "| Percent Improvement:", 100*(accuracy-prev_acc)/prev_acc, '%')
        
        # tracking running best AUC
        if running_max[1]<AUC:
            running_max[0]='Model_' + str(RUN_NUMBER) + '_' + str(epoch)
            running_max[1]=AUC
            
        if (AUC > 0.8):
            save_state(AUC, accuracy, running_max, model_number, model, train_loader_args, device, NUM_EPOCHS, learning_rate, optimizer, criterion)
        model_number+=1

        scheduler.step()
        
        tf=time()
        print("Time for epoch:", tf-ti)
        print('   Running Max:', *running_max,'\n')

        prev_acc = accuracy
