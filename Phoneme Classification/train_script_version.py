import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU, CrossEntropyLoss
from torch import optim

# Search with small map - getitem() O(1) - memory O(n), but storing only end indices
class MyDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = np.load(X_path, allow_pickle=True)
        self.Y = np.load(Y_path, allow_pickle=True)
        self.height = CONTEXT*2+1
        
        self.sample_index = []
        length=0
        for i in range(len(self.Y)):
            length+=self.Y[i].shape[0]
            self.sample_index.append(length-1)
        self.length = length
        
    def __len__(self):
        return self.length
    
    # binary search through data which leverages self.sample_index to track end indices of each sample
    def search(self, index):
        start = 0
        end = len(self.Y)-1
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

    def __getitem__(self, index):
        sample, i = self.search(index)
        
        # pad
        if i<CONTEXT:
            X = self.X[sample][:i+CONTEXT+1]
            X = np.pad(X, pad_width=((self.height-X.shape[0],0),(0,0)), mode='constant', constant_values=0.)
        elif i >= self.Y[sample].shape[0] - CONTEXT:
            X = self.X[sample][i-CONTEXT:]
            X = np.pad(X, pad_width=((0,self.height-X.shape[0]),(0,0)), mode='constant', constant_values=0.)
        else:
            X = self.X[sample][i-CONTEXT:i+CONTEXT+1]

        Y = np.array(self.Y[sample][i])
        
        X=torch.flatten(torch.Tensor(X))
        Y=torch.from_numpy(Y)
        
        return X, Y

class Model(Module): 
    def __init__(self):
        super().__init__()

        # look up feed forward architecture - batch norm, deeper
        # In must be the input size, output can be increased - output of prev = input of next
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

# Validation Function
# This is based on the recitation code.
def validate(model, val_loader):
    model.eval()
    total = len(val_dataset)
    num_correct = 0
    for i, (x,y) in enumerate(val_loader):
        x = x.to(device)
        y = y.reshape(-1).to(device)

        out = model(x)

        out = out.to("cpu")
        y = y.to("cpu")

        batch_predictions = np.argmax(out.data, axis=1)
        num_correct += (batch_predictions == y).sum()
    accuracy = num_correct.item() / total
    model.train()
    return accuracy

def save_state(accuracy, model_number, model, train_loader_args, device, NUM_EPOCHS, learning_rate, optimizer, criterion):
    path = './hw1p2_models/model_' + str(RUN_NUMBER) + '_'+str(model_number)
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model, path+'/model.pt')
    # write parameter tracking file just in case I need them
    parameter_file = open(path+'/hyperparameters.txt', 'w')
    parameter_file.write('Accuracy:\n' + str(accuracy))
    parameter_file.write('\nContext:\n' + str(CONTEXT))
    parameter_file.write('\nModel:\n' + str(model))
    parameter_file.write('\ntrain_loader_args:\n' + str(train_loader_args))
    parameter_file.write('\nDevice:\n' + str(device))
    parameter_file.write('\nNUM_EPOCHS:\n' + str(NUM_EPOCHS))
    parameter_file.write('\nLearning Rate:\n' + str(learning_rate))
    parameter_file.write('\nOptimizer:\n' + str(optimizer))
    parameter_file.write('\nCriterion:\n' + str(criterion))
    parameter_file.close()

# Code was transferred from a python notebook so not modularized
# Many parts of this code are based on the recitation code.
if __name__ == "__main__":
    
    cuda = True # TODO: Make sure you turn on/off CUDA depending on needs


    # context for each data point
    CONTEXT = 40
    IN_OUT = (CONTEXT*2+1)*13 # base number of inputs for model layers
    
    num_workers = 4 if cuda else 0
    batchsize = 256 if cuda else 64

    # Validation Set
    # This code is based on the recitation code.
    print("Loading Validation Set")
    val_X_path = './dev.npy'
    val_Y_path = './dev_labels.npy'
    val_dataset = MyDataset(val_X_path, val_Y_path)
    val_loader_args = dict(shuffle=False, batch_size=batchsize, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batchsize)
    val_loader = DataLoader(val_dataset, **val_loader_args)
    print("Validation Set Loaded")

    # Training Set
    # This code is based on the recitation code.
    print("Loading Training Set")
    train_X_path = './train.npy' 
    train_Y_path = './train_labels.npy' 
    train_dataset = MyDataset(train_X_path, train_Y_path)
    droplast=False
    if len(train_dataset)%batchsize==1:
        print("Dropping last because last batch of size:", len(train_dataset)%batchsize)
        droplast=True
    train_loader_args = dict(shuffle=True, batch_size=batchsize, num_workers=num_workers, pin_memory=True, drop_last=droplast) if cuda else dict(shuffle=True, batch_size=batchsize, drop_last=droplast)
    train_loader = DataLoader(train_dataset, **train_loader_args)
    print("Training Set Loaded")

    # Initialize model
    model = Model()
    print(model)

    RUN_NUMBER = 7  # THIS WILL DEFINE PATH OF WHERE THE MODEL IS SAVED
    device = torch.device("cuda" if cuda else "cpu")
    NUM_EPOCHS = 5 # originally set to 30, but stopped at 5 because it was overfitting
    learning_rate = 1e-3
    mile_stones = [4] #,8,12,16,20,24]
    gamma = 0.1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stones, gamma=gamma)
    criterion = CrossEntropyLoss()
    model.to(device)

    # Training Loop
    # This code is based on the recitation code.
    print("Training")
    val_accuracies=[]
    model.train()
    model_number=0
    prev_acc = 0
    running_max = ['',0.]
    for epoch in range(NUM_EPOCHS):
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.reshape(-1).to(device)

            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            # progress
            if i%1000==0:
                print('Epoch:', epoch, '| Iteration:', i, end='\r')

        # validation
        accuracy = validate(model, val_loader)
        val_accuracies.append(accuracy)
        
        # save progress
        save_state(accuracy, model_number, model, train_loader_args, device, NUM_EPOCHS, learning_rate, optimizer, criterion)
        model_number+=1

        scheduler.step()
        
        # performance update
        print("Epoch", epoch, "Accuracy:", accuracy)
        if prev_acc == 0:
            print("\tImprovement:", accuracy-prev_acc)
        else:
            print("\tImprovement:", accuracy-prev_acc, "| Percent Improvement:", 100*(accuracy-prev_acc)/prev_acc, '%')
        if running_max[1]<accuracy:
            running_max[0]='Model_' + str(RUN_NUMBER) + '_' + str(epoch)
            running_max[1]=accuracy
        print('   Running Max:', *running_max,'\n')
        
        prev_acc = accuracy