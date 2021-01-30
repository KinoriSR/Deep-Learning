import torch
import numpy as np
from torch.nn import CTCLoss, Linear, Module, LSTM, Conv1d, ReLU, Dropout#, LogSoftmax
from torch.nn.functional import log_softmax, softmax
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torchvision
from torch import optim
from ctcdecode import CTCBeamDecoder
from time import time
from Levenshtein import distance
import warnings
import csv
import os
warnings.simplefilter(action='ignore', category=DeprecationWarning)

N_PHONEMES = 41
PHONEME_LIST = [
    " ",
    "SIL",
    "SPN",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "H",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH"
]
# PHONEME_LIST.append('')

PHONEME_MAP = [
    " ",
    ".", #SIL
    "!", #SPN
    "a", #AA
    "A", #AE
    "h", #AH
    "o", #AO
    "w", #AW
    "y", #AY
    "b", #B
    "c", #CH
    "d", #D
    "D", #DH
    "e", #EH
    "r", #ER
    "E", #EY
    "f", #F
    "g", #G
    "H", #H
    "i", #IH 
    "I", #IY
    "j", #JH
    "k", #K
    "l", #L
    "m", #M
    "n", #N
    "N", #NG
    "O", #OW
    "Y", #OY
    "p", #P 
    "R", #R
    "s", #S
    "S", #SH
    "t", #T
    "T", #TH
    "u", #UH
    "U", #UW
    "v", #V
    "W", #W
    "?", #Y
    "z", #Z
    "Z" #ZH
]

class MyDataset(Dataset):
    def __init__(self, X_path, Y_path=None):
        self.X = np.load(X_path, allow_pickle=True)
        if Y_path:
            self.Y = np.load(Y_path, allow_pickle=True)
        else:
            self.Y=None
        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        X = self.X[index]
        if self.Y!=None:
            Y = self.Y[index] # to make sure 0 is reserved for blank
            return torch.DoubleTensor(X), torch.DoubleTensor(Y)+1, torch.LongTensor([X.shape[0]]), torch.LongTensor([Y.shape[0]]) # may have to do .float()/.long()
        else:
            return torch.DoubleTensor(X), torch.LongTensor([X.shape[0]])
class TestCollateFunction(object):
    def __init__(self):
        return
    def __call__(self, batch):
        X = []
        X_len = []
        for tup in batch:
            X.append(tup[0])
            X_len.append(tup[1])
        X = pad_sequence(X)
        X = X.permute(1,2,0)
        X_len = torch.cat(X_len,0).long()
        return X, X_len
    
class CollateFunction(object):
    def __init__(self):
        return
    def __call__(self, batch):
        X = []
        Y = []
        X_len = []
        Y_len = []
        for tup in batch:
            X.append(tup[0])
            Y.append(tup[1])
            X_len.append(tup[2])
            Y_len.append(tup[3])
        X = pad_sequence(X)
        X = X.permute(1,2,0)
        Y = pad_sequence(Y).permute(1,0) # Batch X 
        X_len = torch.cat(X_len,0)#.long()
        Y_len = torch.cat(Y_len,0)#.long()
        return X, Y, X_len, Y_len

class Model(Module):
    def __init__(self, out_vocab, embed_size, hidden_size, in_channel=13):
        super(Model, self).__init__()
        self.conv1d = Conv1d(in_channel, embed_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.lstm1 = LSTM(embed_size, hidden_size, bidirectional=True)
        self.lstm2 = LSTM(hidden_size*2, hidden_size, bidirectional=True)
        self.lstm3 = LSTM(hidden_size*2, hidden_size, bidirectional=True)
        self.predict = False
        
        self.linear3 = Linear(hidden_size*2, out_vocab+1)
        
    def forward(self, X, X_lens):
        X = self.conv1d(X) # CNN requires (batch_size, embedding_size, timesteps)

        X = X.permute(0,2,1)
        packed_X = pack_padded_sequence(X, X_lens, batch_first=True, enforce_sorted=False)
        
        packed_X = self.lstm1(packed_X)[0] # LSTM requires (timesteps, batch_size, embedding_size)
        packed_X = self.lstm2(packed_X)[0]
        packed_X = self.lstm3(packed_X)[0]
        
        X, out_lens = pad_packed_sequence(packed_X)
        
        X = self.linear3(X.permute(1,0,2))
        out = X.log_softmax(2)
        
        return out, out_lens

def log(file,string):
    file.write(string)
    return
# CTCDecode: https://github.com/parlance/ctcdecode
# for decoder must make sure the labels list is the same length as out_vocab (or in my case out_vocab+1)
# when discussing must make sure people understand that the ' ' character is the blank we are trying to account for
def get_decoder(labels, beam=10):
    return CTCBeamDecoder(labels, beam_width=beam, log_probs_input=True)

def predict(model, loader, labels, decoder=None, test=False):
    log_file = open("predict_decoder_logs.txt", "w")
    
    if not decoder:
        decoder = get_decoder(labels, beam=150)
    model.eval()
    model.predict = True
    sequences = []
    sequence_lens = []
    if test:
        for i, (x, x_len) in enumerate(loader):

            output, out_lens = model(x.to(device), x_len)

            log_string = 'output:' + str(output.shape) + str(output) + '\n' + 'out_lens: ' + str(out_lens.shape) + str(out_lens)
            log(log_file, log_string)

            # probabilities, out_lens are passed to ctcdecoder instead of loss
            beam_results, beam_scores, timesteps, out_lens = decoder.decode(output, out_lens)

            if i%10==0:
                print("Predict", i*batchsize, end='\r')

            sequences.append(beam_results.detach())
            sequence_lens.append(out_lens.detach())
        write_submission(sequences, sequence_lens)
    else:
        for i, (x, y, x_len, y_len) in enumerate(loader):
            x.to(device)

            output, out_lens = model(x.to(device), x_len)

            log_string = 'output:' + str(output.shape) + str(output) + '\n' + 'out_lens: ' + str(out_lens.shape) + str(out_lens)
            log(log_file, log_string)

            # probabilities, out_lens are passed to ctcdecoder instead of loss
            beam_results, beam_scores, timesteps, out_lens = decoder.decode(output, out_lens)

            print("Predict", i*batchsize, end='\r')

            sequences.append(beam_results.detach())
            sequence_lens.append(out_lens.detach())
        
    model.train()
    model.predict = False
    log_file.close()
    return sequences, sequence_lens

def write_submission(sequences, sequence_lens, sub_name='./submission.csv'):
    submission = csv.writer(open(sub_name, "w"))
    submission.writerow(['id','label'])
    Id=0
    for batch in range(len(sequences)):
        for i in range(len(sequences[batch])):
            encoded_row = sequences[batch][i,0,:sequence_lens[batch][i,0]]
            
            row = ''
            for phoneme in encoded_row:
                row += PHONEME_MAP[phoneme.item()]
            
            submission.writerow([Id,row])
            Id+=1
            if Id%10 == 0:
                print("Saved {} predictions".format(Id), end='\r')
    print("Submission File COMPLETE")
    
    return Id

if __name__ == "__main__":
    '''
    Change model_path's numbers in 'model_2_21' if want to load different model
    '''
    model_path = 'hw3p2_models/model_2_21/model.pt'

    cuda = True
    device = torch.device("cuda" if cuda else "cpu") 
    numworkers = 4 if cuda else 0
    batchsize = 64 if cuda else 64

    print('Loading Data')
    test_path = 'test.npy'
    test_dataset = MyDataset(test_path)
    testcollatefn = TestCollateFunction()
    test_loader_args = dict(shuffle=False, batch_size=batchsize, num_workers=numworkers, pin_memory=True, collate_fn=testcollatefn) if cuda else dict(shuffle=False, batch_size=batchsize, collate_fn=testcollatefn)
    test_loader = DataLoader(test_dataset, **test_loader_args)
    print('Data Loaded')

    model = torch.load(model_path).to(device)
    print('Model Loaded')

    print('Predicting...')
    predict(model, test_loader, PHONEME_LIST, test=True)
