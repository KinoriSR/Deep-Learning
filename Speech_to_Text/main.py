import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models import *
from train_test import *
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset
from util import plot_attn_flow, plot_grad_flow

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(DEVICE)

# TODO: TA change make lowercase
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def save_state(Levenshtein_dist, running_best, model_version, model_number, model, optimizer, criterion, batch_size):
    path = './hw4p2_models/model_' + str(model_version) + '_'+str(model_number)
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model, path+'/model.pt')
    # write parameter tracking file
    parameter_file = open(path+'/hyperparameters.txt', 'w')
    parameter_file.write('Levenshtein Distance:\n' + str(Levenshtein_dist))
    parameter_file.write('\nRunning Best Levenshtein_dist:\n' + str(running_best[0]) + "  " + str(running_best[1]))
    parameter_file.write('\nbatch_size:\n' + str(batch_size))
    parameter_file.write('\nOptimizer:\n' + str(optimizer))
    parameter_file.write('\nCriterion:\n' + str(criterion))
    parameter_file.write('\nModel:\n' + str(model))
    parameter_file.close()

'''
Get the running best of the specified model version at the model number
'''
def get_running_best(model_version, model_number):
    path = './hw4p2_models/model_' + str(model_version) + '_'+str(model_number)
    parameter_file = open(path+'/hyperparameters.txt', 'r').readlines(-1)
    for l in range(4):
        line = parameter_file[l].split(" ")
    if len(line) == 2:
        return [line[0], float(line[1])]
    else:
        return ['', 1000]

'''
action:
    train - initialize a model and train
    load - load a model and train
    test - predicitons on test set
'''
def main(action="train", model_version=-1, model_number=0, submission_name='submission.csv'):
    # _____-----**********-----__________-----**********-----_____ CHECK THIS *****-----__________-----**********-----__________-----*****
    isAttended = True
    # _____-----**********-----__________-----**********-----_____ CHECK THIS *****-----__________-----**********-----__________-----*****

    if action in ["load", "test", "validate"]:
        path = "./hw4p2_models/model_" + str(model_version) + "_" + str(model_number) + "/model.pt"
        print("Loading model from: {}".format(path))
        model = torch.load(path)
    else:
        print("Initializing NEW model version {}, model number {}".format(model_version, model_number))
        model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=512, value_size=128, key_size=128, isAttended=isAttended)
        # model 3: hidden_dim=256, value_size=128, key_size=128
        # model 4: hidden_dim=512, value_size=128, key_size=128 (helped - best so far)
        # model 5: hidden_dim=512, value_size=256, key_size=256 (not much gained)
        # model 6: hidden_dim=512, value_size=150, key_size=150 input_dropout before first LSTM [(7, 0.15), (10, 0.2)] (no help)
        # model 7: hidden_dim=512, value_size=128, key_size=128 conv1d k=5, pad=2, stride=1, accidental input_dropout of 0.2 later
        # model 8: hidden_dim=512, value_size=128, key_size=128 conv1d k=5, pad=2, stride=1
        # model 9: hidden_dim=512, value_size=128, key_size=128 locked dropout, batchnorm1d between pBLSTM layers
        # model 10: hidden_dim=512, value_size=128, key_size=128 locked dropout (up then down), batchnorm1d between pBLSTM layers, weight decay



    nepochs = 70
    batch_size = 64 if DEVICE == 'cuda' else 1
    num_workers = 4 if DEVICE == 'cuda' else 0

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    criterion = nn.CrossEntropyLoss(reduction='none') # TODO: TA change reduction=None to 'none'

    if action == "train":
        print("Start normal training...")
        learning_rate = 0.001
        mile_stones = [10,15,20,30] # [5,10,15] # [4,7,10,13,16,19,22,25] #
        gamma = 0.1 # changed from 0.3 after looking at models 4, 5
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-6) # TODO: tune
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stones, gamma=gamma)

        input_dropout = 0.
        # [(epoch, input_dropout_prob),]
        input_dropout_schedule = [(15, 0.1), (20, 0.15), (25, 0.2), (30, 0.1), (35, 0.)]

        train_dataset = Speech2TextDataset(speech_train, character_text_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=num_workers, pin_memory=True)

        val_dataset = Speech2TextDataset(speech_valid, text=character_text_valid)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train, num_workers=num_workers, pin_memory=True)

        running_best = ['', 1000.]
        please_learn = 1.
        for epoch in range(nepochs):
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")

            if input_dropout_schedule:
                if input_dropout_schedule[0][0] == epoch:
                    input_dropout = input_dropout_schedule[0][1]
                    input_dropout_schedule = input_dropout_schedule[1:]

            if (epoch+1)%5==0:
                please_learn -= 0.5/(40/5) # by epoch 40 need to be at 50%, reduce every 5 epochs
            model.train()

            loss, run_time = train(model, train_loader, criterion, optimizer, epoch, batch_size, please_learn, model_version, model_number, isAttended, input_dropout=input_dropout)
            
            # plot_grad_flow(model.named_parameters(), './hw4p2_models/model_' + str(model_version) + '_' + str(model_number) + '/gradients.png')
            
            Levenshtein_dist, val_loss = validate(model, val_loader, LETTER_LIST, criterion)

            # Update Me
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Epoch", epoch, "Levenshtein_dist:", Levenshtein_dist, "Perplexity:", loss.item(), "Val Perplexity:", val_loss)
            print("\tTuning Status: Input Dropout = {}, Teacher Forcing = {}".format(input_dropout, please_learn))
            if running_best[1] > Levenshtein_dist:
                running_best[0] = 'Model_' + str(model_version) + '_' + str(model_number)
                running_best[1] = Levenshtein_dist
            print("\tTime for Epoch:", run_time)
            print("\tRunning Best:", *running_best)
            scheduler.step()
            
            save_state(Levenshtein_dist, running_best, model_version, model_number, model, optimizer, criterion, batch_size)
            model_number+=1
    
    elif action == "load":
        print("Start training loaded model...")
        learning_rate = 0.001
        mile_stones = [] # [3,8,13,18,23] # [5,10,15] # [4,7,10,13,16,19,22,25] #
        gamma = 0.1
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-6) # TODO: tune
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stones, gamma=gamma)

        input_dropout = 0. #0.2
        # [(epoch, input_dropout_prob),]
        input_dropout_schedule = []#[(7, 0.15), (10, 0.2)]

        criterion = nn.CrossEntropyLoss(reduction='none') # TODO: TA change reduction=None to 'none'

        train_dataset = Speech2TextDataset(speech_train, character_text_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=num_workers, pin_memory=True)

        val_dataset = Speech2TextDataset(speech_valid, text=character_text_valid)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train, num_workers=num_workers, pin_memory=True)

        running_best = get_running_best(model_version, model_number)
        # if model_number > 10:
        #     please_learn = 1.
        # else:
        #     please_learn = 1. - (model_number//5)*(0.5/8)
        please_learn = 0.8

        model_number += 1
        for epoch in range(model_number, nepochs):
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")
            
            if input_dropout_schedule:
                if input_dropout_schedule[0][0] == epoch:
                    input_dropout = input_dropout_schedule[0][1]
                    input_dropout_schedule = input_dropout_schedule[1:]

            if model_number > 10 and (model_number+1)%5==0:
                please_learn -= 0.5/(40/5)
            model.train()
            loss, run_time = train(model, train_loader, criterion, optimizer, epoch, batch_size, please_learn, model_version, model_number, isAttended, input_dropout=input_dropout)
            
            Levenshtein_dist, val_loss = validate(model, val_loader, LETTER_LIST, criterion)

            # Update Me
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Epoch", epoch, "Levenshtein_dist:", Levenshtein_dist, "Perplexity:", loss.item(), "Val Perplexity:", val_loss)
            if running_best[1] > Levenshtein_dist:
                running_best[0] = 'Model_' + str(model_version) + '_' + str(model_number)
                running_best[1] = Levenshtein_dist
            print("\tTuning Status: Input Dropout = {}, Teacher Forcing = {}".format(input_dropout, please_learn))
            print("\tTime for Epoch:", run_time)
            print("\tRunning Best:", *running_best)
            scheduler.step()

            save_state(Levenshtein_dist, running_best, model_version, model_number, model, optimizer, criterion, batch_size)
            model_number+=1

    elif action == "test":
        print("Start prediction...")
        test_dataset = Speech2TextDataset(speech_test, None, False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

        n, time = test(model, test_loader, LETTER_LIST, random=False, sub_name=submission_name)

        print("{} Predictions COMPLETE in {}".format(n, time))
    elif action == "validate":
        print("Start Validation...")
        val_dataset = Speech2TextDataset(speech_valid, text=character_text_valid)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train, num_workers=num_workers, pin_memory=True)

        Levenshtein_dist, val_loss = validate(model, val_loader, LETTER_LIST, criterion) #, random=True)
        print("Levenshtein Distance:", Levenshtein_dist, "Validation Loss:", val_loss)


def train_test():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction='none') # TODO: TA change reduction=None to 'none'
    nepochs = 1
    batch_size = 64 if DEVICE == 'cuda' else 1
    
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)
    
    val_dataset = Speech2TextDataset(speech_valid, text=character_text_valid)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    
    print("train() Test:")
    running_best = ['',0.]
    for epoch in range(nepochs):
        print("\t epoch", epoch)
        train(model, val_loader, criterion, optimizer, epoch, batch_size, 1.0)
        validate(model, val_loader)
    print("Runs!")


'''
`python3 main.py action model_version model_number`
train: train new model with specified model_version
load: load specified model
test: test specified model
validate: validate specified model
'''
if __name__ == '__main__':
    import sys

    args = sys.argv
    print(args)
    if args[1]:
        action = args[1]
        if len(args) == 3:
            model_version = int(args[2])
            main(action=action,model_version=model_version)
        if len(args) == 4:
            model_version = int(args[2])
            model_number = int(args[3])
            main(action=action, model_version=model_version, model_number=model_number)

    else:
        main()
    # elif args[1]=="test":
    #     model_number = args[2]
    #     version = args[3]
    #     test(model_number, version)
    # elif args[1]=="val":
    #     model_number = args[2]
    #     version = args[3]
    #     validate(model_number, version)
    # train_test()
    # main()