import time
import torch
import numpy as np
from Levenshtein import distance
import os
import csv
from util import plot_attn_flow, plot_grad_flow

### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

# def beam_search(output):
#     for i in range(output.shape[1]):
def get_sample(distribution, value = None):
    if not value:
        value = np.random.uniform() # not the most efficient
    i = 0
    cum_sum_prob = distribution[0]
    while cum_sum_prob < value:
        i += 1
        # i should never be greater than distribution[len(distribution)-1]
        cum_sum_prob += distribution[i]
    return i

# TODO: must softmax distribution within length
def random_search(output, n_samples=10):
    batch_samples = []
    # for each distribution in batch 
    for b in range(output.shape[0]):
        # _Ti=time.time()
        best_sequence = [[],0]
        # make samples
        for s in range(n_samples):
            # generate sample sequence
            sequence = [[],1.]            
            # for the time sequence
            for timestep in range(output.shape[1]):
                # print(output[b,time,:].shape)
                distribution = output[b,timestep,:].softmax(0)
                char = get_sample(distribution) # TODO check this
                sequence[0].append(char)
                sequence[1] *= distribution[char]
                if char == 34: #or len(sequence[0]) == max_len:
                    break
            if sequence[1] > best_sequence[1]:
                best_sequence = sequence
        # print(time.time()-_Ti)

        batch_samples.append(best_sequence[0])
    return batch_samples

def validate_accuracy(model, val_loader):
    model.eval()
    with torch.no_grad():
        num_correct = 0
        total = 0
        for i, (x, y, x_len, y_len) in enumerate(val_loader):
            x = x.to(DEVICE)
            # x_len = x_len.to(DEVICE)
            y = y

            out = model(x, x_len, text_input=y, isTrain=False, validate=True)

            # random search - ARGMAX FIRST
            # print(random_search(out))
            # print("val out shape {}, y shape {}".format(out.shape, y.shape))#, end='\r')
            batch_predictions = np.argmax(out.to('cpu').data, axis=2)
            for b in range(batch_predictions.shape[0]):
                for t in range(batch_predictions.shape[1]):
                    if t >= y_len[b].item(): # can just do a while loop here
                        break
                    if batch_predictions[b,t]==y[b,t]:
                        num_correct += 1
                    total += 1
            # num_correct += (batch_predictions == y).sum()
            # total += x_len.shape[0]
        accuracy = num_correct / total
            # pass each one through model
            # take one with smallest loss
            # compare chosen with ground truth
            # update accuracy

    model.train()
    return accuracy

'''
predict_print: the number of lines you want to print to see if things have improved
'''
def validate(model, val_loader, LETTER_LIST, criterion, predict_print=0, random=False, isAttended=True):
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        num_correct = 0
        total = 0
        L_dist = 0
        running_loss = 0
        for i, (x, y, x_len, y_len) in enumerate(val_loader):
            x = x.to(DEVICE)
            # x_len = x_len.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x, x_len, text_input=y, isTrain=False, validate=True, input_dropout=0)
            # random search

            if isAttended:
                loss = criterion(out.permute(0,2,1), y) # may not need to mask
            else:
                loss = criterion(out.permute(0,2,1), y[:,:out.shape[1]]) # may not need to mask
            # 8) Use the mask to calculate a masked loss.
            # mask = torch.arange(out.shape[1])[None,:] < out_len[:,None] # TODO: x_len??

            mask = torch.arange(out.shape[1])[None,:] >= y_len[:,None] # TODO: which one?  # loss
            mask = mask.to(DEVICE)                                                        # loss
            loss.masked_fill_(mask, 0.)
            loss = loss.sum()
            loss = loss/(mask.sum())                                                         # loss 
            running_loss += loss.item()
            torch.cuda.empty_cache()

            if random:
                batch_predictions = random_search(out)
                for b in range(len(batch_predictions)):
                    if predict_print and b == predict_print:
                        break
                    predict_string = ''
                    y_string = ''
                    letter_index = 33
                    for t in range(len(batch_predictions[b])):
                        # if batch_predictions[b,t]==y[b,t]:
                        letter_index = batch_predictions[b][t]
                        if letter_index == 34: # can just do a while loop here
                            break
                        predict_string += LETTER_LIST[letter_index].lower()
                        
                    for _t in range(y_len[b].item()):
                        letter_index = y[b,_t].item()
                        if letter_index == 34:
                            break
                        y_string += LETTER_LIST[letter_index].lower()
                        
                    # levenshtein distance
                    dist = distance(predict_string, y_string)
                    L_dist += dist
                    total += 1

                    if predict_print and b<predict_print:
                        print(predict_string)
                        print(y_string)
                        print(dist)
                    
            # greedy
            else:
                batch_predictions = np.argmax(out.to('cpu').data, axis=2)
                for b in range(batch_predictions.shape[0]):
                    predict_string = ''
                    y_string = ''
                    letter_index = 33
                    for t in range(batch_predictions.shape[1]):
                        # if batch_predictions[b,t]==y[b,t]:
                        letter_index = batch_predictions[b,t].item()
                        if letter_index == 34: # can just do a while loop here
                            break
                        predict_string += LETTER_LIST[letter_index].lower()
                        
                    for _t in range(y_len[b].item()):
                        letter_index = y[b,_t].item()
                        if letter_index == 34:
                            break
                        y_string += LETTER_LIST[letter_index].lower()
                        
                    # levenshtein distance
                    # if predict_print:
                    #     if i<predict_print:
                    # print(predict_string)
                    # print(y_string)
                    L_dist += distance(predict_string, y_string)
                    total += 1

                    if predict_print and b<predict_print:
                        print(predict_string)
                        print(y_string)
                        print(dist)

            if predict_print:
                break
            print("Validation Iteration {}      ".format(i), end='\r')
            # TODO: track loss here too
    model.train()
    loss = np.exp(running_loss/(i+1))
    return L_dist/total, loss

def train(model, train_loader, criterion, optimizer, epoch, batchsize, please_learn, model_version, model_number, isAttended=True, input_dropout=0):
    model.train()
    model.to(DEVICE)
    start = time.time()

    # 1) Iterate through your loader
    running_loss = 0
    for i, (x, y, x_len, y_len) in enumerate(train_loader):
        _ti = time.time()
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        # torch.autograd.set_detect_anomaly(True) # only while debugging
        optimizer.zero_grad()

        x = x.to(DEVICE)
        # x_len = x_len.to(DEVICE)
        y = y.to(DEVICE)

        if i==0:
            out, attention, context = model(x, x_len, text_input=y, teacher_force_rate=please_learn, plot_attn=True, input_dropout=input_dropout)
            path = './hw4p2_models/model_' + str(model_version) + '_' +str(model_number)
            if not os.path.isdir(path):
                os.makedirs(path)
            path_a = path + '/attention.png'
            path_c = path + '/context.png'
            plot_attn_flow(attention.detach().to('cpu'), path_a)
            plot_attn_flow(context.detach().to('cpu'), path_c)
            
            # print(model.parameters())
            # plot_grad_flow(model.parameters(), './hw4p2_models/model_' + str(model_version) + '_' + str(model_number) + '/gradients.png')


        else:
            out = model(x, x_len, text_input=y, teacher_force_rate=please_learn, input_dropout=input_dropout)
        
        # 5) Generate a mask based on the lengths of the text to create a masked loss. 

        # 6) If necessary, reshape your predictions and original text input TODO: TA edit "origianl"
        # 6.1) Use .contiguous() if you need to. 

        # TODO every 100 print results with greedy
        # if i>0 and i%200==0 and model_number>5:
        #     validate(model, train_loader, LETTER_LIST, criterion, predict_print=1, random=True)

        # 7) Use the criterion to get the loss.
        # print(out.permute(0,2,1).shape, y.shape)
        # print("out", out.permute(0,2,1))
        # print("y",y)
        if isAttended:
            loss = criterion(out.permute(0,2,1), y) # may not need to mask
        else:
            loss = criterion(out.permute(0,2,1), y[:,:out.shape[1]]) # may not need to mask
        # 8) Use the mask to calculate a masked loss.
        # mask = torch.arange(out.shape[1])[None,:] < out_len[:,None] # TODO: x_len??

        mask = torch.arange(out.shape[1])[None,:] >= y_len[:,None] # TODO: which one?  # loss
        mask = mask.to(DEVICE)                                                        # loss
        loss.masked_fill_(mask, 0.)
        loss = loss.sum()
        # loss=loss.mean()

        # 9) Run the backward pass on the masked loss.
        # print("loss.backward()")
        loss.backward()

        if i==1:
            plot_grad_flow(model.named_parameters(), './hw4p2_models/model_' + str(model_version) + '_' + str(model_number) + '/gradients.png')

        # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        # 11) Take a step with your optimizer
        optimizer.step()

        # 12) Normalize the masked loss
        loss = loss/(mask.sum())                                                         # loss 
        running_loss += loss.item()
        torch.cuda.empty_cache()

        _tf = time.time()
        print('Epoch:', epoch, '| Iteration:', i, '| Perplexity', np.exp(running_loss/(i+1)), '| Projected Time Left', ((28480//batchsize)-(i+1))*(_tf-_ti), '| Projected Time Total', ((28480//batchsize))*(_tf-_ti), end='\r')
        # print('Epoch:', epoch, '| Iteration:', i, '| Loss', loss.item())

    end = time.time()
    loss = np.exp(running_loss/(i+1))
    return loss, end-start

def test(model, test_loader, LETTER_LIST, predict_print=0, random=False, sub_name='submission.csv', isAttended=True):
    ### Write your test code here! ###
    # for i, (x, x_len) in enumerate(test_loader):
    # use other inferences as ground truth
    # normalize with length LAS paper pg.5-6
    start = time.time()
    submission = csv.writer(open(sub_name, "w"))
    submission.writerow(['id','label'])
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        total = 0
        for i, (x, x_len) in enumerate(test_loader):
            _start = time.time()
            x = x.to(DEVICE)
            # x_len = x_len.to(DEVICE)

            out = model(x, x_len, isTrain=False)

            torch.cuda.empty_cache()

            if random:
                batch_predictions = random_search(out, n_samples=1)
                for b in range(len(batch_predictions)):
                    if predict_print and b == predict_print:
                        break
                    predict_string = ''
                    letter_index = 33
                    for t in range(len(batch_predictions[b])):
                        # if batch_predictions[b,t]==y[b,t]:
                        letter_index = batch_predictions[b][t]
                        if letter_index == 34: # can just do a while loop here
                            break 
                        predict_string += LETTER_LIST[letter_index].lower()

                    submission.writerow([total, predict_string])
                    total += 1

                    if predict_print and b<predict_print:
                        print(predict_string)
                    
            # greedy
            else:
                batch_predictions = np.argmax(out.to('cpu').data, axis=2)
                for b in range(batch_predictions.shape[0]):
                    predict_string = ''
                    letter_index = 33
                    for t in range(batch_predictions.shape[1]):
                        # if batch_predictions[b,t]==y[b,t]:
                        letter_index = batch_predictions[b,t].item()
                        if letter_index == 34: # can just do a while loop here
                            break
                        predict_string += LETTER_LIST[letter_index].lower()
                    
                    submission.writerow([total, predict_string])
                    total += 1

                    if predict_print and b<predict_print:
                        print(predict_string)
            _end = time.time()
            print("Test Iteration {} | Time {}".format(i, _end-_start), end='\r')
    
    print("Submission File COMPLETE")
    model.train()
    end = time.time()
    return total, end-start


if __name__ == '__main__':
    distribution = [0.25,0.25,0.25,0.25]
    value = 0.55
    print(distribution, "random value:", value)
    print(get_sample(distribution, value))

    distribution = [0.15,0.33,0.12,0.22,0.17,0.01]
    value = 0.01
    print(distribution, "random value:", value)
    print(get_sample(distribution, value))
    value = 0.999
    print(distribution, "random value:", value)
    print(get_sample(distribution, value))