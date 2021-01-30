import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.functional import pad

'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    # destructive to transcript (may have to create version that doesn't destroy transcript)
    # mapped_transcript = []
    # for i in range(len(transcript)):
    #     mapped_transcript.append([])
    #     sentence = transcript[i]
    #     for j in range(len(sentence)):
    #         mapped_transcript[i].append([])
    #         word = sentence[j]
    #         for l in range(len(word)):
    #             letter = word[l]
    #             if letter >= 97 and letter <= 122:
    #                 mapped_transcript[i][j].append(letter - 97 + 1)
    #             elif letter == 39:
    #                 mapped_transcript[i][j].append(28)
    #             else:
    #                 print(i,j,l, word, letter)
    mapped_transcript = []
    for i in range(len(transcript)):
        mapped_transcript.append([])
        sentence = transcript[i]
        # <sos> tag TODO: do I need to train with an <sos> tag
        # mapped_transcript[i].append(33)
        for j in range(len(sentence)):
            # mapped_transcript[i].append([])
            word = sentence[j]
            for l in range(len(word)):
                letter = word[l]
                if letter >= 97 and letter <= 122:
                    mapped_transcript[i].append(letter - 97 + 1)
                elif letter == 39:
                    mapped_transcript[i].append(28)
                else:
                    print(i,j,l, word, letter)
            if not (j==len(sentence)-1):
                mapped_transcript[i].append(32)
        # <eos> tag
        mapped_transcript[i].append(34)
        mapped_transcript[i] = np.array(mapped_transcript[i])
    # mapped_transcript = transcript - 97
    # mapped_transcript[mapped_transcript == 39 - 97] = 28
    return mapped_transcript


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()

    # currently in lowercase because matches transcriptions
    for i in range(len(letter_list)):
        if not (letter_list[i] in letter2index):
            letter2index[letter_list[i].lower()] = i
            index2letter[i] = letter_list[i].lower()
            # can add case sensitivity later
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            # print(self.text[index].dtype)
            # print(self.speech[index].dtype, torch.tensor(self.text[index]))
            return torch.tensor(self.speech[index]), torch.LongTensor(np.array(self.text[index])) #.astype(np.float32)
        else:
            return torch.tensor(self.speech[index])# .astype(np.float32))


def collate_train(batch_data):
    X = []
    Y = []
    X_len = []
    Y_len = []
    for tup in batch_data:
        X.append(tup[0])
        X_len.append(torch.tensor([len(tup[0])]))
        Y.append(tup[1])
        Y_len.append(torch.tensor([len(tup[1])]))

    X = pad_sequence(X, batch_first=True)
    Y = pad_sequence(Y, batch_first=True)
    X_len = torch.cat(X_len,0)
    Y_len = torch.cat(Y_len,0)

    # sort by length
    X_len, idx = torch.sort(X_len, 0, descending=True)
    X = X[idx]
    Y = Y[idx]
    Y_len = Y_len[idx]
    ## Return the padded speech and text data, and the length of utterance and transcript ###
    return X, Y, X_len, Y_len


def collate_test(batch_data):
    X_len = []
    for tup in batch_data:
        X_len.append(torch.tensor([len(tup)]))
    X = pad_sequence(batch_data, batch_first=True)
    X_len = torch.cat(X_len,0)

    # need to keep proper order for writing file
    # X_len, idx = torch.sort(X_len, 0, descending=True)
    # X = X[idx]
    return X, X_len

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Unit testing:")

    LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

    # speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    print("\tCollate train - test")
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    print("\t (speech_valid loaded)")

    batch_size = 5
    valid_dataset = Speech2TextDataset(speech_valid, text=character_text_valid)
    result = collate_train([valid_dataset[0],valid_dataset[1]])


    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    # loop through loader
    print("\tLoop through train data...")
    for i, (x,y, x_len, y_len) in enumerate(valid_loader):
        print("\t x, y:", x.shape, x_len, "\t", y.shape, y_len)
        # print("\t\t", i, x.shape, x_len.shape, y.shape, y_len.shape)
        if i==4:
            break

    print("\tCollate test - test")
    speech_test = np.load('test.npy', allow_pickle=True, encoding='bytes')
    print("\t (speech_test loaded)")

    test_dataset = Speech2TextDataset(speech_test, None, False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)
    print("\tLoop through test data...")
    for i, (x, x_len) in enumerate(test_loader):
        print("\t x.shape, x_len:", x.shape, x_len)
        # print("\t\t", i, x.shape, x_len.shape)
        if i==4:
            break

    print("\tTest runs")

