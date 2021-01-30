import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils as utils
from torch.nn.functional import softmax, pad, gumbel_softmax # log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from util import plot_attn_flow, plot_grad_flow

from torch.utils.data import DataLoader
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# LockedDropout module from torchnlp docs
class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(batch_size, hidden_size) Query is the output of LSTMCell from Decoder
        :param keys: (batch_size, max_len, encoder_size) Key Projection from Encoder
        :param values: (batch_size, max_len, encoder_size) Value Projection from Encoder
        :return context: (batch_size, encoder_size) Attended Context
        :return attention_mask: (batch_size, max_len) Attention mask that can be plotted  
        '''
        energy = torch.bmm(key, query.unsqueeze(2).to(DEVICE))
        # mask using lens when getting softmax create mask... found a faster way than looping somewhere online
        mask = (torch.arange(energy.shape[1])[None,:] >= lens[:,None]).to(DEVICE) # < for positive mask
        mask=mask.unsqueeze(2)
        energy.masked_fill_(mask, -1e10)
        attention = energy.softmax(1)#.softmax(-1) # last dim or dim=1
        context = torch.bmm(attention.permute(0,2,1), value)
        return context.squeeze(1), attention.squeeze(2)

'''
For Pyramid LSTM:
ASSUME `data` is a packed sequence
'''
def scale_down_prep(data, lens=None):
    if type(data).__name__ == "PackedSequence":
        data, lens = pad_packed_sequence(data)
        data = data.permute(1,2,0)
    else:
        data = data.permute(0,2,1)
    if data.shape[2]%2==1:
        data = pad(data, (0,1,0,0,0,0))
    
    # because of padding, time should be divisible by 2
    # the lens should be able to be rounded which for the odd case would round up because the last one was concatenated with padding
    return data, torch.round(torch.true_divide(lens,2)) # NOTE: TA downsizing a short sequence can cause lens to become short which may cause attention softmax to give NaNs this should be 1 at the minimum though...

# Conv Layers
class pBLSTM_conv(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM_conv, self).__init__()
        # self.conv1d
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2) 
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x, lens):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        x, lens = scale_down_prep(x, lens)
        x = self.downsample(x) # B, Ch, T
        # enforce_sorted=True right now - originally False, batch_first=True - originally False
        x = pack_padded_sequence(x.permute(0,2,1), lengths=lens, batch_first=True, enforce_sorted=True)

        outputs, (hidden, context) = self.blstm(x)
        return outputs, hidden, context, lens

# Concatenate
class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x, lens, isTrain=True):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        if type(x).__name__ == "PackedSequence":
            x, lens = pad_packed_sequence(x)
            x = x.permute(1,0,2) # (T, B, H) => (B, T, H)

        if x.shape[1]%2==1:
            x = pad(x, (0,0,0,1,0,0)) # (B, T+1, H)

        x = x.reshape((x.shape[0], x.shape[1]//2, x.shape[2]*2)) # (B, T/2, H*2)
        lens = torch.ceil(torch.true_divide(lens,2)) # round -> ceil

        # enforce_sorted=True right now - originally False, batch_first=True - originally False
        if isTrain:
            x = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=True)
        else:
            x = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)

        outputs, (hidden, context) = self.blstm(x)
        return outputs, hidden, context, lens


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128, isAttended=False):
        super(Encoder, self).__init__()

        # self.input_dropout  = LockedDropout(0.15)

        # self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

        # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

        self.plstm1 = pBLSTM(hidden_dim*4, hidden_dim) # (hidden_dim*4, hidden_dim) if using LSTM
        # self.batchnorm1 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout1 = LockedDropout(0.2)
        self.plstm2 = pBLSTM(hidden_dim*4, hidden_dim)
        # self.batchnorm2 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout2 = LockedDropout(0.2)
        self.plstm3 = pBLSTM(hidden_dim*4, hidden_dim)

        self.isAttended = isAttended
        if self.isAttended:
            self.key_network = nn.Linear(hidden_dim*2, value_size)
            self.value_network = nn.Linear(hidden_dim*2, key_size)
        else:
            self.output = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens, isTrain=True, input_dropout=0):
        # rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=True)
        # outputs = self.lstm(rnn_inp)[0]

        # if input_dropout:
        #     x = x.permute(1,0,2)
        #     self.input_dropout.p = input_dropout
        #     x = self.input_dropout(x)
        #     x = x.permute(1,0,2)

        # x = x.permute(0,2,1)
        # x = self.conv1d(x)
        # x = x.permute(0,2,1)
        if isTrain:
            x = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=True)
        else:
            x = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)

        x = self.lstm(x)[0]

        # print("ENCODER START")
        outputs, hidden, context, lens = self.plstm1(x, lens, isTrain=isTrain)
        # outputs, _ = pad_packed_sequence(outputs)

        # outputs = outputs.permute(1,2,0) # (T,B,H) -> (B,H,T)
        # outputs = self.batchnorm1(outputs) # (B,H,T)
        # outputs = outputs.permute(2,0,1) # (B,H,T) -> (T,B,H)

        # self.dropout1.p = input_dropout
        # outputs = self.dropout1(outputs) # (T,B,H)
        # outputs = outputs.permute(1,0,2) # (T,B,H) -> (B,T,H)
        # print("(B,T,H)", outputs.shape)
        
        outputs, hidden, context, lens = self.plstm2(outputs, lens, isTrain=isTrain)
        # outputs, _ = pad_packed_sequence(outputs)

        # outputs = outputs.permute(1,2,0) # (T,B,H) -> (B,H,T)
        # outputs = self.batchnorm2(outputs) # (B,H,T)
        # outputs = outputs.permute(2,0,1) # (B,H,T) -> (T,B,H)

        # self.dropout2.p = input_dropout
        # outputs = self.dropout2(outputs) # (T,B,H)
        # outputs = outputs.permute(1,0,2) # (T,B,H) -> (B,T,H)
        
        outputs, hidden, context, lens = self.plstm3(outputs, lens, isTrain=isTrain)
    
        outputs, lens = pad_packed_sequence(outputs)

        outputs = outputs.permute(1,0,2)
        # print("Encoder outputs: {}, hidden: {}, context: {}".format(outputs.shape, hidden.shape, context.shape)) #-------------------
        if self.isAttended:
            keys = self.key_network(outputs)
            value = self.value_network(outputs)
            return keys, value, lens, outputs
        else:
            outputs = self.output(outputs)
            return outputs, lens

class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.hidden_dim = key_size + value_size
        self.key_size = key_size
        # print(key_size, value_size, self.hidden_dim)
        self.embedding = nn.Embedding(vocab_size, self.hidden_dim) #, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=self.hidden_dim + value_size, hidden_size=key_size)
        self.lstm2 = nn.LSTMCell(input_size=key_size, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size) # , bias=False)
        # Weight tying - acts as regularization
        self.character_prob.weight = self.embedding.weight

    def forward(self, values, key, teacher_force_rate=1., lens=None, text=None, isTrain=True, validate=False, plot_attn=False):
        '''
        :param key :(N, T, key_size) Output of the Encoder Key projection layer
        :param values: (N, T, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character prediction probability 
        '''
        # print("key", key)
        # print("values",values)
        batch_size = values.shape[0] # [1]

        if isTrain:
            embeddings = self.embedding(text)

        if self.isAttended:
            if (isTrain == True):
                max_len =  text.shape[1]
            elif validate:
                max_len = text.shape[1]
            else:
                max_len = 600
        else:
            max_len = values.shape[1]

        predictions = []
        hidden_states = [None, None]
        prediction = ((torch.ones(batch_size,)*33).long()).to(DEVICE) # TODO: TA change from torch.zeros
        # [a b c <eos>] -> <sos>->a, a->b
        if plot_attn:
            attn = []
            cntxt = []
        # print("Decoder text:", text.shape)
        # print("Decoder max_len:", max_len) #----------------------
        
        if self.isAttended:
            initial_query = torch.zeros((batch_size, self.key_size))
            # print(initial_query.shape, self.key_size)
            # initial_query = torch.ones((batch_size, self.hidden_dim)) # TODO: this may be better
            # print("initial_query", initial_query)
            # print("lens",lens)
            context, attention = self.attention(initial_query, key, values, lens)
            # context = values[:,-1,:] # TODO may try this
        # print("CONTEXT",context)
        for i in range(max_len):
            # * Implement Gumbel noise and teacher forcing techniques  # TODO TA edit word "Gumble"
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do not get index out of range errors.
            if i==0:
                char_embed = self.embedding(prediction) # first one will be 33 = <sos>
            else:
                if (isTrain):
                    # Teacher forcing
                    teacher_force = np.random.binomial(1, teacher_force_rate)
                    # if i == 0:
                    #     char_embed = self.embedding(sos)
                    # else: # if use above then embeddings[:,i-1,:]
                    if teacher_force:
                        char_embed = embeddings[:,i-1,:] # TA check student's character inputs
                    else:
                        # NOTE can random sample here
                        # TODO should I random sample here? so softmax? or softmax output?
                        char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    # if i==0:
                    #     char_embed = self.embedding(prediction) # for the tensor of 33s for first time in iteration
                    # else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))

            # print(char_embed.shape, max_len, values.shape, outputs.shape)
            if self.isAttended:
                inp = torch.cat([char_embed, context], dim=1) # values[i,:,:]
            else:
                inp = torch.cat([char_embed, values[:,i,:]], dim=1) # values[i,:,:]
            
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]

            if self.isAttended:
                context, attention = self.attention(output, key, values, lens)
                # print(attention[0:1,:])
                C = context
            else:
                C = values[:,i,:] # values[i,:,:]
            
            # print(attention_mask[0:1,:].shape)
            # assert(False)#------------------------
            
            # attention plotting
            if plot_attn:
                attn.append(attention[0:1,:])
                cntxt.append(context[0:1,:])

                # print("Kinori", attention_mask[0:1,:,0].shape)

            prediction = self.character_prob(torch.cat([output, C], dim=1))
            # prediction = gumbel_softmax(prediction)
            # prediction = prediction.softmax(2)
            predictions.append(prediction.unsqueeze(1))

        if plot_attn:
            return torch.cat(predictions, dim=1), torch.cat(attn, dim = 0), torch.cat(cntxt, dim = 0)
        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Seq2Seq, self).__init__()
        self.isAttended = isAttended

        self.encoder = Encoder(input_dim, hidden_dim, value_size=value_size, key_size=key_size, isAttended=self.isAttended)
        self.decoder = Decoder(vocab_size, value_size=value_size, key_size=key_size, isAttended=self.isAttended)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True, teacher_force_rate=1., validate=False, plot_attn=False, input_dropout=0):
        if self.isAttended:
            key, value, lens, outputs = self.encoder(speech_input, speech_len, isTrain=isTrain, input_dropout=input_dropout)
        else:
            value, lens = self.encoder(speech_input, speech_len, isTrain=isTrain, dropout=dropout)
            key = None

        if (isTrain == True):
            predictions = self.decoder(value, key, teacher_force_rate, lens, text_input, plot_attn=plot_attn)
        elif validate:
            predictions = self.decoder(value, key, lens=lens, text=text_input, isTrain=False, validate=validate, plot_attn=plot_attn)
        else:
            predictions = self.decoder(value, key, lens=lens, text=None, isTrain=False, plot_attn=plot_attn)
        return predictions

if __name__ == "__main__":
    from dataloader import collate_test, collate_train

    LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

    batch_size = 32
    
    # print("Encoder-Decoder Attended Test:")
    # x=[]
    # for i in range(batch_size):
    #     x.append(torch.rand((torch.randint(100, 200, (1,)),40))) # B,100,Ch
    # x, x_lens = collate_test(x)
    # print(x.shape, x_lens)
    # encoder = Encoder(input_dim=40, hidden_dim=128, isAttended=True) # , value_size=128,key_size=128)
    # key, value, lens, outputs = encoder.forward(x, x_lens)
    # print(key.shape, value.shape, outputs.shape)
    # print(lens)

    # decoder = Decoder(len(LETTER_LIST), hidden_dim=128, isAttended=True).to(DEVICE)
    # predictions, attn = decoder(value.to(DEVICE), key.to(DEVICE), lens=lens, isTrain=False, plot_attn=True)
    # print(predictions.shape, attn.shape)
    # plot_attn_flow(attn.detach().to('cpu'), './testimage.png')

    # print("Encoder-Decoder Not Attended Test:")
    # encoder = Encoder(input_dim=40, hidden_dim=128, isAttended=False) # , value_size=128,key_size=128)
    # outputs, lens = encoder.forward(x, x_lens)
    # print(outputs.shape)
    # print(lens)

    # decoder = Decoder(len(LETTER_LIST), hidden_dim=128, isAttended=False).to(DEVICE)
    # # must pass None for key when not attended
    # predictions = decoder(outputs.to(DEVICE), None, lens=lens, isTrain=False)
    # print(predictions.shape)

    print("Seq2Seq Test:")
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')

    batch_size = 16
    valid_dataset = Speech2TextDataset(speech_valid, text=character_text_valid)
    result = collate_train([valid_dataset[0],valid_dataset[1]])


    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    # loop through loader
    for i, (x,y, x_len, y_len) in enumerate(valid_loader):
        print("Input shapes:", x.shape, x_len.shape, "\t", y.shape, y_len.shape)
        print()
        model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128).to(DEVICE)
        out, attn, context = model.forward(x.to(DEVICE), x_len, text_input=y.to(DEVICE), plot_attn=True, input_dropout=0.2)
        plot_attn_flow(attn.detach().to('cpu'), './testimage.png')
        print()
        print(out.shape)
        # print("\t\t", i, x.shape, x_len.shape, y.shape, y_len.shape)
        if i==0:
            break
    # x=[]
    # for i in range(batch_size):
    #     l = torch.randint(100, 200, (1,))
    #     x.append((torch.rand((l,40)), torch.randint(34, (l,)))) # B,100,Ch
    # x, y, x_lens, y_lens = collate_train(x)

    

