Kinori Rosnow
HW4 Part 2 Submission

Environment: This is the environment in which the code was run.
Pytorch 1.7
Numpy 1.18.1 +
AWS EC2 - Deep Learning AMI Ubuntu 18.04 - g4dn.xlarge

Description:
Used LockedDropout from torchnlp. Copy pasted class into models.py because pip install was causing problems 
with Pytorch version.
Used a downloaded package for levenshtein distance (https://github.com/ztane/python-Levenshtein/). 
Download using commands below:
While in activated pytorch environment...
>>> pip install python-Levenshtein

BiLSTM followed by Pyramid LSTM that padded with extra 0 if odd length, linear, attention, embedding layer, 2 LSTMCells, linear
The model architecture is an Encoder, Attention and Decoder. All LSTMs are bidirectional. The Encoder is an
LSTM followed by 3 pyramidal LSTMs each reducing the sequence length by 2. Odd lengthed sequences are padded
with 0. The value and key are the outputs of separate linear layers. The Decoder is an embedding layer with
tied weights with the final output linear layer. There are 2 LSTM Cell layers after the embedding layer.
I also used scheduled teacher forcing which started with a probability of 100% and reducing every 5 epochs.
I also masked the attention and loss.

I had trouble getting other variations to perform better. I tried using locked dropout at the input and in
between LSTM layers with dropout probabilities 0.1-0.2 which did not seem to help very much. I aldo tried
adding other layers like Conv1d layer as a feature map of the input and Batchnorm1d which also didn't
outperform the baseline model with added hidden dimensionality. The best model had a hidden dim of 512, but
I also tried 256. The key and value size I tried was 128, 150, and 256, but 128 was the best.

I used the Adam optimizer with a learning rate of 0.001 that decreased by a factor of 0.1 every 5 epochs

To train and test the model the training scripts must be run in the order named here.
1) Train:
The script will save the model to './hw4p2_models/model_n_m/model.pt' where n is the model number denoted 
in the script and m is the epoch number. Hyperparameters are also stored in the same directory as the model
in a hyperparameters.txt file. An attention plot of a single input from the first batch of the epoch will
also be saved in this directory. The best model (if it trains like mine did) should be saved in directory 
'./hw4p2_models/model_4_10'.
The command to run:
>>> python3 main.py train 4
To load a model for continued training:
>>> python3 main.py load 4 2

2) Predict:
The model saved by the training script is loaded from the model directory: './hw4p2_models/model_4_10/'. The 
model is used to generate predictions which are saved in a submission csv will be created and put into the 
current directory as 'submission.csv' as specified in the script. If you want to change the lodaded model
just change the model path at the top of the main function. The command to run:
>>> python3 main.py test 4 10

3) Validate:
To see the performance on the validation set:
>>> python3 main.py validate 4 10
.
