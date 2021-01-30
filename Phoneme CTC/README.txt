Kinori Rosnow
HW3 Part 2 Submission

Environment: This is the environment in which the code was run.
Pytorch 1.7
Numpy 1.18.1 +
AWS EC2 - Deep Learning AMI Ubuntu 18.04 - g4dn.xlarge

Description:
Used a downloaded package for levenshtein distance (https://github.com/ztane/python-Levenshtein/). 
Download using commands below:
While in activated pytorch environment...
>>> pip install python-Levenshtein

I tried a few combinations of bidirectional LSTM, Conv1d, Linear and Dropout layers. For some reason my best
performing model was one of the earliest I tried that was based on the baseline having 1 Conv1d layer, 3
LSTM layers followed by a single linear layer with an output through a softmax. I had trouble getting
my models to perform better than 10.2 average Levenshtein distance. I ran out of time as the training
required many epochs.
I made an interesting observation which was that a larger batch size did not speed up the training process
necessarily. I attributed this to the sequences getting padded to the longest sequence length so if my batch
size was large, then not only did I have more "rows" of data, but the rows were extended to the longest 
sequence and longer sequences were more likely with a larger sampling of the training data. I found a happy
medium was around batch size 64.
I tuned the training scheduler and learning rate, but only towards the end did I find a decent schedule. I
started with a learning rate of 1e-3 and halved it every 5-6 epochs for a total of 21-22 epochs.

To train and test the model the training scripts must be run in the order named here.
1) Train classification:
The script will save the model to './hw1p2_models/model_n_m/model.pt' where n is the model number denoted 
in the script and m is the epoch number. Hyperparameters are also stored in the same directory as the model
in a hyperparameters.txt file. The best model (if it trains like mine did) should be saved in directory 
'./hw3p2_models/model_2_21'.
The command to run:
>>> python3 hw3p2_train.py

2) Predict:
The model saved by the training script is loaded from the model directory: './hw3p2_models/model_2_21/'. The 
model is used to generate predictions which are saved in a submission csv will be created and put into the 
current directory as 'submission.csv' as specified in the script. If you want to change the lodaded model
just change the model path at the top of the main function. The command to run:
>>> python3 hw3p2_predict.py

NOTE: By default CUDA is on. If you want to run on CPU, open the file and go to the main function below
the class definitions. The first line will be the `cuda` boolean value you can set to `cuda=False`. I did
not use CPU after initial testing because the model is big.