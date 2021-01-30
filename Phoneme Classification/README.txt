Kinori Rosnow
HW1 Part 2 Submission

Environment: This is the environment in which the code was run.
Pytorch 1.6
Numpy 1.18.1 +
AWS EC2 - Deep Learning AMI Ubuntu 18.04 - g4dn.xlarge

Description:
My model architecture just uses Linear, ReLU and Batchnorm1D layers in a sequence. I started by tuning the
context and had the model width be based on the context. Once I found a context that seemed to be the best
(context=40), I began adding depth and width and various combinations. The width was almost always some 
product of the context and some constant. I also began playing with batch sizes and found that my the batch
sizes of 256-512 were training faster and giving me good results. I didn't try much larger batch sizes. I 
stayed with the same starting learning rate of 1e-3, but tuned the scheduler by having it reduce the learning
rate by a factor of 0.1-0.5 every 4-10 epochs. I found that my best performance happened at 4 epochs, 0.1
learning rate. I wanted to try fewer epochs, but I was concerned about time. I alos had an interesting 
observation that my models were at their best around 4-5 epochs, just after the scheduler reduced the learning
rate.
My data-loading scheme had the same basic structure as the recitation code. The MyDataset class was leveraged
by the DataLoader which was used to iterate through for training. I tried 3 different ways of loading the
data with the most preprocessing up front to make __getitem__() as fast as possible. The first was a front
loaded data prep where I created the context/padding and saved a list of prepped inputs. This didn't work
because it took too much RAM despite being fast. I then tried storing a direct mapping of index to location
in the data. This also took too much memory. I finally settled on a map that tracked total number of data
points that had been passed by the end of each sample in the dataset. This allowed me to binary search to
find which sample the index was in, then take the difference between the previous sample's last index and
the desired index to get the "in sample index". This made __getitem__() O(ln(S)) where S is the number of
utterance samples (not the number of data points). This used much less RAM and was still pretty fast.
Originally before I did all of that I thought I might be able to have __getitem()__ iterate through the
data, but then I realized this may mess with the DataLoader if shuffle was on becuase the I would always
be giving data in the same order no matter the index (at least how I had written it). This idea was out
quickly.

To Train:
To run the training script you just call the hw1p2_train.py. The model will be trained and every epoch it
will print the accuracy of the validation set, save the model to ./hw1p2_models/model_n_m/model.pt where n 
is the model number denoted in the script and m is the epoch number. Hyperparameters are also stored in the
same directory as the model in a hyperparameters.txt file.
>>> python3 hw1p2_train.py

To Predict:
To create predictions to submit to kaggle, call hw1p2_predict.py. A model is loaded from the model path:
./hw1p2_models/model_n_m/ . The model us used to generate predictions which are saved in a submission csv 
will be created and put into the current directory as submissionN.csv where N is the submission number
as specified in the script.
>>> python3 hw1p2_predict.py

NOTE: By default CUDA is on. If you want to run on CPU, open the file and go to the main function below
the class definitions. The first line will be the `cuda` boolean value you can set to `cuda=False`. I did
not use CPU after initial testing because the model is big.