Kinori Rosnow
HW2 Part 2 Submission

Environment: This is the environment in which the code was run.
Pytorch 1.6
Numpy 1.18.1 +
AWS EC2 - Deep Learning AMI Ubuntu 18.04 - g4dn.xlarge

Description:
I experimented with 3 architectures. I started to get results first was with the baseline on the recitation.
This only got me to 0.9 AUC. I then tried MobileNet to see if depth wise convolutions would help. I added
skip connections to both models to see if it would help; however, it only harmed the performance. I would
think it was my implementation of skip connections, but my final architecture based on ResNet18 performed
the best giving me an AUC > 0.92 so it was probably that skip connections were not helpful in those models
or where I put them was not helpful. I tried tuning all models with architectural changes. 
My best model (ResNet variant) was different from the original ResNet in many ways, but I made some distinct
architectural choices. For the first Conv2d my kernel size was 3. The architecture then downsamples by 
having stride 2 for 1 MaxPool2d and then 3 more stride 2 Basic Blocks. Since our image size is 64x64 rather
than 224x224 which is what ResNet was designed for I took one of the downsamples out (the 3rd one). I also
felt that a MaxPool2d may be better for the 2nd down sample to for a little more shift invariance instead of
just a stride 2 Conv2d. I added 2 more BasicBlocks at the bottom to increase dimensionality for a more
expressive embedding. Finally I adjusted the kernel size of the average pool at the bottom to 8 to account
for the new size.
I started with training for classification with Cross Entropy Loss, SGD, batch size=256 learning rate= 1e-1, 
scheduled to reduce by a factor of 0.2 every 4 iterations (best model only did it once... maybe twich if I 
added an extra scheduled drop). Overfitting seemed to happen after only 5-8 epochs. My performance was not 
improving so I took my best models with only 5-7 epochs of training to attempt metric learning. These models 
were not too overfit and still seemed to benefit from metric learning. Later models tended to do worse with 
metric learning. Although many of them were tested with a but in my loss. I tried metric learning because in 
the recitation it was mentioned that contrastive loss can spread out the embedding vectors. I thought this 
would be effective on classification models that have identified features, but were not yet overfit.
I used the contrastive loss function in the writeup but altered the metric to be cosine similarity instead 
of Euclidean distance. I then experimented with adding the Cross Entropy Loss as well. The added loss terms
seemed to harm the training so I left it with just the contrastive loss. The contrastive loss didn't
dramatically harm classfication and even improved it sometimes. The metric learning boosted me a little on
AUC to get me just over the 0.92 cutoff.
In order for the the metric learning to work I had to generate pairs that were correctly labeled as matching
or different. I created a class TrainVerificationMyDataset() that wrapped the ImageFolder for classification
training. It generated a dictionary will all of the image indices hashed by their classficiation. The 
function load_pairs() generated a list of tuples containing the indices of the image in the ImageFolder, 
class and whether or not the images match. The __getitem__() replaced the indices with the images and 
returned all the information for a O(1) function. The metric learning sampled new pairings every epoch which 
is not too slow after the dictionary of images was created.

To train and test the model the training scripts must be run in the order named here.
1) Train classification:
The model will be trained and check accuracy and verification AUC on the validation sets. If the model
performs well enough the script will save the model to './hw1p2_models/model_n_m/model.pt' where n is the 
model number denoted in the script and m is the epoch number. Hyperparameters are also stored in the same 
directory as the model in a hyperparameters.txt file. The best model (if it trains like mine did) should be 
saved in directory './hw1p2_models/model_1_4'.
The command to run:
>>> python3 hw2p2_train.py

2) Train Metric Learning:
This script will automatically load the model saved by the classification training. This script will save
models that perform well in the same fashion as the classification training. The command to run:
>>> python3 hw2p2_metrictrain.py

3) Predict:
The model saved by the metric learning is loaded from the model directory: './hw1p2_models/model_2_7/'. The 
model is used to generate predictions which are saved in a submission csv will be created and put into the 
current directory as 'submission.csv' as specified in the script. The command to run:
>>> python3 hw2p2_predict.py

NOTE: By default CUDA is on. If you want to run on CPU, open the file and go to the main function below
the class definitions. The first line will be the `cuda` boolean value you can set to `cuda=False`. I did
not use CPU after initial testing because the model is big.