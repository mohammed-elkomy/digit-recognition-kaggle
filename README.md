# Digit Recognition Kaggle 
0.99542% accuarcy on test set
## Approach 
I used a pre-trained VGG16 model and replaced the fully connected layers with untrained layers while freezing the fully convolutional layers.

I made 5 % validation set for my hyper-parameters tuning data was augmented to be x10 MNIST,I trained 5 classifiers for 5 epochs and did majority voting.

The results from 3 classifiers is exactly the same as 5 classifiers on the testset :

[digit-recognizer leaderboard](https://www.kaggle.com/c/digit-recognizer/leaderboard)
