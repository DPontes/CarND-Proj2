# **Traffic Sign Recognition**

Diogo Pontes

dpontes11@gmail.com

The steps of this project are the following:

- Load the data set
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities for new images

## **Rubric Points**

### **1. Provide a WriteUp / README that includes all the rubric points and how you addressed each one**

1.1. _The Traffic_Sign_Classifier.ipynb_ notebook file with all questions answered and all code cells executed and displaying output.
1.2. An HTML export of the project notebook with the name _Traffic_Sign_Classifier.html_.
1.3. Additional datasets or images used that are not from the German Traffi Sign Dataset
1.4. A writeup report as a markdown file with the name _Report.md_.

### **2. Dataset Summary & Exploration**

#### **2.1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually**

The Pandas library was used to calculate summary statistics of the traffic signs dataset:

- The size of the training set is 27839
- The size of the validation set is 6960 (20% of the training set)
- The size of the test set is 12630
- the shape of a traffic sign image is (32,32,3)
- The number of unique classes / labels in the dataset is 43

#### 2.2 Include an exploratory visualization of the dataset

For a random image, Image Index ID, Class ID, Sign Name, the shape of the image and the image itself are shown.

### 3. Design and Test a model Architecture

#### 3.1 Describe how you preprocessed the data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique, Preprocessing refers to techniques such as converting to grayscale, normalization, etc.

I have explored three processing methods. Normalization, grayscalling and combined. 

I decided to try normalization so that the data has mean zero and equal variance. This method usually increases the validation rate. I also tried grayscalling because [this paper] (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggested that this method had improved the accuracy of the image classification.

All of the above mentioned preprocessing methods improve the accuracy of image classification. Normalization has shown to be the best method.

#### 3.2 Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
|	RELU				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
|	Flatten				| outputs 400									|
| Fully connected		| outputs 180  									|
|	RELU				| dropout 0.6									|
| Fully connected		| outputs 90  									|
|	RELU				| dropout 0.6									|
| Fully connected		| outputs 43  									|
| Softmax				| tf.nn.softmax_cross_entropy_with_logits()     |
 
#### 3.3 Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer. Total epochs is 20 with batch size equals to 128. 

Hyperparameters includes learning rate, keep probablity, mu and sigma. Learning rate is 0.0006. Keep probablity is 0.6. Mu is 0 and sigma is 0.1.

#### 3.4 Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.984 
* test set accuracy of 0.928

The LeNet model was choosen as a starting point for traffic sign classification bacause of its simplicity and proven effectiveness in classification of images that have similarities. The convolution layer can extract features and the fully connected layers can be trained to properly make decisions.

The original architecture does not include dropout which is effective in reducing overfitting. Thus, I have added dropout to fully connected layer which turned out to be effective. I also tuned hyperparameters of feature extraction layers.

I have tried all three pre-processing methods. The best one turned out to be normalization. Normalized grayscalling is the second best method.

Also, I adjusted the epochs to reduce both underfitting and overfitting. I tunned the learning rate and found out that 0.005 to 0.001 yield very good results.

### 5. Test a Model on New Images

#### 5.1 Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


I was able to find 

These images might be difficult to classify because of low resolution，lighting conditions (e.g. low-contrast), stickers, sun glare and viewpoint variations.
