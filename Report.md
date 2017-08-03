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
