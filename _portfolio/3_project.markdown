---
layout: post
title: Hand-Written Digit Recognition using Google Colab
tags: Deep Learning | Image Recognition | HOG | LBP | SVM | ANN | KNN
img: /img/Digit.jpg
---

The maximum data collected in the past is of handwritten nature. Even today most of the data collected is in written form. The manual summery process carried out by various organizations is a tough, tedious and time-consuming process. To overcome this instance, the use of handwritten digit classification system will be implemented. The digit classification will be carried out to extract features using various feature extraction methods such as LBP, HOG or RAW data. The extraction process will be carried out to calculate the trained set accuracy and the test set accuracy using various parameters that will be discussed in depth in following sections and subsections.

A combination of KNN using LBP and HOG, SVM using LBP and HOG and ANN using Raw data and HOG is used in this report. The HOG feature extraction will be carried out with the calculation of magnitude and direction of the pixel intensity with the help of various kernel parameters and cell size. The will be followed by normalization to ease out the effect of overlapping. KNN and SVM classifiers will be used to accommodate HOG as it is best suited for both. For LBP the pixel is calculated by using a threshold value which is then compared to its adjacent values to calculate the pixel value in binary. The reason LBP was selected for this project is because of its invariance with respect to alleviating level change. LBP will be used as a feature extraction technique as it has low computation overhead and works well with intensity parameters. The raw data will be used directly in the classifier to train the model.

The LBP, HOG and RAW data combinations will help us understand the above described classifiers with regards to feature extraction by calculating the accuracy.

<font size="+3">SVM</font>

With LBP: As discussed in the experimental settings of KNN using LBP, the same procedure is carried out to build the LBP except the classifier. Here we are using SVM as the classifier. The model building is carried out as the SVM.svc function is initialized with ‘rbf’ as the parameter for the kernel. The kernel function is used for similarity check. The C parameter is used to calculate the soft margin cost function to minimize overfitting problem. The random_state is set to 42 for shuffling the data to calculate probability. This will calculate the training data. The accuracy function described above will calculate the test data accuracy and the confusion metrics.  

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
from sklearn import metrics
%matplotlib inline
import cv2
import seaborn as sns
```


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).



```python
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
```


```python
cd /content/gdrive/My Drive/42028-DL-CNN/Final
```

    /content/gdrive/My Drive/42028-DL-CNN/Final



```python
ls
```

    [0m[01;34mcache[0m/                     train-images-idx3-ubyte.gz
    t10k-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz
    t10k-labels-idx1-ubyte.gz



```python
X_train, y_train = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='train')
X_test, y_test = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='t10k')

labelNames = ['zero','one','two','three','four','five','six','seven','eight','nine']
```


```python
print(np.shape(X_train))
print(np.shape(X_test))
```

    (60000, 784)
    (10000, 784)



```python
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)


print("Train dataset after reshaping:{}".format(np.shape(X_train)))
print("Test dataset after reshaping :{}".format(np.shape(X_test)))
```

    Train dataset after reshaping:(60000, 28, 28)
    Test dataset after reshaping :(10000, 28, 28)



```python
img_index = 10
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.axis('off')
ax1.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,2)
ax2.axis('off')
img_index = 1000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,3)
ax2.axis('off')
img_index = 20000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,4)
ax2.axis('off')
img_index = 30000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])
```

    three
    zero
    five
    three



<img src="{{site.baseurl}}/img/output_83_1.png">



```python
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def LBPfeatures(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
    # Form the histogram
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist
```


```python
# Create an object of LocalBinaryPatterns class and initial the parameters.
desc = LocalBinaryPatterns(24, 8)
data_train = []
labels_train = []

# loop over the training images
for img_index in range(len(X_train)):
	# load the image, convert it to grayscale, and extract LBP features
	image = (X_train[img_index])
	hist = desc.LBPfeatures(image)

	# extract the label from the image path, then update the
	# label and data lists
	labels_train.append(y_train[img_index])
	data_train.append(hist)
```


```python
# train a SVM clasifier on the training data
# Initialize the SVM model
model = svm.SVC(kernel='rbf',C=100.0, random_state=42) # rbf Kernel
# Start training the SVM classifier
model.fit(data_train, labels_train)

print(np.shape(data_train))
print(np.shape(labels_train))
```

    /usr/local/lib/python3.6/dist-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    (60000, 26)
    (60000,)



```python
# Check the training accuray
print("Train set Accuracy: {:.2f}".format(model.score(data_train,labels_train)))
```

    Train set Accuracy: 0.45



```python
predictions=[]
predict_label=[]
# Exract LBP features for each test sample and classify it with the trained SVM classifier
for im_index in range(len(X_test)):
  imag = X_test[im_index]
  # Extract LBP feature
  histo = desc.LBPfeatures(imag)
  # Perform classification
  prediction = model.predict(histo.reshape(1, -1))
  # Store the classfication result
  predictions.append(prediction)
  predict_label.append(y_test[im_index])
```


```python
accuracy = metrics.accuracy_score(y_test, predictions)
print("Accuracy on test dataset:",accuracy)
```

    Accuracy on test dataset: 0.4612



```python
# plot the confusion matrix
cm  = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Plot confusion matrix using seaborn library
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);
```

    [[ 880    0    9   14    4   25    0    2    3   43]
     [   0 1086    8    1    3    0    0   35    2    0]
     [  40    2  485   61   32   73   17  179  116   27]
     [  97   21  159  182   81   96   40  161  110   63]
     [  11    3  106   30  319   18   16  105   77  297]
     [ 134   11  123   88   53  142   18  181   30  112]
     [  81   18  210   90   70   67   52  148  121  101]
     [   9   77   92   21   54   40   14  643   20   58]
     [  86    3   99  116   52   32   56   51  404   75]
     [  88   19   21   51  183   32   37  111   48  419]]



<img src="{{site.baseurl}}/img/output_153_1.png">



```python
# Display the some classification result on test samples
images = []

# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(y_test)), size=(16,)):
  # classify the clothing
  histog = desc.LBPfeatures(X_test[i])
  prediction = model.predict(histog.reshape(1, -1))
  label = labelNames[prediction[0]]
  orig_label=labelNames[y_test[i]]
  image = X_test[i]
  color = (0, 255, 0)
  image = cv2.merge([image] * 3)
  image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
  cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
  images.append(image)
```


```python
np.shape(images[1])
```




    (96, 96, 3)




```python
## Display the classification results
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(images[1])
print(orig_label[:])
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(images[2])
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(images[3])
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(images[4])
```

    eight





    <matplotlib.image.AxesImage at 0x7f0ff7157748>




<img src="{{site.baseurl}}/img/output_183_2.png">


<font size="+3">KNN</font>

With LBP: The dataset is flattened using reshape function. Feature.LBP function is used to represent the LBP representation using the above variables as the parameter along with image and uniform method. The histogram is plotted as a set of arrays of numbers with +3 and +2 as the bins and range to navigate to. Further, the histogram is normalized as float datatype and calculating the sum. An object of local binary patter class is created to loop over the training images using the train parameter and the image index, which is 10. The next is to extract the label from the image path and update the label and data list using append function. The KNN model is built using KNeighborsClassifier with n_neighbors i.e. the k value to be 3. The same procedure is carried out for test set replacing the train data and image index with prediction function to calculate the accuracy in the further step. The accuracy on the test data is calculated using a metrics.accuracy_score on the test data. A confusion metrics is used to calculate the accuracy score using accuracy model.    

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature # This pacakge is used for LBP feature extraction
from sklearn import svm # This pacakge is used for svm classification
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
%matplotlib inline
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).



```python
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
```


```python
cd /content/gdrive/My Drive/42028-DL-CNN/Final
```

    /content/gdrive/My Drive/42028-DL-CNN/Final



```python
ls
```

    [0m[01;34mcache[0m/                     train-images-idx3-ubyte.gz
    t10k-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz
    t10k-labels-idx1-ubyte.gz



```python
X_train, y_train = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='train')
X_test, y_test = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='t10k')

labelNames = ['zero','one','two','three','four','five','six','seven','eight','nine']
```


```python
# The 28X28 images are flattened to feature vector of size 784
# There are 60,000 training examples in the training dataset
# There are 10,000 test sample in the testing dataset
print(np.shape(X_train))
print(np.shape(X_test))
```

    (60000, 784)
    (10000, 784)



```python
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)

# print the size of the result reshaped train and test data splits

print("Train dataset after reshaping:{}".format(np.shape(X_train)))
print("Test dataset after reshaping :{}".format(np.shape(X_test)))
```

    Train dataset after reshaping:(60000, 28, 28)
    Test dataset after reshaping :(10000, 28, 28)



```python
# view few images and print its corresponding label
img_index = 10
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.axis('off')
ax1.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,2)
ax2.axis('off')
img_index = 1000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,3)
ax2.axis('off')
img_index = 20000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,4)
ax2.axis('off')
img_index = 30000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])
```

    three
    zero
    five
    three



<img src="{{site.baseurl}}/img/output_82_1.png">



```python
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def LBPfeatures(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
    # Form the histogram
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist
```


```python
# Create an object of LocalBinaryPatterns class and initial the parameters.
desc = LocalBinaryPatterns(24, 8)
data_train = []
labels_train = []

# loop over the training images
for img_index in range(len(X_train)):
	# load the image, convert it to grayscale, and extract LBP features
	image = (X_train[img_index])
	hist = desc.LBPfeatures(image)

	# extract the label from the image path, then update the
	# label and data lists
	labels_train.append(y_train[img_index])
	data_train.append(hist)
```


```python
model = KNeighborsClassifier(n_neighbors=3)
model.fit(data_train, labels_train)

print(np.shape(data_train))
print(np.shape(labels_train))
```

    (60000, 26)
    (60000,)



```python
print("Train set Accuracy: {:.2f}".format(model.score(data_train,labels_train)))
```

    Train set Accuracy: 0.65



```python
predictions=[]
predict_label=[]
# Exract LBP features for each test sample and classify it with the trained SVM classifier
for im_index in range(len(X_test)):
  imag = X_test[im_index]
  # Extract LBP feature
  histo = desc.LBPfeatures(imag)
  # Perform classification
  prediction = model.predict(histo.reshape(1, -1))
  # Store the classfication result
  predictions.append(prediction)
  predict_label.append(y_test[im_index])
```


```python
accuracy = metrics.accuracy_score(y_test, predictions)
print("Accuracy on test dataset:",accuracy)
```

    Accuracy on test dataset: 0.4369



```python
# plot the confusion matrix
cm  = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Plot confusion matrix using seaborn library
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);
```

    [[ 872    0   10   23    6   41    4    1    4   19]
     [   0 1097    7    4    0    3    2   18    4    0]
     [  44   13  519  142   77   46   41   81   58   11]
     [  49   12  243  311   88   98   66   50   63   30]
     [  21    8  167  130  352   54   41   24   49  136]
     [ 136   16  173  194   63  129   44   75   30   32]
     [  61   11  267  182   80   70   99   48   80   60]
     [  22   62  158  125   74   86   40  424   11   26]
     [  45    4  174  182   77   29   72   18  338   35]
     [  67   23   63  177  229   70   90   25   37  228]]



<img src="{{site.baseurl}}/img/output_152_1.png">



```python
# Display the some classification result on test samples
images = []

# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(y_test)), size=(16,)):
  # classify the clothing
  histog = desc.LBPfeatures(X_test[i])
  prediction = model.predict(histog.reshape(1, -1))
  label = labelNames[prediction[0]]
  orig_label=labelNames[y_test[i]]
  image = X_test[i]
  color = (0, 255, 0)
  image = cv2.merge([image] * 3)
  image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
  cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
  images.append(image)
```


```python
np.shape(images[1])
```




    (96, 96, 3)




```python
## Display the classification results
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(images[1])
print(orig_label[:])
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(images[2])
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(images[3])
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(images[4])
```

    eight





    <matplotlib.image.AxesImage at 0x7fdd0d52b5f8>




<img src="{{site.baseurl}}/img/output_182_2.png">


<font size="+3">KNN</font>

With HOG: The dataset is flattened using reshape function. In feature.hog function, train data, orientation = 9 where the orientation is quantized into 9 bins which are distributed from 0 degrees to 180 degrees. Pixel per cell is set to (10*10) and the cells per block is calculated with variations and calculated the best fit of 2*2 as it is small to calculate the digits and is efficient likewise to analyse. The value of k is set to 3.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
%matplotlib inline
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).



```python
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
```


```python
cd /content/gdrive/My Drive/42028-DL-CNN/Final
```

    /content/gdrive/My Drive/42028-DL-CNN/Final



```python
X_train, y_train = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='train')
X_test, y_test = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='t10k')

labelNames = ['zero','one','two','three','four','five','six','seven','eight','nine']
```


```python
np.shape(X_train)
np.shape(X_test)
```




    (10000, 784)




```python
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)

# print the size of the result reshaped train and test data splits

print("Train dataset after reshaping:{}".format(np.shape(X_train)))
print("Test dataset after reshaping :{}".format(np.shape(X_test)))
```

    Train dataset after reshaping:(60000, 28, 28)
    Test dataset after reshaping :(10000, 28, 28)



```python
# view few images and print its corresponding label
img_index = 10
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.axis('off')
ax1.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,2)
ax2.axis('off')
img_index = 1000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,3)
ax2.axis('off')
img_index = 20000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,4)
ax2.axis('off')
img_index = 30000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])
```

    three
    zero
    five
    three



<img src="{{site.baseurl}}/img/output_71_1.png">



```python
# initialize the data matrix and labels
print("Extracting features from training dataset...")
data_train = []
labels_train = []

# loop over the training images
for img_index in range(len(X_train)):
  # load the image, and extract HOG features
  image = (X_train[img_index])
  #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

  # update the data and labels
  data_train.append(H)
  labels_train.append(y_train[img_index])

print(np.shape(data_train))
print(np.shape(labels_train))
```

    Extracting features from training dataset...
    (60000, 36)
    (60000,)



```python
model = KNeighborsClassifier(n_neighbors=3)
model.fit(data_train, labels_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=3, p=2,
               weights='uniform')




```python
print("Train set Accuracy: {:.2f}".format(model.score(data_train,labels_train)))
```

    Train set Accuracy: 0.93



```python
# initialize the data matrix and labels
print("Extracting features from test dataset...")
predict_test = []
labels_test = []
data_test=[]
# loop over the training images
for img_ind in range(len(X_test)):
  # load the image, and extract HOG features
  img=X_test[img_ind]
  H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),
                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
  pred = model.predict(H.reshape(1, -1))[0]
  # update the data and labels
  predict_test.append(pred)
  data_test.append(H)

  labels_test.append(y_test[img_ind])

print(np.shape(predict_test))
print(np.shape(labels_test))
```

    Extracting features from test dataset...
    (10000,)
    (10000,)



```python
# Test set Accuracy
accuracy = metrics.accuracy_score(y_test, predict_test)
print("Accuracy on test dataset:",accuracy)
```

    Accuracy on test dataset: 0.8676



```python
# plot the confusion matrix
cm  = metrics.confusion_matrix(y_test, predict_test)
print(cm)

# Plot confusion matrix using seaborn library
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);
```

    [[ 904    5    6    2    1    5   20    2   17   18]
     [   5 1120    2    2    1    0    4    0    1    0]
     [  18    3  879   45    7    3    3   40   16   18]
     [   4    1   55  833    0   31    0   22   50   14]
     [   3    4   13    1  840    1  103    0    6   11]
     [   6    3   16   46    2  754   17    0   33   15]
     [  21    2    5    2   43   17  855    1    5    7]
     [  22    7   47   43    1    8    1  869    8   22]
     [  53    1   25   48    3   32   24    4  729   55]
     [  19    3    4   16    4   12   10    7   41  893]]



<img src="{{site.baseurl}}/img/output_131_1.png">



```python
images = []
orig_labels=[]
# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(y_test)), size=(16,)):
  # classify the clothing
  test_img = (X_test[i])
  H1 = feature.hog(test_img, orientations=9, pixels_per_cell=(10, 10),
                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
  pred = model.predict(H1.reshape(1, -1))[0]
  #prediction = model.predict(test_img.reshape(1, -1))
  label = labelNames[pred]
  orig_labels.append(labelNames[y_test[i]])
  color = (0, 255, 0)
  test_img = cv2.merge([test_img] * 3)
  test_img = cv2.resize(test_img, (96, 96), interpolation=cv2.INTER_LINEAR)
  cv2.putText(test_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
  images.append(test_img)
```


```python
orig_labels[1]
```




    'one'




```python
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(images[1])
print(orig_labels[1])
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(images[2])
print(orig_labels[2])
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(images[3])
print(orig_labels[3])
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(images[4])
print(orig_labels[4])
```

    one
    two
    four
    one



<img src="{{site.baseurl}}/img/output_161_1.png">



```python

```


<font size="+3">SVM</font>

With HOG: The dataset is flattened using reshape function. In feature.hog function, train data, orientation = 9 where the orientation is quantized into 9 bins which are distributed from 0 degrees to 180 degrees. Pixel per cell is set to (10*10) and the cells per block is calculated with variations and calculated the best fit of 2*2 as it is small to calculate the digits and is efficient likewise to analyse. The SVM model parameters are set to RBF as the kernel which is efficient in linear model. C is set to 100 and random_state is set to 42.   

```
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature # This pacakge is used for LBP feature extraction
from sklearn import svm # This pacakge is used for svm classification
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
%matplotlib inline
import cv2
import seaborn as sns # This pacakge is used for better visualization of data (e.g confusion matrix)
```


```
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).



```
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
```


```
cd /content/gdrive/My Drive/42028-DL-CNN/Final
```

    /content/gdrive/My Drive/42028-DL-CNN/Final



```
X_train, y_train = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='train')
X_test, y_test = load_mnist('/content/gdrive/My Drive/42028-DL-CNN/Final', kind='t10k')

labelNames = ['zero','one','two','three','four','five','six','seven','eight','nine']
```


```
# The 28X28 images are flattened to feature vector of size 784
# There 60,000 training examples in the training dataset
np.shape(X_train)
np.shape(X_test)
```




    (10000, 784)




```
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)

# print the size of the result reshaped train and test data splits

print("Train dataset after reshaping:{}".format(np.shape(X_train)))
print("Test dataset after reshaping :{}".format(np.shape(X_test)))
```

    Train dataset after reshaping:(60000, 28, 28)
    Test dataset after reshaping :(10000, 28, 28)



```
# view few images and print its corresponding label
img_index = 10
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.axis('off')
ax1.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,2)
ax2.axis('off')
img_index = 1000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,3)
ax2.axis('off')
img_index = 20000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])

ax2 = fig.add_subplot(2,2,4)
ax2.axis('off')
img_index = 30000
ax2.imshow(X_train[img_index])
print(labelNames[y_train[img_index]])
```

    three
    zero
    five
    three



<img src="{{site.baseurl}}/img/output_7_1.png">



```
# initialize the data matrix and labels
print("Extracting features from training dataset...")
data_train = []
labels_train = []

# loop over the training images
for img_index in range(len(X_train)):
  # load the image, and extract HOG features
  image = (X_train[img_index])
  #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

  # update the data and labels
  data_train.append(H)
  labels_train.append(y_train[img_index])

print(np.shape(data_train))
print(np.shape(labels_train))
```

    Extracting features from training dataset...
    (60000, 36)
    (60000,)



```
img_index
```




    59999




```
model = svm.SVC(kernel='rbf',C=100.0, random_state=42) # rbf Kernel
model.fit(data_train, labels_train)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)





    SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=42,
      shrinking=True, tol=0.001, verbose=False)




```
print("Train set Accuracy: {:.2f}".format(model.score(data_train,labels_train)))
```

    Train set Accuracy: 0.88



```
# initialize the data matrix and labels
print("Extracting features from test dataset...")
predict_test = []
labels_test = []
data_test=[]
# loop over the training images
for img_ind in range(len(X_test)):
  # load the image, and extract HOG features
  img=X_test[img_ind]
  H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),
                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
  pred = model.predict(H.reshape(1, -1))[0]
  # update the data and labels
  predict_test.append(pred)
  data_test.append(H)

  labels_test.append(y_test[img_ind])

print(np.shape(predict_test))
print(np.shape(labels_test))
```

    Extracting features from test dataset...
    (10000,)
    (10000,)



```
# Test set Accuracy
accuracy = metrics.accuracy_score(y_test, predict_test)
print("Accuracy on test dataset:",accuracy)
```

    Accuracy on test dataset: 0.8859



```
# plot the confusion matrix
cm  = metrics.confusion_matrix(y_test, predict_test)
print(cm)

# Plot confusion matrix using seaborn library
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);
```

    [[ 906    5   12    0    5    6   16    1   23    6]
     [   7 1113    2    3    4    0    3    1    2    0]
     [  10    3  894   41    8    5    2   39   24    6]
     [   2    0   43  872    0   29    0   20   32   12]
     [   6    5    9    0  874    2   69    3    5    9]
     [   2    3   11   41    1  786    6    3   28   11]
     [  19    4    0    0   38    9  879    0    5    4]
     [   4    7   55   39    1    5    0  894    6   17]
     [  42    2   17   44    9   26   23    3  763   45]
     [  14    6    7   12    4   22    2   16   48  878]]



<img src="{{site.baseurl}}/img/output_141_1.png">



```
images = []
orig_labels=[]
# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(y_test)), size=(16,)):
  # classify the clothing
  test_img = (X_test[i])
  H1 = feature.hog(test_img, orientations=9, pixels_per_cell=(10, 10),
                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
  pred = model.predict(H1.reshape(1, -1))[0]
  #prediction = model.predict(test_img.reshape(1, -1))
  label = labelNames[pred]
  orig_labels.append(labelNames[y_test[i]])
  color = (0, 255, 0)
  test_img = cv2.merge([test_img] * 3)
  test_img = cv2.resize(test_img, (96, 96), interpolation=cv2.INTER_LINEAR)
  cv2.putText(test_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
  images.append(test_img)
```


```
orig_labels[1]
```




    'five'




```
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(images[1])
print(orig_labels[1])
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(images[2])
print(orig_labels[2])
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(images[3])
print(orig_labels[3])
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(images[4])
print(orig_labels[4])
```

    five
    two
    one
    zero



<img src="{{site.baseurl}}/img/output_17_1.png">


<font size="+3">ANN</font>

With HOG: Sequential model is used to build the model. Flatten function is used to flatten the image of size 28*28. Number of nodes in the hidden layer = 128. Activation function used is ReLU. Output nodes used = 10. Activation function used is Softmax. The model compilation is carried out using Adam optimizer.  The loss function used is sparse_categorical_crossentropy and evaluation metric is accuracy.     

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
%matplotlib inline
import cv2
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

```


```python
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
features = np.array(x_train, 'int16')
labels = np.array(y_train,'int')

```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step



```python
list_hog_fd = []
for pfeature in features:
 fd = feature.hog(pfeature.reshape((28,28)),pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True, block_norm="L2-Hys")
 list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd,'float64')
import tensorflow as tf
```


```python
print(np.shape(hog_features))
print(np.shape(features))

model_ann_hog_1 = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model_ann_hog_1.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```

    (60000, 36)
    (60000, 28, 28)
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.



```python
model_ann_hog_1.fit(hog_features, labels, epochs=10)

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
features_test = np.array(x_test, 'int16')
labels_test = np.array(y_test,'int')
```

    Epoch 1/10
    60000/60000 [==============================] - 3s 47us/sample - loss: 0.7897 - acc: 0.7771
    Epoch 2/10
    60000/60000 [==============================] - 3s 44us/sample - loss: 0.4964 - acc: 0.8409
    Epoch 3/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.4505 - acc: 0.8550
    Epoch 4/10
    60000/60000 [==============================] - 3s 45us/sample - loss: 0.4174 - acc: 0.8662
    Epoch 5/10
    60000/60000 [==============================] - 3s 44us/sample - loss: 0.3900 - acc: 0.8744
    Epoch 6/10
    60000/60000 [==============================] - 3s 43us/sample - loss: 0.3694 - acc: 0.8817
    Epoch 7/10
    60000/60000 [==============================] - 3s 43us/sample - loss: 0.3524 - acc: 0.8857
    Epoch 8/10
    60000/60000 [==============================] - 3s 43us/sample - loss: 0.3388 - acc: 0.8892
    Epoch 9/10
    60000/60000 [==============================] - 3s 43us/sample - loss: 0.3284 - acc: 0.8927
    Epoch 10/10
    60000/60000 [==============================] - 3s 44us/sample - loss: 0.3191 - acc: 0.8963



```python
list_hog_fd_test = []
for pfeature_test in features_test:
 fd_test = feature.hog(pfeature_test.reshape((28,28)), orientations = 9, pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True, block_norm="L2-Hys")
 list_hog_fd_test.append(fd_test)
hog_features_test = np.array(list_hog_fd_test,'float64')
model_ann_hog_1.evaluate(hog_features_test, labels_test)
```

    10000/10000 [==============================] - 0s 24us/sample - loss: 0.3099 - acc: 0.8954





    [0.30989556450843814, 0.8954]




```python
val_loss_hog , val_acc_hog = model_ann_hog_1.evaluate(hog_features_test, y_test)
print ('The accuracy of this Neural Network is : ', val_acc_hog)

```

    10000/10000 [==============================] - 0s 22us/sample - loss: 0.3099 - acc: 0.8954
    The accuracy of this Neural Network is :  0.8954



<font size="+3">ANN</font>

With Raw data: Range is 0-9. Sequential model is used for model building process. Flatten function is used to flatten the image of size 28*28. Number of nodes in hidden layer = 512. Activation function used in the hidden layer is ReLU. Dropout = 0.2 to prevent overfitting during training. Output nodes used =10. Activation function used Softmax. The model compilation is carried out using Adam optimizer. The loss function used is sparse_categorical_crossentropy and evaluation metric is accuracy.  


```python
ANN_Raw_Hog = model_ann_hog_1.predict([hog_features_test])
pred_Hog = []
for i in range(len(ANN_Raw_Hog)):
    pred_Hog.append(np.argmax(ANN_Raw_Hog[i]))
pred_Hog = np.array(pred_Hog)
```


```python
plt.figure(figsize=(9,9))
cm = metrics.confusion_matrix(y_test, pred_Hog)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'ANN_Hog_Accuracy Score : {0}'.format(val_acc_hog)
plt.title(all_sample_title, size = 15);
```


<img src="{{site.baseurl}}/img/output_8_0.png">



```python
import tensorflow as tf
from tensorflow import keras
```


```python

mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```


```python
# Converting from integers to floating point
X_train, X_test = X_train / 255.0, X_test / 255.0
```


```python

# Visualizing the first few digits using matplotlib
import matplotlib.pyplot as plt

for i in range(0, 9):
  digit = X_train[i]
  plt.subplot(330 + i + 1)
  digit = digit.reshape(28, 28)
  plt.imshow(digit)

```


<img src="{{site.baseurl}}/img/output_3_0.png">



```python
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


```


```python
model.fit(X_train, Y_train, epochs=5)
model.evaluate(X_test, Y_test)
```

    Epoch 1/5
    60000/60000 [==============================] - 16s 272us/step - loss: 0.2204 - acc: 0.9341
    Epoch 2/5
    60000/60000 [==============================] - 15s 245us/step - loss: 0.0976 - acc: 0.9700
    Epoch 3/5
    60000/60000 [==============================] - 14s 239us/step - loss: 0.0686 - acc: 0.9787
    Epoch 4/5
    60000/60000 [==============================] - 14s 239us/step - loss: 0.0532 - acc: 0.9830
    Epoch 5/5
    60000/60000 [==============================] - 14s 234us/step - loss: 0.0434 - acc: 0.9858
    10000/10000 [==============================] - 1s 55us/step





    [0.06589923885471653, 0.9811]


<font size="+3">Model Evaluation</font>


<img src="{{site.baseurl}}/img/handwritten.png">

As seen in the above table, ANN performed the best using Raw data and produced reasonably good accuracy using HOG. KNN and SVM produced reasonably good accuracy with HOG feature extraction. LBP performed poor for both KNN and SVM.

The feature extraction carried out with the help of LBP and HOG concludes that SVM and KNN both perform better when used with HOG. The magnitude and direction combination were beneficial to HOG and LBP lacked one major attribute of magnitude. Multiple combinations carried out with changes in parameters for KNN and SVM were crucial to find the maximum accuracy. ANN performed the best with RAW data. The n number of hidden layers used, and the optimizers present made the model predict maximum accuracy on the test dataset. The time and space complexity were way better than SVM or KNN with either of the extraction methods used when compared with ANN.
