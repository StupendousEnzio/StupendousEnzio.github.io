---
layout: post
title: Finding the probability of an applicant to get into US
tags: [Python, Data-preprocessing, Scikit-Learn, DecisionTree, CNN, AdaBoost]
img: /img/screenshot1.png
---


In this project, we will test several supervised learning algorithms to accurately model if an applicants will get into the United States or not. The data was collected by the US department of Labor from June 2016 to June 2017. Moving forward we will choose the best candidate algorithm from preliminary results and further optimize the algorithm to best model the data.
This type of dataset is important to not only the candidate seeking to migrate but also the department of labor to monitor and predict the applicants and the most viable attributes necessary for the applicant to further introduce new policies which can help US government take appropriate decisions.


```python
#Data Preprocessing and basic statistics
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#To make visualizations look pretty for notebooks
%matplotlib inline

#For serious exploratory data analysis
import pandas_profiling

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
```


```python
#Load the US department of labor dataset
from __future__ import division
try:
    df = pd.read_csv(r'C:\Users\adity\Downloads\Final_1.csv')
    print('The US depatment of labor dataset has {} samples with {} features each.'.format(*df.shape))
except:
    print('Dataset could not be loaded')
```

    The US depatment of labor dataset has 147230 samples with 14 features each.



```python
#A brief discription of the dataset
display(df.describe())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>case_status</th>
      <th>employer_num_employees</th>
      <th>case_solve_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>147230.000000</td>
      <td>1.472060e+05</td>
      <td>147229.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.606942</td>
      <td>2.569243e+04</td>
      <td>205.356146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.739243</td>
      <td>6.912882e+05</td>
      <td>193.493659</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>8.900000e+01</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.582000e+03</td>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2.250000e+04</td>
      <td>203.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2.635506e+08</td>
      <td>3184.000000</td>
    </tr>
  </tbody>
</table>
</div>


The dataset consists of three numerical features of the 14 features present.
To get further insight from the data we are going to use 'pandas prifile' function to get summary statistics.


```python
#Summary statistics using pandas profile, the magic one line code
df.profile_report(style={'full_width':True}, title='US Department of Labor', pool_size=4)
```
<img src="{{site.baseurl}}/img/Screenshot_2019-11-26 US Department of Labor.png" style="width:200px;height:3000px;">

The pandas_profile function has some interesting things to portray:
- The dataset has a total of 0.9% missing values followed by 12.4% Duplicate rows.
- Of the 14 features,
    [case_solve_days, employer_num_employees] are numerical(2)
    [case_status, class_of_admission, foreign_worker_info_education, job_info_education, pw_level_9089, pw_source_name_9089, pw_unit_of_pay_9089, wage_offer_unit_of_pay_9089] are categorical(8)
    [job_info_alt_combo_ed_exp, job_info_alt_field, job_info_experience, job_info_job_req_normal] are Boolean(4)
- The employer_num_employees feature is highly skewed to the right.
- There is a strong positive linear correlation between case solved days and case status(Target Column) according to pearson's and spearman's correlation matrix.
- A mid negative correlation exists between number of employees and case status(Target column).  
- The phik and Cramers V correlational plots provides loose evidence that there exist stong correlation between job education and worker education and a mild correlation between job education and class of admission. Although, phik correlation needs further validation by pairwise evaluation of all the correlations, significance and outlier significance for further feature consideration. Likewise, a Chi-Square test needs to determine significance for Cramers V to determine strength of association.  

From the initial screening, the categorical and boolean variables need to be transformed in continuous values


```python
#Replace blanks and null values in the dataframe with nan
df = df.replace([" ", "NULL"], np.nan)

#Remove null values from the dataset
df = df.dropna()
print()

#Drop duplicate rows from the dataset
df = df.drop_duplicates()

#Drop the columns having extreme values
df = df.drop(['pw_source_name_9089','pw_unit_of_pay_9089', 'wage_offer_unit_of_pay_9089'], axis=1)
```




# Preparing the data


Before the dataset can be used as input for machine learning algorithms, it has to be cleaned formatted and restructured - typically we call it as preprocessing. We have carried out the cleaning of the dataset by dropping null values, duplicates and unnecessary columns with the help of domain expert.

# Transforming Skewed continuous data

A dataset may sometimes contain at least one feature who's values tend to lie near a single number, but can have non-trivial numbers in vast proportion of a large or small single number. Algorithms tend to be sensitive to such distribution and have to be normalized for them to not underperform. With the US census data, *employer_num_employees* tend to behave in such a manner.

Lets plot a histogram to measure the skewness.


```python
#Splitting the data in features and target label
case_status = df['case_status']
features_raw = df.drop('case_status', axis=1)
```


```python
#Skewed columns
df.skew()
print('\nSkewed data is {}'.format(df.skew()))
#Visualizing skewed data
df['employer_num_employees'].plot.hist(alpha=0.5, bins=20, grid=True, legend=None)
plt.xlabel('Feature Value')
plt.title('Histogram')
plt.show()
```


    Skewed data is case_status                 1.347406
    employer_num_employees    332.616928
    case_solve_days             4.924871
    dtype: float64



<img src="{{site.baseurl}}/img/output_8_1.png">


The skewed percentage for *case_solve_days* is fairly minimum and we will leave the feature for logarithmic transformation. For highly skewed feature distribution such as *employer_num_employees*, it is common practice to apply <a href="https://www.r-statistics.com/2013/05/log-transformations-for-skewed-and-wide-distributions-from-practical-data-science-with-r/"> *logarithmic transformation*</a> so that greater and smaller values do not have an impact on the performance of the algorithm. It can further reduce the range of values caused by outliers.


```python
#Carrying out log transformations
skewed = ['employer_num_employees']
features_raw[skewed] = df[skewed].apply(lambda x: np.log(x+1))
features_raw[skewed].plot.hist(alpha=0.5, bins=20, grid=True, legend=None)
plt.xlabel('Feature Value')
plt.title('Histogram')
plt.show()
```


<img src="{{site.baseurl}}/img/output_9_0.png">


# Normalizing Numerical Features

In addition to performing transformations on features that are highly skewed, it is a good practice to perform scaling on numerical data. Applying scaling does not change the shape of the features distribution *employer_num_employees* or *case_solve_days* however, normalization ensures that there is no bias among features when assessed by supervised learners.
*Note: Once scalling is applied, you cannot observe the data in it's raw form.*


```python
#Import sklearn.preprocessing.StandardScalar
from sklearn.preprocessing import MinMaxScaler

#Initialize to the scalar then apply it to the features
scalar = MinMaxScaler()
numerical = ['case_solve_days', 'employer_num_employees']
features_raw[numerical] = scalar.fit_transform(df[numerical])

#Show an example of the record
display(features_raw.head(n=1))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_of_admission</th>
      <th>employer_num_employees</th>
      <th>foreign_worker_info_education</th>
      <th>job_info_alt_combo_ed_exp</th>
      <th>job_info_alt_field</th>
      <th>job_info_education</th>
      <th>job_info_experience</th>
      <th>job_info_job_req_normal</th>
      <th>pw_level_9089</th>
      <th>case_solve_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>H-1B</td>
      <td>8.461373e-07</td>
      <td>Master's</td>
      <td>N</td>
      <td>N</td>
      <td>Master's</td>
      <td>Y</td>
      <td>N</td>
      <td>Level II</td>
      <td>0.030705</td>
    </tr>
  </tbody>
</table>
</div>


# Data Preprocessing

From *pandas_profile* above, we can see there are several features that are non-numeric. Generally, algorithms expect input to be numeric, which calls for conversion to numbers. A popular method known as *OneHotEncoding* which creates *dummy variables* for each category of non-numerical value.


```python
#One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

#Print the number of features after one-hot encoding
encoded = list(features.columns)
print('{} total features after one hot encoding'.format(len(encoded)))
```

    74 total features after one hot encoding


# Shuffle and Split the Data

Now all categorical features have been converted to numerical and all numerical have been normalized. Now it is time to split the data into train-test set. 80% of the data will be used for training and 20% for testing.


```python
#To split the data in train and validation set
from sklearn.model_selection import train_test_split

#Split the features and case_status data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, case_status, test_size=0.2, random_state=0)

#Show the result of the split
print('Training set has {} samples.'.format(X_train.shape[0]))
print('Testing set has {} samples'.format(X_test.shape[0]))
```

    Training set has 91554 samples.
    Testing set has 22889 samples


# Evaluating Model Performace

In this section, we will investigate three different algorithms and determine the best fit for the algorithm.

# Supervised learning approach
# Model Application

# Decision Trees


- Real world application: Decision Trees and, in general, CART (Classification and Regression Trees) are often used in financial analysis. A concrete example of it is: for predicting which stocks to buy based on past peformance.
- Strengths:
    - Able to handle categorical and numerical data.
    - Doesn’t require much data pre-processing, and can handle data which hasn’t been normalized,  or encoded for Machine Learning Suitability.
    - Simple to understand and interpret.
- Weaknesses:
    - Complex Decision Trees do not generalize well to the data and can result in overfitting.
    - Unstable, as small variations in the data can result in a different decision tree. Hence they are usually used in an ensemble (like Random Forests) to build robustness.
    - Can create biased trees if some classes dominate.
- Candidacy: Since a decision tree can handle both numerical and categorical data, it’s a good candidate for our case (although, the pre-processing steps might already mitigate whatever advantage we would have had). It’s also easy to interpret, so we will know what happens under the hood to interpret the results.

# Multi-layer Neural Network

- Real world application: Neural networks are widely used in object detection and image recognition and in supply chain and logistics industry. The use of Multi-layered NN in supply chain operations reference for performance system as the metric was an important leap in logistics domain. <a href:"https://www.sciencedirect.com/science/article/pii/S0925527319300490"> Reference</a>
- Strengths:
    - MLP classifier performs best on most challanging datasets in the history of Artificial Intelligence.
    - They address one of the greatest issue present with a data scientist i.e. **Feature Engineering**
    - They can encode features useful across problem domains.
- Weakness:
    - They require huge amount of data, atleast a minimum of 10,000 examples. Other algorithms such as Decision Tree, Logistic Regression, Naive Bayes can perform well with less data.
    - They are very computationally expensive to train.
    - Working out of topology, i.e. **Grid Search** requires years of practice on how to implement this black-art and explore possibilities.
- Candidacy: Since we have a dataset of over a million rows with categorical and numerical data. Additionaly, with different **solvers** present, we can check the most suited with mix and match.

# Ensemble method: AdaBoost


- Real world application: Ensemble methods are used extensively in Kaggle competitions, usually in image detection. A real world example of Adaboost is object detection in image, ex: identifying players during a game of basketball.
- Strength:
    - Ensemble methods, including Adaboost are more robust than single estimators, have improved generalizability.
    - Simple models can be combined to build a complex model, which is computationally fast.
- Weaknesses:
    - If we have a biased underlying classifier, it will lead to a biased boosted model.
- Candidacy: Ensemble methods are considered to be high quality classifiers, and adaboost is the one of most popular boosting algorithms. We also have a class imbalance in our dataset, which boosting might be robust to.


# Creating a Training and Predicting Pipeline

To properly evaluate the performance of each model we’ve chosen, it’s important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data.


```python
#Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
from time import time

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    Inputs:
        - learner: The learning algorithm to be trained and predicted on
        - sample_size: The size of samples (number) to be drawn from training set
        - X_train: Features training set
        - y_train: case_status training set
        - X_test: Features testing set
        - y_test: case_status testing set
    '''

    results = {}

    #Fit the learner to the training data using slicing with 'sample_size'
    start = time()
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time()

    #Calculate training time
    results['train_time'] = end - start

    #Get the predictions on the test set followed by predictions on first 300 training samples
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    #Calculate the total prediction time
    results['pred_time'] = end - start

    #Compute accuracy on first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    #Compute accuracy on the test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    #Compute Fscore on first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5, average='micro')

    #Compute Fscore on testing set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5, average='micro')

    #Success
    print("{} trained on {} samples".format(learner.__class__.__name__, sample_size))

    #Return the results
    return results
```


```python
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from time import time
from sklearn.metrics import f1_score, accuracy_score

def evaluate(results):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 4, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):

                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    #ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[0, 2].axhline(y = fscore, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[1, 2].axhline(y = fscore, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')

    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
```

# Model Evaluation

Let’s train and test the models on training sets of different sizes to see how it affects their runtime and predictive performance (both on the test, and training sets).


```python
#Import supervised learning algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

#Initialize the model with random state to reproduce
clf_A = DecisionTreeClassifier(random_state=101)
clf_B = MLPClassifier(solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(15,), random_state=101)
clf_C = AdaBoostClassifier(random_state=101)

#Claculate the number of samples for 1%, 10% and 100% of the training data
samples_1 = int(round(len(X_train) / 100))
samples_10 = int(round(len(X_train) / 10))
samples_100 = len(X_train)

#Collect result on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)
```

    DecisionTreeClassifier trained on 916 samples
    DecisionTreeClassifier trained on 9155 samples
    DecisionTreeClassifier trained on 91554 samples
    MLPClassifier trained on 916 samples
    MLPClassifier trained on 9155 samples
    MLPClassifier trained on 91554 samples
    AdaBoostClassifier trained on 916 samples
    AdaBoostClassifier trained on 9155 samples
    AdaBoostClassifier trained on 91554 samples



```python
#Run metrics visualization for the three supervised learning models chosen
evaluate(results)
```


<img src="{{site.baseurl}}/img/output_24_0.png">


We can also print out the values in the visualizations above to examine the results in more detail.


```python
#Printing out the values
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))
```

    DecisionTreeClassifier



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.659007</td>
      <td>0.678929</td>
      <td>0.696492</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.659007</td>
      <td>0.678929</td>
      <td>0.696492</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.037992</td>
      <td>0.040226</td>
      <td>0.048337</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.024728</td>
      <td>0.155348</td>
      <td>2.668206</td>
    </tr>
  </tbody>
</table>
</div>


    MLPClassifier



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.503211</td>
      <td>0.688322</td>
      <td>0.693259</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.680000</td>
      <td>0.680000</td>
      <td>0.686667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.503211</td>
      <td>0.688322</td>
      <td>0.693259</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.680000</td>
      <td>0.680000</td>
      <td>0.686667</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.052022</td>
      <td>0.048294</td>
      <td>0.048879</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.635137</td>
      <td>3.700641</td>
      <td>34.817144</td>
    </tr>
  </tbody>
</table>
</div>


    AdaBoostClassifier



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.618419</td>
      <td>0.741666</td>
      <td>0.766656</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.666667</td>
      <td>0.726667</td>
      <td>0.726667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.618419</td>
      <td>0.741666</td>
      <td>0.766656</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.666667</td>
      <td>0.726667</td>
      <td>0.726667</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.965872</td>
      <td>1.176639</td>
      <td>1.195347</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.198045</td>
      <td>1.545547</td>
      <td>15.971731</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Visualize the confusion matrix for each classifier
from sklearn.metrics import confusion_matrix

for i,model in enumerate([clf_A, clf_B, clf_C]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize the data

    # view with a heatmap
    plt.figure(i)
    sns.heatmap(cm, annot=True, annot_kws={"size":10},
            cmap='Blues', square=True, fmt='.3f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix for:\n{}'.format(model.__class__.__name__));
```


<img src="{{site.baseurl}}/img/output_26_0.png">


<img src="{{site.baseurl}}/img/output_26_1.png">


<img src="{{site.baseurl}}/img/output_26_2.png">


Looking at the results above, out of the three models, AdaBoost is the most appropriate for our task.

First and foremost, it is the classifier that performs the best on the testing data, in terms of both the accuracy and f-score. It also takes resonably low time to train on the full dataset, which is half the time taken by MLP, the next best classifier to train on the full training set. So it should scale well even if we have more data.

By default, Adaboost uses a decision stump i.e. a decision tree of depth 1 as its base classifier, which can handle categorical and numerical data. Weak learners are relatively faster to train, so the dataset size is not a problem for the algorithm.

# How will AdaBoost help us?

1. Adaboost works by combining several simple learners such as decision trees, creating an enesemble of learners that will predict whether probable applicant will get into US or not.

2. Features related to **case_status** with learners which in this case are decision trees further creating set of rules that can predict the probability of the visa status.

3. The training processes lasts for numerous rounds and prioritizes the correctly predicted instances in the next round of training.

4. With each iteration, the model tries to find the best learner to incorporate into the enesemble till there are no further improvements in the predictions.

5. All the learners are then combined to make a final enesemble model, where they vote to predict the probability of visa status. Usually majority of vote is considered for final prediction.

6. Using these features, we can predict what will be the probable status of the applicant.

# Improving our Model: Model Tuning

Using grid search (GridSearchCV) with different parameter/value combinations, we can tune our model for even better results.

For Adaboost, we’ll tune the n_estimators and learning rate parameters, and also the base classifier paramters (remember our base classifier for the Adaboost ensemble is a Decision tree!).


```python
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# TODO: Create the parameters list you wish to tune
parameters = {'n_estimators':[50, 120],
              'learning_rate':[0.1, 0.5, 1.],
              'base_estimator__min_samples_split' : np.arange(2, 8, 2),
              'base_estimator__max_depth' : np.arange(1, 4, 1)
             }

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score,beta=0.5, average='micro')

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average='micro')))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='micro')))
print(best_clf)

```

    c:\users\adity\appdata\local\programs\python\python37\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)


    Unoptimized model
    ------
    Accuracy score on testing data: 0.7257
    F-score on testing data: 0.7257

    Optimized Model
    ------
    Final accuracy score on the testing data: 0.7725
    Final F-score on the testing data: 0.7725
    AdaBoostClassifier(algorithm='SAMME.R',
                       base_estimator=DecisionTreeClassifier(class_weight=None,
                                                             criterion='gini',
                                                             max_depth=3,
                                                             max_features=None,
                                                             max_leaf_nodes=None,
                                                             min_impurity_decrease=0.0,
                                                             min_impurity_split=None,
                                                             min_samples_leaf=1,
                                                             min_samples_split=2,
                                                             min_weight_fraction_leaf=0.0,
                                                             presort=False,
                                                             random_state=None,
                                                             splitter='best'),
                       learning_rate=0.1, n_estimators=50, random_state=None)


# Final model evaluation

The optimized model has an accuracy of 0.7725 and f-score of 0.7725.
It is not significant improvement, but it is important to know that with a 0.1% increase in the model has positive effects when the metric evaluation threshold is relatively tight.

# Extracting feature importance


```python
def feature_plot(importances, X_train, y_train):

    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)

    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()
```


```python
#Train the supervised model on the training set
model = AdaBoostClassifier().fit(X_train, y_train)

#Extract the feature importnace
importances = model.feature_importances_

#Plot
feature_plot(importances, X_train, y_train)
```


<img src="{{site.baseurl}}/img/output_30_0.png">


Understanding the most important features and then using the top five specified features to further increase accuracy can be of great impact for the model.   

# Feature Selection

An interesting question here is, how does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of all features present in the data. This hints that we can attempt to reduce the feature space and simplify the information required for the model to learn.

Let’s see how a model that is trained only on the selected features, performs.


```python
#Import functionality for a cloning model
from sklearn.base import clone

#Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

#Train on the best model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

#Make new prediction
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='micro')))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5, average='micro')))
```

    Final Model trained on full data
    ------
    Accuracy on testing data: 0.7725
    F-score on testing data: 0.7725

    Final Model trained on reduced data
    ------
    Accuracy on testing data: 0.7736
    F-score on testing data: 0.7736


# Effect of Feature Selection

On reduced feature data, the accuracy and f-score are still very comparable to the results on the full dataset.

The final accuracy and f-score has increased by 0.0011. Even though Adaboost is relatively faster than one of the other classifiers that we tried out, I’d still consider training on the reduced data (acc. to features) if training time was a factor, and we have more training points to process. This decision will also depend on how important accuracy and f-scores are to make a final decision.


<!--
In this project, we will test several supervised learning algorithms to accurately model if an applicants will get into the United States or not. The data was collected by the US department of Labor from June 2016 to June 2017. Moving forward we will choose the best candidate algorithm from preliminary results and further optimize the algorithm to best model the data.
This type of dataset is important to not only the candidate seeking to migrate but also the department of labor to monitor and predict the applicants and the most viable attributes necessary for the applicant to further introduce new policies which can help US government take appropriate decisions.   

```python
#Data Preprocessing and basic statistics
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#To make visualizations look pretty for notebooks
%matplotlib inline

#For serious exploratory data analysis
import pandas_profiling

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
```


```python
#Load the US department of labor dataset
from __future__ import division
try:
    df = pd.read_csv(r'C:\Users\adity\Downloads\Final_1.csv')
    print('The US depatment of labor dataset has {} samples with {} features each.'.format(*df.shape))
except:
    print('Dataset could not be loaded')
```

    The US depatment of labor dataset has 147230 samples with 14 features each.



```python
#A brief discription of the dataset
display(df.describe())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>case_status</th>
      <th>employer_num_employees</th>
      <th>case_solve_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>147230.000000</td>
      <td>1.472060e+05</td>
      <td>147229.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.606942</td>
      <td>2.569243e+04</td>
      <td>205.356146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.739243</td>
      <td>6.912882e+05</td>
      <td>193.493659</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>8.900000e+01</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.582000e+03</td>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2.250000e+04</td>
      <td>203.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2.635506e+08</td>
      <td>3184.000000</td>
    </tr>
  </tbody>
</table>
</div>


The dataset consists of three numerical features of the 14 features present.
To get further insight from the data we are going to use 'pandas profile' function to get summary statistics.


```python
#Summary statistics using pandas profile, the magic one line code
#df.profile_report(style={'full_width':True}, title='US Department of Labor', pool_size=4)
```

The pandas_profile function has some interesting things to portray:
- The dataset has a total of 0.9% missing values followed by 12.4% Duplicate rows.
- Of the 14 features,
    [case_solve_days, employer_num_employees] are numerical(2)
    [case_status, class_of_admission, foreign_worker_info_education, job_info_education, pw_level_9089, pw_source_name_9089, pw_unit_of_pay_9089, wage_offer_unit_of_pay_9089] are categorical(8)
    [job_info_alt_combo_ed_exp, job_info_alt_field, job_info_experience, job_info_job_req_normal] are Boolean(4)
- The employer_num_employees feature is highly skewed to the right.
- There is a strong positive linear correlation between case solved days and case status(Target Column) according to pearson's and spearman's correlation matrix.
- A mid negative correlation exists between number of employees and case status(Target column).  
- The phik and Cramers V correlational plots provides loose evidence that there exist stong correlation between job education and worker education and a mild correlation between job education and class of admission. Although, phik correlation needs further validation by pairwise evaluation of all the correlations, significance and outlier significance for further feature consideration. Likewise, a Chi-Square test needs to determine significance for Cramers V to determine strength of association.  

From the initial screening, the categorical and Boolean variables need to be transformed in continuous values


```python
#Replace blanks and null values in the dataframe with nan
df = df.replace([" ", "NULL"], np.nan)

#Remove null values from the dataset
df = df.dropna()
print()

#Drop duplicate rows from the dataset
df = df.drop_duplicates()

#Drop the columns having extreme values
df = df.drop(['pw_source_name_9089','pw_unit_of_pay_9089', 'wage_offer_unit_of_pay_9089'], axis=1)
```

# Preparing the data


Before the dataset can be used as input for machine learning algorithms, it has to be cleaned formatted and restructured - typically we call it as preprocessing. We have carried out the cleaning of the dataset by dropping null values, duplicates and unnecessary columns with the help of domain expert.

# Transforming Skewed continuous data

A dataset may sometimes contain at least one feature who's values tend to lie near a single number, but can have non-trivial numbers in vast proportion of a large or small single number. Algorithms tend to be sensitive to such distribution and have to be normalized for them to not underperform. With the US census data, *employer_num_employees* tend to behave in such a manner.

Lets plot a histogram to measure the skewness.


```python
#Splitting the data in features and target label
case_status = df['case_status']
features_raw = df.drop('case_status', axis=1)
```


```python
#Skewed columns
df.skew()
print('\nSkewed data is {}'.format(df.skew()))
#Visualizing skewed data
df['employer_num_employees'].plot.hist(alpha=0.5, bins=20, grid=True, legend=None)
plt.xlabel('Feature Value')
plt.title('Histogram')
plt.show()
```


    Skewed data is case_status                 1.347406
    employer_num_employees    332.616928
    case_solve_days             4.924871
    dtype: float64



<img src="{{site.baseurl}}/img/output_8_1.png">

The skewed percentage for *case_solve_days* is fairly minimum and we will leave the feature for logarithmic transformation. For highly skewed feature distribution such as *employer_num_employees*, it is common practice to apply <a href="https://www.r-statistics.com/2013/05/log-transformations-for-skewed-and-wide-distributions-from-practical-data-science-with-r/"> *logarithmic transformation*</a> so that greater and smaller values do not have an impact on the performance of the algorithm. It can further reduce the range of values caused by outliers.

```python
#Carrying out log transformations
skewed = ['employer_num_employees']
features_raw[skewed] = df[skewed].apply(lambda x: np.log(x+1))
features_raw[skewed].plot.hist(alpha=0.5, bins=20, grid=True, legend=None)
plt.xlabel('Feature Value')
plt.title('Histogram')
plt.show()
```


<img src="{{site.baseurl}}/img/output_9_0.png">


# Normalizing Numerical Features

In addition to performing transformations on features that are highly skewed, it is a good practice to perform scaling on numerical data. Applying scaling does not change the shape of the features distribution *employer_num_employees* or *case_solve_days* however, normalization ensures that there is no bias among features when assessed by supervised learners.
*Note: Once scalling is applied, you cannot observe the data in it's raw form.*

```python
#Import sklearn.preprocessing.StandardScalar
from sklearn.preprocessing import MinMaxScaler

#Initialize to the scalar then apply it to the features
scalar = MinMaxScaler()
numerical = ['case_solve_days', 'employer_num_employees']
features_raw[numerical] = scalar.fit_transform(df[numerical])

#Show an example of the record
display(features_raw.head(n=1))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_of_admission</th>
      <th>employer_num_employees</th>
      <th>foreign_worker_info_education</th>
      <th>job_info_alt_combo_ed_exp</th>
      <th>job_info_alt_field</th>
      <th>job_info_education</th>
      <th>job_info_experience</th>
      <th>job_info_job_req_normal</th>
      <th>pw_level_9089</th>
      <th>case_solve_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>H-1B</td>
      <td>8.461373e-07</td>
      <td>Master's</td>
      <td>N</td>
      <td>N</td>
      <td>Master's</td>
      <td>Y</td>
      <td>N</td>
      <td>Level II</td>
      <td>0.030705</td>
    </tr>
  </tbody>
</table>
</div>


# Data Preprocessing

From *pandas_profile* above, we can see there are several features that are non-numeric. Generally, algorithms expect inupt to be numeric, which calls for conversion to numbers. A popular method known as *OneHotEncoding* which creates *dummy variables* for each category of non-numerical value.

```python
#One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

#Print the number of features after one-hot encoding
encoded = list(features.columns)
print('{} total features after one hot encoding'.format(len(encoded)))
```

    74 total features after one hot encoding


# Shuffle and Split the Data

Now all categorical features have been converted to numericals and all numericals have been normalized. Now it is time to split the data into train-test set. 80% of the data will be used for training and 20% for testing.

```python
#To split the data in train and validation set
from sklearn.model_selection import train_test_split

#Split the features and case_status data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, case_status, test_size=0.2, random_state=0)

#Show the result of the split
print('Training set has {} samples.'.format(X_train.shape[0]))
print('Testing set has {} samples'.format(X_test.shape[0]))
```

    Training set has 91554 samples.
    Testing set has 22889 samples


# Evaluating Model Performance

In this section

Supervised learning approach

Decision Tree, Support Vector Machines(SVM), Ensemble methods: AdaBoost  

Creating a Training and Predicting Pipeline


```python
#Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
from time import time

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    Inputs:
        - learner: The learning algorithm to be trained and predicted on
        - sample_size: The size of samples (number) to be drawn from training set
        - X_train: Features training set
        - y_train: case_status training set
        - X_test: Features testing set
        - y_test: case_status testing set
    '''

    results = {}

    #Fit the learner to the training data using slicing with 'sample_size'
    start = time()
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time()

    #Calculate training time
    results['train_time'] = end - start

    #Get the predictions on the test set followed by predictions on first 300 training samples
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    #Calculate the total prediction time
    results['pred_time'] = end - start

    #Compute accuracy on first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    #Compute accuracy on the test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    #Compute Fscore on first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5, average='micro')

    #Compute Fscore on testing set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5, average='micro')

    #Success
    print("{} trained on {} samples".format(learner.__class__.__name__, sample_size))

    #Return the results
    return results
```


```python
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from time import time
from sklearn.metrics import f1_score, accuracy_score

def evaluate(results):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 4, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):

                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    #ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[0, 2].axhline(y = fscore, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    #ax[1, 2].axhline(y = fscore, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')

    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
```

Model Evaluation


```python
#Import supervised learning algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

#Initialize the model with random state to reproduce
clf_A = DecisionTreeClassifier(random_state=101)
clf_B = SVC(random_state=101)
clf_C = AdaBoostClassifier(random_state=101)

#Claculate the number of samples for 1%, 10% and 100% of the training data
samples_1 = int(round(len(X_train) / 100))
samples_10 = int(round(len(X_train) / 10))
samples_100 = len(X_train)

#Collect result on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)
```

    DecisionTreeClassifier trained on 916 samples
    DecisionTreeClassifier trained on 9155 samples
    DecisionTreeClassifier trained on 91554 samples


    c:\users\adity\appdata\local\programs\python\python37\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    SVC trained on 916 samples


    c:\users\adity\appdata\local\programs\python\python37\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    SVC trained on 9155 samples


    c:\users\adity\appdata\local\programs\python\python37\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    SVC trained on 91554 samples
    AdaBoostClassifier trained on 916 samples
    AdaBoostClassifier trained on 9155 samples
    AdaBoostClassifier trained on 91554 samples



```python
#Run metrics visualization for the three supervised learning models chosen
evaluate(results)
```


<img src="{{site.baseurl}}/img/output_24_0.png">



```python
#Printing out the values
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))
```

    DecisionTreeClassifier



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.659007</td>
      <td>0.678929</td>
      <td>0.696492</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.659007</td>
      <td>0.678929</td>
      <td>0.696492</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.028949</td>
      <td>0.015956</td>
      <td>0.019946</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.009972</td>
      <td>0.073812</td>
      <td>1.028250</td>
    </tr>
  </tbody>
</table>
</div>


    SVC



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.512910</td>
      <td>0.512910</td>
      <td>0.512517</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.456667</td>
      <td>0.456667</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.512910</td>
      <td>0.512910</td>
      <td>0.512517</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.456667</td>
      <td>0.456667</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>1.607213</td>
      <td>16.138873</td>
      <td>160.303248</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.085803</td>
      <td>8.453359</td>
      <td>1040.355758</td>
    </tr>
  </tbody>
</table>
</div>


    AdaBoostClassifier



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.618419</td>
      <td>0.741666</td>
      <td>0.766656</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.666667</td>
      <td>0.726667</td>
      <td>0.726667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.618419</td>
      <td>0.741666</td>
      <td>0.766656</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.666667</td>
      <td>0.726667</td>
      <td>0.726667</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.434796</td>
      <td>0.385969</td>
      <td>0.392952</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.068992</td>
      <td>0.514631</td>
      <td>5.504270</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Visualize the confusion matrix for each classifier
from sklearn.metrics import confusion_matrix

for i,model in enumerate([clf_A, clf_B, clf_C]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize the data

    # view with a heatmap
    plt.figure(i)
    sns.heatmap(cm, annot=True, annot_kws={"size":10},
            cmap='Blues', square=True, fmt='.3f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix for:\n{}'.format(model.__class__.__name__));
```

<img src="{{site.baseurl}}/img/output_26_0.png">


<img src="{{site.baseurl}}/img/output_26_1.png">


<img src="{{site.baseurl}}/img/output_26_2.png">



```python
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# TODO: Create the parameters list you wish to tune
parameters = {'n_estimators':[50, 120],
              'learning_rate':[0.1, 0.5, 1.],
              'base_estimator__min_samples_split' : np.arange(2, 8, 2),
              'base_estimator__max_depth' : np.arange(1, 4, 1)
             }

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score,beta=0.5, average='micro')

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average='micro')))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='micro')))
print(best_clf)

```

    c:\users\adity\appdata\local\programs\python\python37\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)


    Unoptimized model
    ------
    Accuracy score on testing data: 0.7331
    F-score on testing data: 0.7331

    Optimized Model
    ------
    Final accuracy score on the testing data: 0.7725
    Final F-score on the testing data: 0.7725
    AdaBoostClassifier(algorithm='SAMME.R',
                       base_estimator=DecisionTreeClassifier(class_weight=None,
                                                             criterion='gini',
                                                             max_depth=3,
                                                             max_features=None,
                                                             max_leaf_nodes=None,
                                                             min_impurity_decrease=0.0,
                                                             min_impurity_split=None,
                                                             min_samples_leaf=1,
                                                             min_samples_split=2,
                                                             min_weight_fraction_leaf=0.0,
                                                             presort=False,
                                                             random_state=None,
                                                             splitter='best'),
                       learning_rate=0.1, n_estimators=50, random_state=None)


Extracting feature importance


```python
def feature_plot(importances, X_train, y_train):

    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)

    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()
```


```python
#Train the supervised model on the training set
model = AdaBoostClassifier().fit(X_train, y_train)

#Extract the feature importnace
importances = model.feature_importances_

#Plot
feature_plot(importances, X_train, y_train)
```

<img src="{{site.baseurl}}/img/output_30_0.png">



```python
#Import functionality for a cloning model
from sklearn.base import clone

#Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

#Train on the best model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

#Make new prediction
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='micro')))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5, average='micro')))
```

    Final Model trained on full data
    ------
    Accuracy on testing data: 0.7725
    F-score on testing data: 0.7725

    Final Model trained on reduced data
    ------
    Accuracy on testing data: 0.7736
    F-score on testing data: 0.7736



```python

```
-->
