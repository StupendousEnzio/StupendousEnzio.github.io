---
layout: post
title: Predicting Customer Churn in Telecom Industry
tags: Python Exploratory | Data Analysis | Randomized SearchCV | Logistic Regression | SVM | XGBoost |Light GBM
img: /img/Churn.png
---

For all the major telecommunication industries, customer attrition is an important business metric. As it is said, "it's easy to retain a customer than to get in a new one". Organizations invest heavily to understand customer attrition and in what way can we reduce it. Customer attrition analysis and customer attrition rate are carried out to understand customer attrition.

The customer attrition/churn is broadly divided into two i.e. Voluntary churn and Involuntary churn. Voluntary churn is when a customer moves to some other service provider, whereas involuntary churn occurs due to reasons such as death, relocation so on and so forth. It is usually the case that involuntary churn is not included when understanding customer churn. Voluntary churn, on the other hand is considered as it is directly linked to the services provided by the company.

Predicting that specific group of people having high chance of likeliness to churn can be done effectively using prediction models. Carrying out effective customer retention marketing campaigns on subset of customers can help retain potential defectors.    


```python
#Importing libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt #visualization
from PIL import  Image
%matplotlib inline
import pandas as pd
import seaborn as sns #visualization
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py #visualization
py.init_notebook_mode(connected=True) #visualization
import plotly.graph_objs as go #visualization
import plotly.subplots as tls #visualization
import plotly.figure_factory as ff #visualization
#Exploratory data analysis
import pandas_profiling
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>



<font size="+3">Data</font>


```python
telcom = pd.read_csv(r"C:\Users\adity\Downloads\telco-customer-churn(1)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
telcom.head()
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
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



<font size="+3">Data Overview Using Pandas Profiling</font>


```python
tel = telcom.profile_report(style={'full_width':True}, title='Exploratory analysis Customer Churn', pool_size=4)
tel.to_file(output_file="Profile1.html")
```
<img src="{{site.baseurl}}/img/Screenshot_2019-11-26 Exploratory analysis Customer Churn.png">

<font size="+2">Pandas profiling has given some key insights which we can summarize as:</font>

   - The column *TotalCharges contains spaces
   - The dataset contains two numerical, twelve categorical and 6 Boolean variable types.
   - There exists a third category for 6 columns i.e. *No and *No internet services which may be considered the same.
   - Feature "Tenure" has a normal distribution.
   - There exits a positive correlation between Senior Citizen and monthly charges and tenure and monthly charges.

<font size="+3">Preparing the Data</font>

Before we begin we have to carry out transformations to the dataset. Restructuring the data will be helpful to reduce the complexity in dataset.


```python
#Replacing spaces with null values in total charges column
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)

#Dropping null values from total charges column which contain .15% missing data
telcom = telcom[telcom["TotalCharges"].notnull()]
telcom = telcom.reset_index()[telcom.columns]

#convert to float type
telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

#replace 'No internet service' to No for the following columns
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols :
    telcom[i]  = telcom[i].replace({'No internet service' : 'No'})

#replace values
telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})
col_name = ['SeniorCitizen','Churn']
telcom[col_name] = telcom[col_name].astype(object)

#Separating churn and non churn customers
churn     = telcom[telcom["Churn"] == "Yes"]
not_churn = telcom[telcom["Churn"] == "No"]

#Separating catagorical and numerical columns
Id_col     = ['customerID']
target_col = ["Churn"]
cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

churn_rate = telcom[cat_cols + num_cols]
churn_rate.shape
```




    (7032, 19)



<font size="+3">Exploratory Data Visualization</font>

Customer attrition in Data


```python
#fuction for percentage of customer attrition
labels =telcom["Churn"].value_counts().keys().tolist()
sizes = telcom["Churn"].value_counts().values.tolist()

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
```


<img src="{{site.baseurl}}/img/output_11_0.png">


<font size="+2">Variable distribution in customer attrition</font>

Numerical column distribution


```python
telcom[num_cols].hist(bins=15, figsize=(15, 6), layout=(2, 4))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A2592B0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A2F03C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A31CD68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A354710>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A3930B8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A3BD978>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A3FC358>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000026B6A42AD30>]],
          dtype=object)




<img src="{{site.baseurl}}/img/output_14_1.png">


<font size="+2">Pretty Visualization</font>


```python
def plot_distribution_num(data_select) :
    sns.set_style("ticks")
    a = sns.FacetGrid(telcom, hue = 'Churn',aspect = 2.5)
    a.map(sns.kdeplot, data_select, shade = True, alpha = 0.8)
    a.set(xlim=(0, telcom[data_select].max()))
    a.add_legend()
    a.set_axis_labels(data_select, 'proportion')
    a.fig.suptitle(data_select)
    plt.show()

plot_distribution_num('tenure')
plot_distribution_num('MonthlyCharges')
plot_distribution_num('TotalCharges')
```


<img src="{{site.baseurl}}/img/output_16_0.png">



<img src="{{site.baseurl}}/img/output_16_1.png">



<img src="{{site.baseurl}}/img/output_16_2.png">


<font size="+2">Scatter plot for all numerical values</font>


```python
#palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'grey'
fig = plt.figure(figsize=(18,5))
alpha = 0.8

plt.subplot(131)
ax1 = sns.scatterplot(x = telcom['TotalCharges'], y = telcom['tenure'], hue = "Churn",
                    data = telcom, alpha = alpha)
plt.title('TotalCharges vs tenure')

plt.subplot(132)
ax2 = sns.scatterplot(x = telcom['TotalCharges'], y = telcom['MonthlyCharges'], hue = "Churn",
                    data = telcom, alpha = alpha)
plt.title('TotalCharges vs MonthlyCharges')

plt.subplot(133)
ax2 = sns.scatterplot(x = telcom['MonthlyCharges'], y = telcom['tenure'], hue = "Churn",
                    data = telcom, alpha = alpha)
plt.title('MonthlyCharges vs tenure')

fig.suptitle('Numeric features', fontsize = 20)
plt.show()
```


<img src="{{site.baseurl}}/img/output_18_0.png">


<font size="+2">Understanding the collinearity between Numeric features using Spearman's correlational matrix</font>


```python
df_quant = telcom.select_dtypes(exclude=[object])
df_quant.head()
corr_quant = df_quant.corr()

fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_quant, annot=True, cmap = 'plasma',
                 linewidths = .1, linecolor = 'grey', fmt=".2f")
ax.invert_yaxis()
ax.set_title("Correlation")
plt.show()
```


<img src="{{site.baseurl}}/img/output_20_0.png">


<font size="+2">Categorical Column distribution</font>


```python
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(cat_cols, ax.flatten()):
    sns.countplot(telcom[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
```


<img src="{{site.baseurl}}/img/output_22_0.png">


<font size="+2">Visualizing all categorical columns with respect to target column</font>


```python
def plot_distribution_cat(feature1,feature2, df):
    plt.figure(figsize=(18,5))
    plt.subplot(121)
    s = sns.countplot(x = feature1, hue='Churn', data = df,
                       alpha = 0.8,
                      linewidth = 0.4, edgecolor='grey')
    s.set_title(feature1)
    for p in s.patches:
        s.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+30))

    plt.subplot(122)
    s = sns.countplot(x = feature2, hue='Churn', data = df,
                       alpha = 0.8,
                      linewidth = 0.4, edgecolor='grey')
    s.set_title(feature2)
    for p in s.patches:
        s.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+30))
    plt.show()

plot_distribution_cat('SeniorCitizen', 'gender', telcom)
plot_distribution_cat('Partner', 'Dependents', telcom)
plot_distribution_cat('MultipleLines', 'InternetService', telcom)
plot_distribution_cat('OnlineSecurity', 'TechSupport', telcom)
plot_distribution_cat('DeviceProtection', 'StreamingTV', telcom)
plot_distribution_cat('StreamingMovies', 'PaperlessBilling', telcom)
plot_distribution_cat('PaymentMethod', 'Contract', telcom)
```


<img src="{{site.baseurl}}/img/output_242_0.png">



<img src="{{site.baseurl}}/img/output_24_1.png">



<img src="{{site.baseurl}}/img/output_24_2.png">



<img src="{{site.baseurl}}/img/output_24_3.png">



<img src="{{site.baseurl}}/img/output_24_4.png">



<img src="{{site.baseurl}}/img/output_24_5.png">



<img src="{{site.baseurl}}/img/output_24_6.png">


<font size="+3">Data Preprocessing</font>



```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#customer id col
Id_col     = ['customerID']
#Target columns
target_col = ["Churn"]
#categorical columns
cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
#numerical columns
num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    telcom[i] = le.fit_transform(telcom[i])

#Duplicating columns for multi value columns
telcom = pd.get_dummies(data = telcom,columns = multi_cols )

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(telcom[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#dropping original values merging scaled values for numerical columns
df_telcom_og = telcom.copy()
telcom = telcom.drop(columns = num_cols,axis = 1)
telcom = telcom.merge(scaled,left_index=True,right_index=True,how = "left")
```

  <font size="+3">Understanding Correlation</font>


```python
tel2 = telcom.profile_report(style={'full_width':True},
                      title='Exploratory analysis Customer Churn', pool_size=4)
tel2.to_file(output_file="Profile2.html")
```
<img src="{{site.baseurl}}/img/Screenshot_2019-11-26 Exploratory analysis Customer Churn(1).png">

<font size="+2">Removing collinear features</font>


```python
#Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = telcom.corr().abs()
corr_matrix.head()

# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove :' % (len(to_drop)))

telcom = telcom.drop(columns = to_drop)

to_drop
```

    There are 1 columns to remove :





    ['MultipleLines_No_phone_service']



<font size="+2">Splitting the data into train and test</font>


```python
#Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score
from yellowbrick.classifier import DiscriminationThreshold

#To split the data in train and validation set
from sklearn.model_selection import train_test_split

#Split the features and case_status data into training and testing sets
train, test = train_test_split(telcom, test_size=0.2, random_state=0)

#Seperating dependent and independent variables
cols    = [i for i in telcom.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]

#Show the result of the split
print('Training set has {} samples.'.format(train_X.shape[0]))
print('Testing set has {} samples'.format(test_X.shape[0]))
```

    Training set has 5625 samples.
    Testing set has 1407 samples


<font size="+3">Building Model Pipeline</font>

The model building process will focus on numerous metrics. We will be considering one primary metric and one satisficing metric as it is a good practice to narrow down the metrics for efficiency as what is expected when working on deliverables for decision making.

The model will be evaluated on number of metrics and later on the best suited metric will be considered for hyperparameter tuning.

<font size="+1">Confusion Matrix</font>

A confusion matrix is an N X N matrix, where N is the number of classes being predicted. For the problem in hand, we have N=2, and hence we get a 2 X 2 matrix. Here are a few definitions, you need to remember for a confusion matrix :

<img src="{{site.baseurl}}/img/Confusion.jpg">


1. Accuracy : the proportion of the total number of predictions that were correct.
2. Positive Predictive Value or Precision : the proportion of positive cases that were correctly identified.
3. Negative Predictive Value : the proportion of negative cases that were correctly identified.
4. Sensitivity or Recall : the proportion of actual positive cases which are correctly identified.
5. Specificity : the proportion of actual negative cases which are correctly identified.

<font size="+1">F1 Score</font>

The harmonic mean of precision and recall values for a classification problem. We consider harmonic mean because it pushes extreme values more.

<img src="{{site.baseurl}}/img/F1.png">

<font size="+1">Area under ROC curve</font>

The ROC curve is the plot between sensitivity and (1- specificity). (1- specificity) is also known as false positive rate and sensitivity is also known as True Positive rate. Following is the ROC curve for the case in hand.

<img src="{{site.baseurl}}/img/ROC.png">

<font size="+1">Feature Importance</font>

Feature importance is a great way to understand the features depended on forcasting. In this case i.e. predicting customer churn will provide us an understanding of which features will impact for a probable customer to churn.
* Pros:
    * fast calculation
    * easy to retrieve - one command
* Cons:
    * biased approach, as it has a tendency to inflate the importance of continuous features or high-cardinality categorical variables


```python
def churn_pred(algo, training_X, testing_X, training_Y, testing_Y, cols,
               cf):
    #fit the algorithms
    algo.fit(training_X, training_Y)
    predictions = algo.predict(testing_X)
    probability = algo.predict_proba(testing_X)

    #Coeff
    if cf == 'coefficients':
        coefficients = pd.DataFrame(algo.coef_.ravel())
    elif cf == 'features':
        coefficients = pd.DataFrame(algo.feature_importances_)

    column_df     = pd.DataFrame(cols)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                              right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)

    print (algo)
    print ("\n Classification report : \n", classification_report(testing_Y,predictions))
    print ("Accuracy   Score : ", accuracy_score(testing_Y,predictions))

    #confusion matrix
    conf_matrix = confusion_matrix(testing_Y,predictions)

    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_Y,predictions)
    print ("Area under curve : ",model_roc_auc,"\n")
    fpr,tpr,thresholds = roc_curve(testing_Y,probability[:,1])


    trace1 = go.Heatmap(z = conf_matrix ,
                        x = ["Not churn","Churn"],
                        y = ["Not churn","Churn"],
                        showscale  = False,colorscale = "Blackbody",
                        name = "matrix")

    trace2 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))
    trace3 = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))
    trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "Blackbody",
                                  line = dict(width = .6,color = "black")))

    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                           'Feature Importances'))

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,1,2)
    fig.append_trace(trace4,2,1)

    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(0,0,0,0.2)',
                         paper_bgcolor = 'rgba(0,0,0,0.2)',
                         margin = dict(b = 195))
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                        tickangle = 90))
    py.iplot(fig)

```

<font size="+2">Logistic Regression</font>

Logistic Regression is a predictive analysis algorithm of machine learning used for classification problem, hence it is given the first choice and as you can see below, it did quite well with default parameters. Logistic Regression uses a sigmoid function to categorize labels(in this case binary) using a Sigmoid function for mapping the variables between 0 and 1, cost parameter, so on and so forth. You can find introductory part for Logistic Regression <a href="https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148">here</a>
* Pros:
    * As the target variable is categorical, and the explanatory variables can take any form.
    * Linear combination of parameters β and the input vector will be incredibly easy to compute.
    * It can easily feature engineer most non-linear features into linear ones.
* Cons:
    * It can suffer with multicollinearity


```python
from sklearn.linear_model import LogisticRegression

Logit = LogisticRegression(solver="liblinear")    
churn_pred(Logit, train_X, test_X, train_Y, test_Y, cols,
          "coefficients")
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

     Classification report :
                   precision    recall  f1-score   support

               0       0.85      0.90      0.87      1038
               1       0.66      0.55      0.60       369

        accuracy                           0.81      1407
       macro avg       0.75      0.72      0.74      1407
    weighted avg       0.80      0.81      0.80      1407

    Accuracy   Score :  0.806680881307747
    Area under curve :  0.7231347024452903

<img src="{{site.baseurl}}/img/output_38_1.png">

<font size="+2">Support Vector Machine</font>

SVM is used on wide variety of problems such as text classification tasks like category assignment, sentiment analysis, spam email detection. The internal working of SVM is such that it can categorized jumbled data by mapping in higher dimensions with the help of
<a href="https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d">Kerneling</a>
* Pros:
    * Accuracy
    * Efficient on small and clean datasets
* Cons:
    * Isn’t suited to larger datasets as the training time with SVMs can be high
    * Less effective on noisier datasets with overlapping classes



```python
#Import SVM
from sklearn import svm

#Predict
SVM = svm.SVC(kernel='linear', probability=True)
churn_pred(SVM, train_X, test_X, train_Y, test_Y, cols,
          "coefficients")
```

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
        kernel='linear', max_iter=-1, probability=True, random_state=None,
        shrinking=True, tol=0.001, verbose=False)

     Classification report :
                   precision    recall  f1-score   support

               0       0.84      0.89      0.87      1038
               1       0.64      0.53      0.58       369

        accuracy                           0.80      1407
       macro avg       0.74      0.71      0.72      1407
    weighted avg       0.79      0.80      0.79      1407

    Accuracy   Score :  0.798862828713575
    Area under curve :  0.7125961433024736

<img src="{{site.baseurl}}/img/output_41_1.png">

<font size="+2">XGBoost</font>

eXtreme gradient boosting is an ensemble technique popular among Kaggle competitions and Data scientists in industry. It is based on modified <a href="https://en.wikipedia.org/wiki/Gradient_boosting">Boosting</a>, you can find complete guide <a href="https://www.kdnuggets.com/2017/10/xgboost-top-machine-learning-method-kaggle-explained.html">here</a>   
* Pros:
    * It is highly flexible and versatile.
    * It supports distributed training platforms such as AWS, Azure etc.
    * The model formalization is in a regularized manner to tackle overfitting for better performance.
* Cons:
    * It is difficult to carry out parallel computing.


```python
from xgboost import XGBClassifier

XG = XGBClassifier(silent=False,
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.5,
                      subsample = 0.9,
                      objective='binary:logistic',
                      n_estimators=500,
                      reg_alpha = 0.3,
                      max_depth=3,
                      gamma=5)

churn_pred(XG, train_X, test_X, train_Y, test_Y, cols,
          "features")
```

    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.5, gamma=5,
                  learning_rate=0.01, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=500, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0.3, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=False, subsample=0.9, verbosity=1)

     Classification report :
                   precision    recall  f1-score   support

               0       0.84      0.91      0.87      1038
               1       0.67      0.50      0.57       369

        accuracy                           0.80      1407
       macro avg       0.75      0.70      0.72      1407
    weighted avg       0.79      0.80      0.79      1407

    Accuracy   Score :  0.8031272210376688
    Area under curve :  0.7041331829503266

<img src="{{site.baseurl}}/img/output_44_1.png">

<font size="+2">Light GBM</font>

LightGBM is a gradient boosting framework that uses tree based learning algorithms.
* Pros:
    * Faster training speed and higher efficiency.
    * Lower memory usage.
    * Better accuracy.
    * Support of parallel and GPU learning.
    * Capable of handling large-scale data.
* Cons:
    * Has overhead for split as every individual worker has to retain the complete dataset to avoid communication with other workers to perform the split.


```python
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier

lgbm_clf = lgb.LGBMClassifier(n_estimators=1500, random_state = 42)
churn_pred(lgbm_clf, train_X, test_X, train_Y, test_Y, cols,
          "features")
```

    LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                   importance_type='split', learning_rate=0.1, max_depth=-1,
                   min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                   n_estimators=1500, n_jobs=-1, num_leaves=31, objective=None,
                   random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                   subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

     Classification report :
                   precision    recall  f1-score   support

               0       0.84      0.87      0.85      1038
               1       0.58      0.52      0.55       369

        accuracy                           0.78      1407
       macro avg       0.71      0.69      0.70      1407
    weighted avg       0.77      0.78      0.77      1407

    Accuracy   Score :  0.7761194029850746
    Area under curve :  0.6928152952049752

<img src="{{site.baseurl}}/img/output_47_1.png">

<font size="+3">Optimization using hyperparameter tuning</font>

We have considered three models of the above four models for Randomized SearchCV. But, what exactly is Hyperparameter tuning and why is it necessary?
1. In contrast to model parameters used while training, model hyperparameters are something set ahead of training and control implementation aspects of the model.

2. Hyperparameters can be thought of as model settings.

3. The settings need to be tuned for each problem because the best model hyperparameters for one particular dataset will not be the best for all the datasets.

There are several approaches to hyperparameter tuning:

1. Manual: select hyperparameters based on intuition/experience/guessing, train the model with the hyperparameters, and score on the validation data. Repeat process until you run out of patience or are satisfied with the results.

2. Grid Search: set up a grid of hyperparameter values and for each combination, train a model and score on the validation data. In this approach, every single combination of hyperparameters values is tried which can be very inefficient!

3. Random search: set up a grid of hyperparameter values and select random combinations to train the model and score. The number of search iterations is set based on time/resources.

<font size="+2">Optimizing Metric = Accuracy, Satisficing Metric = F-Score</font>

1. Single number evaluation metric such as accuracy in this case allows to sort all the models according to their performance on this metric, so we can quickly decide what is working best.

2. It speeds up your ability to make decisions while selecting from n classifiers to give clear preference ranking among all of them, in-turn giving a clear direction of where to progress.

3. On the other hand Satisficing metric is the one where your algorithm just has to be good enough, setting a threshold of 60% and more for F-Score is what is expected from the algorithm, anything lower needs will be discarded.

<a href="https://www.deeplearning.ai/machine-learning-yearning/">Machine Learning Yearning</a> is a good place to know more about the metric utilization. I found the book really helpful!

<font size="+2">Randomized SearchCV Logistic Regression</font>

As discussed above, randomized search is an efficient way when considered with manual or grid search. We have considered regularization penalty space and uniform distribution as the tuned parameters and implemented with 5 fold cross validation and 100 iterations.


```python
#import necessary libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from scipy.stats import uniform
from sklearn.metrics import fbeta_score

# Create logistic regression
logistic = linear_model.LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, )

# Create randomized search 5-fold cross validation and 100 iterations
clf = RandomizedSearchCV(logistic, hyperparameters,
                         random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

# Fit randomized search
best_model = clf.fit(train_X, train_Y)

# Get the estimator
best_clf = best_model.best_estimator_

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

#Make predictions
best_predictions = best_clf.predict(test_X)


# Report the before-and-afterscores
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(test_Y,
                                                                               best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(test_Y,
                                                                     best_predictions,
                                                                     beta = 0.5)))

best_acc = accuracy_score(test_Y, best_predictions)
best_fbeta = fbeta_score(test_Y, best_predictions, beta = 0.5)
```

    Best Penalty: l2
    Best C: 1.828819231947953

    Optimized Model
    ------
    Final accuracy score on the testing data: 0.8060
    Final F-score on the testing data: 0.6309


<font size="+2">Randomized SearchCV XGBoost</font>

We have selected a list of combinations for each specific parameter.

1. Min_child_weight = The minimum weight required to create a new node in the tree. We have selected 1,5,10 to check for multiple weights as 1 will create children corresponding to fewer samples, allowing for more complex trees but likely to overfit.
2. Subsample = corresponds to the fraction of observations (the rows) to subsample at each step.
3. colsample_bytree = corresponds to the fraction of features (the columns) to use.
4. max_depth = It is the maximum number of nodes allowed from the root to the farthest leaf of a tree. Deeper trees can model more complex relationships by adding more nodes, but as we go deeper, splits become less relevant and are sometimes only due to noise, causing the model to overfit.

The randomized cv will select set of random combinations to give efficient results of them all.


```python
#import necessary libraries
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour,
                                                                      tmin, round(tsec, 2)))

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)



folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb,
                                   scoring='roc_auc', n_jobs=4, cv=skf.split(train_X,train_Y),
                                   verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
gs = random_search.fit(train_X, train_Y)
best_clf = random_search.best_estimator_

best_predictions1 = best_clf.predict(test_X)
timer(start_time)

# Report the before-and-afterscores

print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(test_Y, best_predictions1)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(test_Y, best_predictions1, beta = 0.5)))

best_acc2 = accuracy_score(test_Y, best_predictions)
best_fbeta2 = fbeta_score(test_Y, best_predictions, beta = 0.5)
```

    Fitting 5 folds for each of 5 candidates, totalling 25 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:   29.3s finished



     Time taken: 0 hours 0 minutes and 33.45 seconds.

    Optimized Model
    ------
    Final accuracy score on the testing data: 0.8017
    Final F-score on the testing data: 0.6212


<font size="+2">Randomized SearchCV Light GBM</font>

1. Fit_params = We have selected a number of parameters to avoid overtraining and optimize the number of trees.
2. Param_test = The hyperparameters to be used.


```python
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

fit_params = {"early_stopping_rounds" : 50,
             "eval_metric" : 'binary',
             "eval_set" : [(test_X,test_Y)],
             'eval_names': ['valid'],
             'verbose': 0,
             'categorical_feature': 'auto'}

param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000],
              'num_leaves': sp_randint(6, 50),
              'min_child_samples': sp_randint(100, 500),
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#number of combinations
n_iter = 200

lgbm_clf = lgb.LGBMClassifier(silent=True, metric='None', n_jobs=4)
random_search = RandomizedSearchCV(
    estimator=lgbm_clf, param_distributions=param_test,
    n_iter=n_iter,
    scoring='accuracy',
    cv=5,
    refit=True,
    verbose=True)

start_time = timer(None) # timing starts from this point for "start_time" variable
gs = random_search.fit(train_X, train_Y)
best_clf = gs.best_estimator_

predictions = gs.predict(test_X)
best_predictions = best_clf.predict(test_X)
timer(start_time)

print("Accuracy score on testing data: {:.4f}".format(accuracy_score(test_Y, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(test_Y, predictions, beta = 0.5)))

print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(test_Y, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(test_Y, best_predictions, beta = 0.5)))

best_acc3 = accuracy_score(test_Y, best_predictions)
best_fbeta3 = fbeta_score(test_Y, best_predictions, beta = 0.5)
```

    Fitting 5 folds for each of 200 candidates, totalling 1000 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:  4.4min finished



     Time taken: 0 hours 4 minutes and 26.42 seconds.
    Accuracy score on testing data: 0.8102
    F-score on testing data: 0.6409

    Optimized Model
    ------
    Final accuracy score on the testing data: 0.8102
    Final F-score on the testing data: 0.6409


<font size="+3">Model Performance</font>

Best model evaluation of all the models


```python
#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    predictions  = model.predict(testing_x)
    accuracy     = accuracy_score(testing_y,predictions)
    f1score      = f1_score(testing_y,predictions)

    df = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "f1_score"        : [f1score]
                      })
    return df

#outputs for every model
model1 = model_report(Logit,train_X,test_X,train_Y,test_Y,
                      "Logistic Regression Classifier")
model2 = model_report(SVM, train_X, test_X, train_Y, test_Y,
                      "SVM Classifier Linear")
model3 = model_report(XG, train_X, test_X, train_Y, test_Y,
                      "XGBoost Classifier")
model4 = model_report(lgbm_clf, train_X, test_X, train_Y, test_Y,
                      "Light GBM Classifier")

#concat all models
model_performances = pd.concat([model1,model2,model3,
                                model4],axis = 0).reset_index()

model_performances = model_performances.drop(columns = "index",axis =1)

table  = ff.create_table(np.round(model_performances,4))

py.iplot(table)
```
<img src="{{site.baseurl}}/img/output_58_0.png">

```python
def model_report1(model, best_prediction, best_fscore):

    df = pd.DataFrame({"Model"           : [model],
                       "Accuracy_score"  : [best_prediction],
                       "f1_score"        : [best_fscore],
                      })
    return df

model1 = model_report1("Tuned Logistic Regression Classifier", best_acc, best_fbeta)
model2 = model_report1("Tuned XGBoost Classifier", best_acc2, best_fbeta2)
model3 = model_report1("Tuned Light GBM Classifier", best_acc3, best_fbeta3)

#concat all models
model_performances = pd.concat([model1,model2,model3],axis = 0).reset_index()

model_performances = model_performances.drop(columns = "index",axis =1)

table  = ff.create_table(np.round(model_performances,4))

py.iplot(table)
```
<img src="{{site.baseurl}}/img/output_59_0.png">

<font size="+3">Final Model Evaluation</font>

The accuracy of all the models is floating in and around 80, of which Tuned Light GBM classifier had the maximum accuracy of 81.02 from 79.74 and f1-score from 57.53 to 64.09%. Further, there are numerous ways to model along with further modifications by feature merge with feature selection and training on reduced data. Accuracy and F1-score will depend on the importance to the decision and can be replaced by other metric deemed fit for the stakeholders.
