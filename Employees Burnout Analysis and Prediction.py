#!/usr/bin/env python
# coding: utf-8

# In[8]:


import warnings
warnings.filterwarnings("ignore")


# In[9]:


import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[10]:


pd.set_option('display.max_columns', None)

burnoutDf=pd.read_csv('C:\Users\91917\OneDrive\Desktop/train.csv')

burnoutDf


# In[5]:


# converting into datetime datatype
burnoutDf["Date of Joining"]= pd.to_datetime(burnoutDf["Date of Joining"])


# In[6]:


#give the number of rowsand columns
burnoutDf.shape


# In[7]:


# general information
burnoutDf.info()


# In[8]:


# show top 5 rows
burnoutDf.head()


# In[9]:


# extract all columns of the dataset
burnoutDf.columns


# In[10]:


#check the duplicate values
burnoutDf.duplicated().sum()


# In[11]:


# caluclate the mean,std,min ,max and count of every attributes
burnoutDf.describe()


# In[14]:


# show the uniqe values
for i,col in enumerate(burnoutDf.columns):
    print(f"\n\n{burnoutDf[col].unique()}")
    print(f"\n{burnoutDf[col].value_counts()}\n\n")


# In[15]:


# drop irrelevent columns
burnoutDf=burnoutDf.drop(['Employee ID'],axis=1)


# In[20]:


# check the skewness of the attributes

intFloatburnoutDf=burnoutDf.select_dtypes([np.int, np.float])
for i, col in enumerate(intFloatburnoutDf.columns):
    if (intFloatburnoutDf[col].skew() >= 0.1): 
        print("\n",col, "feature is Positively skewed and value is: ", intFloatburnoutDf[col].skew())
    elif (intFloatburnoutDf[col].skew() <= -0.1):
        print("\n",col, "feature is Negtively Skewed and value is: ", intFloatburnoutDf[col].skew())
    else:
        print("\n",col, "feature is Normally Distributed and value is: ", intFloatburnoutDf[col].skew())


# In[25]:


#Replace the null values with mean
burnoutDf[ 'Resource Allocation'].fillna(burnoutDf[ 'Resource Allocation'].mean(), inplace=True)

burnoutDf['Mental Fatigue Score'].fillna(burnoutDf[ 'Mental Fatigue Score'].mean(), inplace=True)

burnoutDf['Burn Rate'].fillna(burnoutDf[ 'Burn Rate'].mean(), inplace=True)


# In[26]:


#check for null values
burnoutDf.isna().sum()


# In[27]:


#show the correlation
burnoutDf.corr()


# In[33]:


#Plotting Heat map to check Correlation

Corr=burnoutDf.corr()
sns.set(rc={'figure.figsize':(14,12)})
fig = px.imshow(Corr, text_auto=True, aspect="auto")
fig.show()


# In[35]:


#count plot distribution of "Gender"
plt.figure(figsize=(10,8))
sns.countplot(x="Gender", data=burnoutDf, palette="magma")
plt.title("Plot Distribution of Gender")
plt.show()


# In[37]:


# Count plot distribution of "Company Type"

plt.figure(figsize=(10,8))
sns.countplot(x="Company Type", data=burnoutDf, palette="Spectral") 
plt.title("Plot Distribution of Company Type")
plt.show()


# In[40]:


#Count plot distribution of "w Setup Available"
plt.figure(figsize=(10,8))
sns.countplot(x="WFH Setup Available", data=burnoutDf, palette="dark:salmon_r")
plt.title("Plot Distribution of WFH Setup Available") 
plt.show()


# In[11]:


#count-Plot Distribution of attributes with the help of Histogram

burn_st=burnoutDf.loc[:, 'Date of Joining': 'Burn Rate']
burn_st=burn_st.select_dtypes([int, float]) 
for i, col in enumerate(burn_st.columns):
                      fig= px.histogram(burn_st, x=col, title="Plot Distribution of "+col, color_discrete_sequence=["indianred"])
                      fig.update_layout (bargap=0.2)
                      fig.show()


# In[12]:


# Plot distribution of burn  rate on the basis of Designation
fig =px.line(burnoutDf,y="Burn Rate", color="Designation", title="burn rate on the basis of Designation",color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(bargap=0.1)
fig.show()


# In[59]:


# Plot distribution of burn  rate on the basis of gender
fig =px.line(burnoutDf,y="Burn Rate", color="Gender", title="burn rate on the basis of Gender",color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(bargap=0.2)
fig.show()


# In[60]:


# Plot distribution of mental fatigue score  on the basis of Desigination
fig = px.line(burnoutDf, y="Mental Fatigue Score", color="Designation", title="Mental Fatigue Score vs  Designation",color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(bargap=0.2)
fig.show()


# In[61]:


# Plot Distribution of "Designation vs mental fatigue" as per Company type, Burn rate and Gender 
sns.relplot( 
    data=burnoutDf, x="Designation", y="Mental Fatigue Score", col="Company Type",
    hue="Company Type", size="Burn Rate", style="Gender", 
    palette=["g", "r"], sizes=(50, 200)
)


# In[62]:


# label encoding and assign in new variable
from sklearn import preprocessing 
Label_encode = preprocessing.LabelEncoder()


# In[80]:


#Assign in new variable
burnoutDf[ 'GenderLabel'] = Label_encode.fit_transform(burnoutDf['Gender'].values) 
burnoutDf['Company TypeLabel'] = Label_encode.fit_transform(burnoutDf['Company Type'].values) 
burnoutDf[ 'WFH_Setup_AvailableLabel'] = Label_encode.fit_transform(burnoutDf[ 'WFH Setup Available'].values)


# In[81]:


#check assigned values 
gn = burnoutDf.groupby('Gender')
gn = gn['GenderLabel']
gn.first()


# In[88]:


#check assigned values 
wsa = burnoutDf.groupby('WFH Setup Available')
wsa = wsa['WFH_Setup_AvailableLabel']
wsa.first()


# In[84]:


#check assigned values 
ct = burnoutDf.groupby('Company Type')
ct = ct['Company TypeLabel']
ct.first()


# In[85]:


#show last 10 rows
burnoutDf.tail(10)


# In[91]:


# Feature selection

Columns=['Designation', 'Resource Allocation', 'Mental Fatigue Score', 
         'GenderLabel', 'Company TypeLabel', 'WFH_Setup_AvailableLabel']
x=burnoutDf[Columns]
y=burnoutDf['Burn Rate']


# In[92]:


print(x)


# In[93]:


print(y)


# In[15]:


sns_plot = sns.pairplot(burnoutDf, height=2.5)
sns_plot.savefig("pairplot.png")


# In[99]:


#Principle component Analyssis

from sklearn.decomposition import PCA

pca = PCA(0.95)
x_pca = pca.fit_transform(x)

print("PCA shape of x is: ",x_pca.shape, "and orignal shape is: ", x.shape)
print("% of importance of selected features is:", pca.explained_variance_ratio_)
print("The number of features selected through PCA is:", pca.n_components_)


# In[116]:


# data splitting in train and test
from sklearn.model_selection import train_test_split
x_train_pca,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.25,random_state=10)


# In[117]:


#print the shape of splitted data
print(x_train_pca.shape,X_test.shape,y_train.shape,y_test.shape)


# In[118]:


from sklearn.metrics import r2_score


# In[119]:


#Random Forest regressor
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(x_train_pca, y_train)

train_pred_rf = rf_model.predict(x_train_pca)
train_r2 = r2_score(y_train, train_pred_rf)
test_pred_rf = rf_model.predict(x_test)
test_r2 = r2_score(y_test, test_pred_rf)

#Accuracy score

print("Accuracy score of tarin data: "+str(round(100*train_r2, 4))+" %") 
print("Accuracy score of test data: "+str(round(100*test_r2, 4))+" %")


# In[122]:


#AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

abr_model = AdaBoostRegressor()
abr_model.fit(x_train_pca, y_train)

train_pred_adboost = abr_model.predict(x_train_pca)
train_r2 = r2_score(y_train, train_pred_adboost)
test_pred_adaboost = abr_model.predict(x_test)
test_r2 = r2_score(y_test, test_pred_adaboost)

#Accuracy score

print("Accuracy score of tarin data: "+str(round(100*train_r2, 4))+" %") 
print("Accuracy score of test data: "+str(round(100*test_r2, 4))+" %")


# In[ ]:




