import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings("ignore")
Default = pd.read_csv('Default.csv')
print(Default.head())
print(Default.shape)
print(Default.describe())
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(y=Default['balance'])
plt.subplot(1,2,2)
sns.boxplot(x=Default['income'])
plt.show()
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(Default['student'])
plt.subplot(1,2,2)
sns.countplot(Default['default'])
plt.show()
print(Default['student'].value_counts())
print(Default['default'].value_counts())
print(Default['student'].value_counts(normalize=True))
print(Default['default'].value_counts(normalize=True))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(Default['default'],Default['balance'])
plt.subplot(1,2,2)
sns.boxplot(Default['default'],Default['income'])
plt.show()
pd.crosstab(Default['student'],Default['default'],normalize='index').round(2)
sns.heatmap(Default[['balance','income']].corr(),annot=True)
plt.show()
Default.isnull().sum()
Q1 , Q3 = Default['balance'].quantile([.25,.75])
IQR = Q3-Q1
LL = Q3-1.5*Q3
UL = Q3+1.5*IQR
print(UL)
df = Default[Default['balance']>UL]
print(df)
#print(df['default'].counts())
print(df['default'].value_counts(normalize=True))
print(df['default'].value_counts())
Default['balance']=np.where(Default['balance']>UL,UL,Default['balance'])
sns.boxplot(y=Default['balance'])
plt.show()
Default = pd.get_dummies(Default,drop_first=True)
print(Default.head())
Default.columns = ['balance','income','default','student']
print(Default.head())
from sklearn.model_selection import train_test_split
x = Default.drop('default',axis = 1)
y = Default['default']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=x,stratify=y)
print(x_train.shape)
print(x_test.shape)
print(y_train.value_counts(normalize=True).round(2))
print(" ")
print(y_test.value_counts(normalize=True).round(2))
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=33,sampling_strategy=0.75)
x_res,y_res = sm.fit_sample(x_train,y_train)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_res,y_res)
y_pred = lr.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
