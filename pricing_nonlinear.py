# Author: Jenny_RuochengGu
# CreateTime: 2024/4/26
#  Filename: pricing_nonlinear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data = pd.read_csv('/Users/ruocheng.gu/Desktop/corporate_bond_return.20240424212205443.csv')
data.drop('Year-month',axis=1,inplace=True)
data.drop('Year',axis=1,inplace=True)
data.drop('Month',axis=1,inplace=True)
data.drop('Trdmnt',axis=1,inplace=True)
data.drop('Bondcode',axis=1,inplace=True)
data.drop('code',axis=1,inplace=True)
data.drop('Date',axis=1,inplace=True)
print(data['return'].shape)
print(data[1:].shape)
print(data.columns)
train, test = train_test_split(data, test_size=0.2, random_state=42)
train_n = train[[c for c in train.columns if train[c].dtypes!='object']].copy()
test_n = test[[c for c in test.columns if test[c].dtypes != 'object']].copy()
print(train_n.columns)

corrmat = train_n.corr()
fig,ax = plt.subplots(figsize = (12,12))
sns.heatmap(corrmat,vmax=.8,square=True, ax=ax ,annot=True,fmt='.2f',annot_kws={'size':12})
n=15
top15_cols=corrmat.nlargest(n,'return')['return'].index
corrmat_top15=train_n[top15_cols].corr()
fig1,ax1 = plt.subplots(figsize=(8,8))
sns.heatmap(corrmat_top15,vmax=.8,square=True,ax=ax1)
fig1,ax1 = plt.subplots(figsize=(8,8))
sns.heatmap(corrmat_top15,vmax=.8,square=True,ax=ax1,annot=True,fmt='.2f',annot_kws={'size':12})
plt.savefig('/Users/ruocheng.gu/Desktop/heatmap.pdf')
plt.show()
#co-linear:mve-H3;10.cashspr-A001212000;EquityNature-LargestHolerRate;LargestHolerRate-H3;LargestHolerrate-mve;EquityNature-H3;EquityNature-mve
sns.set(style='darkgrid')
flg,ax=plt.subplots(3,2,figsize=(15,15))
sns.scatterplot(x=train_n['return'],y=train_n['mve'],ax=ax[0,0],color='coral')
sns.scatterplot(x=train_n['return'],y=train_n['H3'],ax=ax[0,1],color='coral')
sns.scatterplot(x=train_n['return'],y=train_n['10.cashspr'],ax=ax[1,0],color='coral')
sns.scatterplot(x=train_n['return'],y=train_n['A001212000'],ax=ax[1,1],color='coral')
sns.scatterplot(x=train_n['return'],y=train_n['EquityNature'],ax=ax[2,0],color='coral')
sns.scatterplot(x=train_n['return'],y=train_n['LargestHolerRate'],ax=ax[2,1],color='coral')
sns.scatterplot(x=train_n['return'],y=train_n['FS_Comins.B001101000'],ax=ax[3,0],color='coral')
fig.tight_layout()
plt.savefig('/Users/ruocheng.gu/Desktop/colinear.pdf')
plt.show()

train_n1=train_n.drop(['mve','H3','10.cashspr','A001212000','EquityNature','LargestHolerRate'],axis=1)
test_n1=test_n.drop(['mve','H3','10.cashspr','A001212000','EquityNature','LargestHolerRate'],axis=1)
print(train_n1.shape)
fig, ax = plt.subplots(68, 2, figsize=(20, 60))

def graph(x, y, r, c, title):
    sns.scatterplot(x=x, y=y, color='orange', ax=ax[r, c])
    ax[r, c].set_xlabel(x.name)
    ax[r, c].set_ylabel(y.name)
    ax[r, c].set_title(title)

for r, col in enumerate(train_n1.columns):
    if r < 68:
        c = r % 2
        graph(train_n1[col], train['return'], r, c, col)


plt.savefig('/Users/ruocheng.gu/Desktop/nonlinear_vs_linear.pdf')
plt.show()
print(train_n1.columns)
train_n1 = train_n1.drop(['Betavals','ToverOsM','chmom','ILLIQ_M'],axis=1)
test_n1 = test_n1.drop(['Betavals','ToverOsM','chmom','ILLIQ_M'],axis=1)
fig = plt.figure(figsize=(15,5))
train_null=train_n1.isnull().sum()[train_n1.isnull().sum()!=0]
sns.barplot(y=train_null.index,x=train_null)
plt.show()
fig = plt.figure(figsize=(15,5))
test_null=test_n1.isnull().sum()[test_n1.isnull().sum()!=0]
sns.barplot(y=test_null.index,x=test_null)
plt.show()
for column in train_n1.columns:
    if train_n1[column].count()<2000:
        print(column)
train_n1 = train_n1.drop(['depr','rd_mve','dividendprice'],axis=1)
test_n1 = test_n1.drop(['depr','rd_mve','dividendprice'],axis=1)


for column in train_n1.columns:
    median_value = train_n1[column].median()
    train_n1[column].fillna(median_value,inplace=True)
for column in test_n1.columns:
    test_n1[column] = test_n1[column].fillna(test_n1[column].median())
# print(test_n1.shape)
# print(train_n1.columns)

Train_X_n=train_n1[1:]
Train_Y_n=train_n1['return']
Train_Y_n.drop(Train_Y_n.index[-1],inplace=True)
# print(Train_X_n.shape)
# print(Train_Y_n.shape)
print(Train_X_n.skew())
fig,ax=plt.subplots(1,2,figsize=(10,5))
sns.displot(Train_Y_n,ax=ax[0],color='green')
sns.displot(np.log1p(Train_Y_n),ax=ax[1],color='green')
plt.show()
y_train=np.log1p(Train_Y_n)

scaler=RobustScaler()
final_train_n=pd.DataFrame(scaler.fit_transform(Train_X_n),columns=Train_X_n.columns)
final_test_n=pd.DataFrame(scaler.fit_transform(test_n1[1:]),columns=Train_X_n.columns)
# print(final_train_n.head())
X_train,X_test,Y_train,Y_test = train_test_split(final_train_n,y_train,test_size=.3,random_state=0)
print(X_train.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score

rf = RandomForestRegressor()
params = {"max_depth":[15,20,25], "n_estimators":[27,30,33]}
rf_reg = GridSearchCV(rf, params, cv = 10, n_jobs =10)
rf_reg.fit(X_train,Y_train)
print(rf_reg.best_estimator_)
best_estimator=rf_reg.best_estimator_
y_pred_train = best_estimator.predict(X_train)
y_pred_test = best_estimator.predict(X_test)

print('R-square train for RF= ' + str(r2_score(Y_train, y_pred_train)))
print('R-square test for RF= ' + str(r2_score(Y_test, y_pred_test)))
