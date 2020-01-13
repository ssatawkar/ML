#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import resample
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[84]:


train = read_csv("C:/aa mtas/AA_ER/ml/python-code/driver-prediction/train.csv")
test = read_csv("C:/aa mtas/AA_ER/ml/python-code/driver-prediction/test.csv")


# In[85]:


train.shape , test.shape


# # Randaom Forest (mean for remaining) 0.26296 (#3880)

# # Sample data 

# In[86]:


#X_train, X_test,y_train, y_test = train_test_split(train.iloc[:,0:-1],train.iloc[:,-1],test_size = 0.2 ,random_state = 123)#


# In[87]:


#X_train.shape, X_test.shape ,y_train.shape, y_test.shape 


# In[88]:


#X_traind = pd.DataFrame( X_train)
#X_testd = pd.DataFrame(X_test)
#y_traind = pd.DataFrame(y_train)
#y_testd = pd.DataFrame(y_test)


# # Sample data set to Train 

# In[89]:


train.shape, test.shape, train.columns, test.columns


# In[90]:


train_cat= train.loc[:,train.columns.str.contains(pat='cat')]


# In[91]:


train_bin = train.loc[:,train.columns.str.contains(pat='bin')]


# In[92]:


cat_bin_col =  train.columns.str.contains(pat='cat') + train.columns.str.contains(pat='bin')


# In[93]:


test_cat = test.loc[:,test.columns.str.contains(pat ='cat')]


# In[94]:


test_bin = test.loc[:,test.columns.str.contains(pat ='bin')]


# In[95]:


train_cat.shape, test_cat.shape ,train_bin.shape , test_bin.shape


# In[96]:


train_remain = train.loc[:,~train.columns.isin(train_cat + train_bin)]
train_remain.shape


# In[97]:


test_remain = test.loc[:,~test.columns.isin(test_bin + test_cat)]
test_remain.shape


# In[98]:


train_remain.columns.intersection(test_remain.columns).size, test_remain.columns.intersection(train_remain.columns).size


# In[99]:


train_remain.columns.difference(test_remain.columns), test_remain.columns.difference(train_remain.columns) 


# In[100]:


train_cat.ps_car_01_cat.value_counts()


# In[101]:


train_cat_col_ascat = train_cat.astype('category')


# In[102]:


train_cat_col_ascat.shape


# In[103]:


test_cat_col_ascat = test_cat.astype("category")


# # Impute missing value for category

# In[104]:


from sklearn.impute import SimpleImputer


# In[105]:


imputer = SimpleImputer (missing_values=-1, strategy='most_frequent')


# In[106]:


train_cat_impute = pd.DataFrame(imputer.fit_transform(train_cat_col_ascat),columns= train_cat_col_ascat.columns)


# In[107]:


train_cat_impute.describe()


# In[108]:


train_cat_impute_dummy = pd.get_dummies(train_cat_impute)


# In[109]:


train_cat_impute_dummy.shape ,train_cat.shape


# # Missing value : category for test

# In[110]:


#imputer= SimpleImputer(missing_values = -1, strategy='most_frequent',verbose = 1)
test_cat.shape, test_cat.describe(), test_cat_col_ascat.describe()


# In[111]:


#train_cat_impute = pd.DataFrame(imputer.fit_transform(train_cat_col_ascat),columns= train_cat_col_ascat.columns)
test_cat_impute = pd.DataFrame(imputer.fit_transform(test_cat_col_ascat),  columns= test_cat_col_ascat.columns)


# In[112]:


#test_cat_impute.apply(pd.value_counts)


# In[113]:


for i in test_cat_impute.columns:
    print ("------------%s-----------" %i)
    print (test_cat_impute[i].value_counts())


# In[114]:


#train_cat_impute_dummy = pd.get_dummies(train_cat_impute
test_cat_impute_dummy = pd.get_dummies(test_cat_impute)


# In[115]:


test_cat.shape , test_cat_impute.shape , test_cat_impute_dummy.shape


# # Impute missing:  Binary

# In[116]:


#check for missing value
#train_bin.min(), test_bin.min()
test_bin.isnull().sum()


# In[117]:


# no missing value found, imputation is not required
#imputer_n = SimpleImputer(missing_values=-1, strategy='most_frequent',verbose=1)
#train_bin_impute = pd.DataFrame(imputer_n.fit_transform(train_bin),columns= train_bin.columns)


# In[118]:


train_bin_impute = train_bin
train_bin.shape


# # Impute missing:  Binary TEST 

# In[119]:


train_cat_col_ascat.apply(pd.value_counts)


# In[120]:


test_bin_impute = test_bin
test_bin_impute.shape


# # Impute missing value: Remaining

# In[121]:


train_remain.describe()


# In[122]:


#imputer_mean = SimpleImputer(missing_values=-1, strategy='mean')  # 0.26296 using mean
imputer_mean = SimpleImputer(missing_values=-1, strategy='median') 


# In[123]:


train_remain_impute = pd.DataFrame(imputer_mean.fit_transform(train_remain),columns= train_remain.columns)


# In[124]:


train_remain.shape, train_remain_impute.shape, train_remain_impute.columns


# # Impute missing value: Remaining Test

# In[125]:


test_remain_impute = pd.DataFrame(imputer_mean.fit_transform(test_remain),columns= test_remain.columns)


# In[126]:


test_remain.shape , test_remain_impute.shape, 


# # Merge Category, binary and remaining col

# In[127]:


train_merge = pd.concat ([train_bin_impute, train_cat_impute_dummy,train_remain_impute],axis=1)


# In[128]:


train_bin_impute.shape, train_cat_impute_dummy.shape,train_remain_impute.shape , train_merge.shape


# In[129]:


test_merge = pd.concat ([test_bin_impute, test_cat_impute_dummy, test_remain_impute],axis=1)


# In[130]:


test_bin_impute.shape, test_cat_impute_dummy.shape, test_remain_impute.shape, test_merge.shape


# In[131]:


#train_col = train_merge.columns , test_col = test_merge.columns


# In[132]:


#train_col.size, test_col.size


# In[133]:


#common_col = train_col.intersection(test_col)
#common_col.size


# In[134]:


#test_col.intersection(train_col).size


# In[135]:


#test_data = test_merge[common_col]
#train_data = train_merge[common_col]


# In[136]:


#df = df.loc[:,~df.columns.duplicated()]
#test_data_f = test_data.loc[:,test_data.columns.duplicated()]
#test_data.columns.duplicated()
#test_data_f.columns


# In[137]:


#data = df[df.columns.intersection(lst)]
#train_data= train_merge[train_merge.columns.intersection(test_col)]
#train_data.columns, train_data.shape, train_merge.shape


# In[138]:


train_merge_orig = train_merge
test_merge_orig= test_merge
print("train_merge= ",  train_merge.shape, "\n", "test_merge= " , test_merge.shape) 


# In[139]:


train_merge.target.value_counts()


# # Upsample the train data

# In[140]:


majority_up = train_merge[train_merge.target==0.0]
minority_up = train_merge[train_merge.target==1.0]
majority_up.shape, minority_up.shape


# In[141]:



minority_upsampled = resample(minority_up, 
                                 replace=True,     # sample with replacement
                                 n_samples=  int((majority_up.shape[0])*.80),    # to match majority class
                                 random_state=123) # reproducible results
print( minority_upsampled.shape)
print( train_merge['target'].value_counts())
print(  minority_upsampled['target'].value_counts())


# In[142]:


upsampled = pd.concat([majority_up, minority_upsampled])
upsampled.shape, majority_up.shape,minority_upsampled.shape, minority_up.shape, upsampled['target'].value_counts()
#train_merge = upsampled


# In[143]:


#train_merge = upsampled
upsampled['target'].value_counts()


# # Downsample the train data (0.24741)

# In[144]:


majority_down = train_merge[train_merge.target==0.0]
minority_down = train_merge[train_merge.target==1.0]
print (majority_down.shape, minority_down.shape  ) 


# In[145]:


majority_downsampled = resample(majority_down, 
                                 replace=False,    # sample without replacement
                                 n_samples=int(minority_down.shape[0]),     # to match minority class
                                 random_state=123) # reproducible results
majority_downsampled.shape


# In[146]:


downsampled = pd.concat([majority_downsampled, minority_down])
downsampled.shape


# # Assign data for algoritham

# In[147]:


print (train_merge.shape, downsampled.shape,upsampled.shape)


# In[148]:


train_merge =train_merge_orig


# # split train in train and test

# In[149]:


X_train, X_test,y_train, y_test = train_test_split(train_merge.loc[:,train_merge.columns != "target"],train_merge[["target"]], 
                                                   test_size = 0.10 ,random_state = 123)


# In[150]:


X_train.shape, X_test.shape ,y_train.shape, y_test.shape


# In[151]:


print("train=" , y_train.apply(pd.value_counts),"\n test=" ,y_test.apply(pd.value_counts))


# In[152]:


from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.values.ravel()) 
X_train_res.shape, y_train_res.shape


# In[153]:


print("train=" , pd.DataFrame(y_train_res).apply(pd.value_counts),"\n test=" ,y_train.apply(pd.value_counts))


# In[154]:


X_train = X_train_res


# In[162]:


y_train = y_train_res
y_train


# # Apply RandomForest Algoritham

# In[159]:


#RF_model= RandomForestClassifier(200,oob_score=True,random_state=13,n_jobs = -1, min_samples_leaf = 100,verbose=1)
RF_model = RandomForestClassifier(verbose=1)
                                     


# In[160]:


RF_model


# In[163]:


RF_model.fit(X_train, y_train)


# In[164]:


X_train_y_pred = (RF_model.predict_proba(X_train)[:,1]>=0.5).astype(int)
X_train_y_pred


# In[166]:


X_train_y_pred_prob = RF_model.predict_proba(X_train)
X_train_y_pred_prob


# In[167]:


pd.DataFrame(X_train_y_pred_prob)[0].value_counts(bins = 10)


# In[170]:


pd.DataFrame(X_train_y_pred).apply(pd.value_counts)


# In[171]:


X_train_y_pred_prob_pred = RF_model.predict(X_train)
X_train_y_pred_prob_pred


# In[172]:


X_train_y_pred, X_train_y_pred_prob, X_train_y_pred_prob_pred


# In[175]:


print("---------prediction-------------")
print(pd.DataFrame(X_train_y_pred).apply(pd.value_counts))

print("---------true value -------------")
print(pd.DataFrame(y_train).apply(pd.value_counts))


# In[183]:


## Accuracy MATRIX
print('RF Score: ', metrics.accuracy_score(y_train, X_train_y_pred))
## CONFUSION MATRIX

print("\n\n Confusion Matrix: \n", metrics.confusion_matrix(X_train_y_pred,y_train, ))


# In[180]:


print(metrics.classification_report(y_train,X_train_y_pred))


# In[ ]:


# import numpy as np
# import sklearn.metrics 
# y_pred1 = [0, 1, 1, 0,0]
# y_true = [0, 1, 0, 1,0]
# print(metrics,accuracy_score(y_true, y_pred1))
# print(metrics.confusion_matrix(y_true,y_pred1))


# # Prediction for X_test

# In[ ]:


X_test_y_pred = (RF_model.predict_proba(X_test)[:,1]>=0.5).astype(int)


# In[ ]:


print("---------true value-------------")
print(y_test.apply(pd.value_counts))
print("---------predication value-------------")
print(pd.DataFrame(X_test_y_pred).apply(pd.value_counts))


# In[ ]:


##Accuracy score
print('RF accuracy Score: ', metrics.accuracy_score(y_test, X_test_y_pred))
## CONFUSION MATRIX
print(metrics.confusion_matrix(y_test,X_test_y_pred))


#  # Prediction for test

# In[ ]:


y_pred = (RF_model.predict_proba(test_merge)[:,1]>=0.5).astype(int)


# In[ ]:


pd.DataFrame(y_pred).apply(pd.value_counts)


# In[ ]:


y_pred_prob = (RF_model.predict_proba(test_merge))


# In[ ]:


y_pred_prob


# In[ ]:


y_pred_prob_test = pd.DataFrame(y_pred_prob)
y_pred_prob_test.head(5)


# In[ ]:


submit = pd.DataFrame()
submit['id'] = test.id


# In[ ]:


submit.shape, test.shape


# In[ ]:


#pd.DataFrame(pred_values.iloc[:,1])
submit['target'] = y_pred_prob_test.iloc[:,1]


# In[ ]:


submit.head(5)


# In[ ]:


submit= submit.set_index('id')


# In[ ]:


submit.to_csv("driver-upsample.csv")


# In[ ]:




