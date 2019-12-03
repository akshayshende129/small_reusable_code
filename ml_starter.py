import pandas as pd

####Pre-Processing######
org_data = pd.DataFrame(pd.read_excel('./data/lecture12.xlsx',na_values='not available'))

def convert_income(cell):
    return 'NaN' if cell < 0 else cell

def convert_age(cell):
    cell = int(cell)
    return 'NaN' if cell < 0 or cell > 100 or cell == 0 else cell


df_cust = pd.read_excel('./data/lecture12.xlsx',na_values=['not available','n.a'],converters={'income':convert_income,'age':convert_age})


#df_cust.to_excel('./data/lecture12_my.xlsx') #NA not shown and indexs are enbaled
df_cust.to_excel('./data/lecture12_my.xlsx',na_rep='NaN',index=False,sheet_name='customer')

df_cust.head(10)

#Data Pre-Processing
df_cust['income'].isnull().sum()
df_cust.isnull().sum()
df_cust['is.employed']
max(df_cust['is.employed'])

isemp = df_cust['is.employed'].max()
inc = df_cust['income'].median()
housetyp = df_cust['housing.type'].mode()
rmove = df_cust['recent.move'].max()
nveh = df_cust['num.vehicles'].mean()
age = df_cust['age'].mean()

ndf_cust = df_cust.fillna({'is.employed':isemp,'income':inc,'housing.type':housetyp,'recent.move':rmove,'num.vehicles':nveh,'age':age})
ndf_cust.to_excel('./data/lecture12_my.xlsx',na_rep='NaN',index=False,sheet_name='customer')

#To replace one value to another value
ndf_cust['is.employed'].replace(to_replace = 1,value =True, inplace=True, method = None)
ndf_cust['is.employed'].replace(to_replace = 0,value =False, inplace=True, method = None)
ndf_cust['recent.move'].replace(to_replace = 1,value =True, inplace=True, method = None)
ndf_cust['recent.move'].replace(to_replace = 0,value =False, inplace=True, method = None)

#Remove Duplicates
ndf_cust_new = pd.read_excel('./data/lecture12_my_dup.xlsx')
#ndf_cust_new.to_excel('./data/lecture12_my_dup.xlsx')
#ndf_cust_new.duplicated()


#Summary Stats 
import numpy as np
import pandas as pd
import scipy
from scipy import stats

ndf_cust.sum()
ndf_cust.describe()

#Pearson Corelation
import scipy
from scipy.stats.stats import pearsonr

df_cust.__dict__
pearsonObj = stats.pearson3(ndf_cust['income'],ndf_cust['age'])

pearsonVal,coeff = pearsonr(ndf_cust['income'],ndf_cust['age'])
print(pearsonVal,coeff)
ndf_cust.corr()

#DBSCAN Clustering
import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

dbscan_df = pd.read_excel('./data/lecture12_org_pca.xlsx')
dbscan_df_new = dbscan_df.loc[0:,'hair':'eggs']
model = DBSCAN(eps=0.5,min_samples=4).fit(dbscan_df_new)
print(Counter(model.labels_)) # -1:Outliers
outlier_df = pd.DataFrame(dbscan_df_new)
print(outlier_df[model.labels_ == -1])

#KMeans Clustering
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.metrics as sm


df_animals = pca_df
#df_animals.columns.name
v1 = preprocessing.scale(df_animals['hair'].values)
v2 = preprocessing.scale(df_animals['feathers'].values)
v3 = preprocessing.scale(df_animals['eggs'].values)
v17 = df_animals['type'].values

data = np.array(list(zip(v1,v2,v3)))
target = np.array(list(zip(v17)))

target_list = [v for v in target if v==3]
target_list
    
	
#KNN
import os
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics

path = "./data/KNN/knn.xls"
df = pd.read_excel(path)
#Predictors
cer_data = df.loc[:,'calories':'rating']
#Predicting Vlaue
typ = df.loc[:,'shelf'].values
#Preprocessing
cer = sklearn.preprocessing.scale(cer_data)
#Splitting
x = cer_train,cer_test,typ_train,typ_test = train_test_split(cer,typ,test_size = .33,random_state = 15)
#Initilize model object
classifierObj = neighbors.KNeighborsClassifier()
#Fitting adata according to the model
classifierFittting = classifierObj.fit(cer_train,typ_train)
#Predicting
typ_expected = typ_test
typ_predict = classifierFittting.predict(cer_test)
print(metrics.accuracy_score(typ_expected,typ_predict))

p = [70,3,4,123,15,2,6,234,35]
pr = np.array(p).reshape(1,9)
#shelf = classifierFittting.predict(pr)
print(pr)
#print(shelf)


#Navie Bayes
import os
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics


path = "./data/Navie Bayes/naviebayes.xls"
df = pd.read_excel(path)
X = vechile_predictor = df.loc[:,'Compactness':'Hollows_ratio']
Y = vechile_prediting = df.loc[:,'type']

#train_var_scale = sklearn.preprocessing.normalize(train_var)
train_var_scale = sklearn.preprocessing.minmax_scale(X)
#   Spilts data set into into(Train,Validation Tests)
# 1) X_Train Set
# 2) X_Validation Set
# 3) Y_Train Set
# 4) Y_Validation Set
Datasets = vechile_predictor_train,vechile_predictor_validation, vechile_prediting_train,vechile_prediting_validation = train_test_split(X,Y,test_size = .33, random_state = 15)

# Create Bernoulli Naive Bayes object with prior probabilities of each class
modelInit = []
modelInit.append(('Bernoulli',BernoulliNB()))
modelInit.append(('Gaussian',GaussianNB()))
modelInit.append(('Multinomial',MultinomialNB()))
for i in range(len(modelInit)):
    model = modelInit[i][1]
    #Train Model
    model = model.fit(vechile_predictor_train,vechile_prediting_train)
    #predicting
    predict = model.predict(vechile_predictor_validation)
    print(modelInit[i][0],metrics.accuracy_score(vechile_prediting_validation,predict),sep = " : ")
	
	
#logistics Regression
import os
import numpy as np
import pandas as pd
import sklearn
import scipy 
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

path = "./data/Logistics/logi.xls"
df = pd.read_excel(path)
#df.columns
X = cer_data = df.loc[:,['calories','fat','carbo','vitamins']]
Y = typ_data = df.loc[:,'shelf']
df.pro = df.protein.values
df.fat = df.fat.values

x = peasron_coeff,p_valu = spearmanr(df.protein.values,df.fat.values)

cer_data_scaled = sklearn.preprocessing.scale(cer_data)
y = X_train,X_Validation,Y_train,Y_Validation = train_test_split(X,Y,test_size = .33,random_state = 101)
#model
model = LogisticRegression()
model = model.fit(X_train,Y_train)
#predicting
predict = model.predict(X_Validation)
print(metrics.accuracy_score(Y_Validation,predict),sep = " : ")

#Linear Regression
import os
import numpy as np
import pandas as pd
import sklearn
import scipy 
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#for model summary
import statsmodels.api as sm


path = "./data/Linear/linear.xlsx"
df = pd.read_excel(path)
df.columns
X = sale_data = df.loc[:,'Lot Area':'Garage Area']
Y = typ_data = df.loc[:,'SalePrice']

X_scaled = sklearn.preprocessing.scale(X)
y = X_train,X_Validation,Y_train,Y_Validation = train_test_split(X,Y,test_size = .33,random_state = 101)

#model
model = LinearRegression()
model = model.fit(X_train,Y_train)
#predicting
predict = model.predict(X_Validation)

#Linear Regression
import os
import numpy as np
import pandas as pd
import sklearn
import scipy 
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#for model summary
import statsmodels.api as sm


path = "./data/Linear/linear.xlsx"
df = pd.read_excel(path)
df.columns
X = sale_data = df.loc[:,'Lot Area':'Garage Area']
Y = typ_data = df.loc[:,'SalePrice']

X_scaled = sklearn.preprocessing.scale(X)
y = X_train,X_Validation,Y_train,Y_Validation = train_test_split(X,Y,test_size = .33,random_state = 101)

#model
model = sm.OLS(Y_train,X_train).fit()
predict = model.predict(X_Validation)
#model.summary()

#Random Forest
import os
import numpy as np
import pandas as pd
import sklearn
import scipy 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#for model summary
import statsmodels.api as sm
import matplotlib.pyplot as mplt

path = "./data/Random Forest/randomforest.xlsx"
df = pd.read_excel(path)	
label = df.iloc[3,1:].values
mplt.imshow(label.reshape(28,28))
X = df_predictor = df.iloc[:,1:]
Y = df_target = df.iloc[:,0]
Datasets = X_train,X_Validation,Y_train,Y_Validation = train_test_split(X,Y,test_size = .33,random_state = 101)

rfModelObj = RandomForestClassifier(n_estimators= 100)
modelTrain = rfModelObj.fit(X_train,Y_train)
predict = modelTrain.predict(X_Validation)
