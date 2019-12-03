# SMALL CODE

<!-- toc -->

  * [Given a time in 12-hour AM/PM format, convert it to military (24-hour) time. _Input Must be in string_](#Given-a-time-in-12-hour-AMPM-format-convert-it-to-military-24-hour-time-_Input-Must-be-in-string_)
  * [Checking float or not in a series](#Checking-float-or-not-in-a-series)
  * [GroupBy Impute](#GroupBy-Impute)
- [Feature Stats](#Feature-Stats)
  * [VIF](#VIF)
  * [Chi<sup>2</sup> Test](#Chisup2sup-Test)
- [Correlation](#Correlation)
  * [Plot](#Plot)
- [Model](#Model)
  * [Regression Neural Network](#Regression-Neural-Network)
- [Getting NULL count in %](#Getting-NULL-count-in-)
- [Remove High Correlated Features](#Remove-High-Correlated-Features)
- [Check which feature are highly realted to other](#Check-which-feature-are-highly-realted-to-other)
  * [Single Feature Testing](#Single-Feature-Testing)
  * [Entire Dataset : Return Dict with realted columns](#Entire-Dataset--Return-Dict-with-realted-columns)

<!-- tocstop -->



## Given a time in 12-hour AM/PM format, convert it to military (24-hour) time. _Input Must be in string_

```
def timeConversion(s):
    s = s.lower()
    if s.find('pm') != -1:
        hr = s.split(':')[0]
        if int(hr) < 12:
            return str(12 + int(hr))+''.join(x for x in s[len(hr):]).strip('pm')
        elif hr == 12:
            return str(12 - int(hr))+''.join(x for x in s[len(hr):]).strip('pm')
        else:
            return s.strip('pm')
    if s.find('am') != -1:
        return s.strip('am')
        
print(timeConversion('7:12:14PM')
print(timeConversion('7:12:14AM'))
```

## Checking float or not in a series
```
def check_float(x):
    try:
        if x % 1 > 0:
            return x
    except:
        pass
```
        
        
        
## GroupBy Impute
```
def impute_value(data,columns,valuetype,group=None):
    assert isinstance(columns,list), 'Columns name must be in a list'
    assert len(columns) != 0, 'No columns passed'
    assert group, "No grouping column passed"
    grouped = data.groupby(group)
    if valuetype.lower() == 'integer':
        for col in columns:
            data[col].fillna(grouped[col].transform('mean').astype('int'),inplace=True)
    elif valuetype.lower() == 'median':
        for col in columns:
            data[col].fillna(grouped[col].transform('median'),inplace=True)
    elif valuetype.lower() == 'mode':
        for col in columns:
            data[col].fillna(grouped[col].transform('mode'),inplace=True)
    else:
        for col in columns:
            data[col].fillna(grouped[col].transform('mean'),inplace=True)
    
    return data
```  

# Feature Stats

## VIF
**String columns are not accepted, Please drop those or one-hot encode them before using**
```
import statsmodels.api as sm
import statsmodels.formula.api as smf

def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

```

## Chi<sup>2</sup> Test

```
import itertools
from scipy import stats
#stats.chi2_contingency(cont)
def get_chi2_stats(data,columns):
    columns = list(itertools.product(L,L))
    result = {}
    for c1,c2 in columns:
    if c1 != c2:
        cont = pd.crosstab(data[c1],data[c2],margins=True)
        chi_score, p_value, dof = stats.chi2_contingency(cont)[:3]
        #result[(c1,c2)] = {'chi_score' : chi_score, 'p-value' : p_value, 'dof':dof}
        result[(c1,c2)] = round(p_value,2)
    return result

```



# Correlation 

## Plot
```
def get_corr_plot(data,figsize=(25,25)):
    sns.set(style="whitegrid")
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    return sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)

get_corr_plot(X,(40,40)) # data,plot_size in tuple format

```



# Model

## Regression Neural Network
```
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#  For Your understanding using random sample
X = np.random.random_sample(25).reshape(5,5)
y = np.random.random_sample(5).reshape(5,1)

X_shape = X.shape[0]
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_shape]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])

model.summary()

model.fit(X, y, epochs=1000, validation_split = 0.2, verbose=0)

model.predict(np.random.random_sample(25).reshape(5,5))

```


# Getting NULL count in % 
```
def get_null_count(data):
    null = pd.DataFrame((data.isnull().sum() / data.shape[0]) * 100)
    null.reset_index(inplace=True)
    null.columns = ['features','percentage']
    null.sort_values('percentage',ascending=False,inplace=True)
    return null.loc[null['percentage'] > 0,:].reset_index(drop=True)
```

# Remove High Correlated Features

```
def remove_highly_correalated_fature(data, thresh=0.8):
    corr = data.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    drop = [col for col in upper if any(upper[col] > thresh)]
    return data.drop(drop,axis=1)
```

# Check which feature are highly realted to other

## Single Feature Testing
```
def get_correlated_features(feature_name,data,thresh=0.7):
    corr = data.corr().abs() > thresh
    corr = corr.where(np.tri(corr.shape[0],corr.shape[1],k=-1).astype(bool))
    return corr.where(corr.loc['homicides_per_100k',:].dropna() > thresh).dropna(how='all').index.tolist()
    
```
    # Example
    dataset : data
    features = [ 'a','b','c','d']
    get_correlated_feature('a',data,0.8)
    # O/P : highly related features with feature a 
    ['b','d']

## Entire Dataset : Return Dict with realted columns 
```
import numpy as np
import pandas as pd

def get_correlated_features(data,thresh=0.7):
    corr_dict = {}
    corr = data.corr().abs() > thresh
    columns = corr.columns
    corr = corr.where(np.tri(corr.shape[0],corr.shape[1],k=0).astype(bool))
    for feature_name in columns:
        try: 
            lst = corr.where(corr.loc[feature_name,:].dropna() > thresh).dropna(how='all').index.tolist()
            if len(lst) > 0 :
                corr_dict[feature_name] = lst
        except Exception as e:
            print(e)
    return corr_dict
```

























