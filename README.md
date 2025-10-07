## EXNO:4-DS
## Name : VIJAYASHREE B
## Reg No : 212223040238
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```

<img width="1350" height="867" alt="image" src="https://github.com/user-attachments/assets/f9a8fcdf-30d4-4528-a1d9-f25783986ad0" />

```
data.isnull().sum()
```

<img width="472" height="647" alt="image" src="https://github.com/user-attachments/assets/f65992d2-6bef-4841-ae4d-8842b331774f" />

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1335" height="823" alt="image" src="https://github.com/user-attachments/assets/300875e6-ba1c-4aae-b540-f0a36de84610" />

```
data2=data.dropna(axis=0)
data2
```

<img width="1357" height="838" alt="image" src="https://github.com/user-attachments/assets/d0e7f79f-4a1c-4ca8-a281-474a1c8681a1" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="1350" height="496" alt="image" src="https://github.com/user-attachments/assets/fd888f92-9551-4b09-8060-fc82af40c799" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="601" height="608" alt="image" src="https://github.com/user-attachments/assets/ebb431f6-aec6-4d8b-bc4c-f73217e025a8" />

```
data2
```

<img width="1360" height="772" alt="image" src="https://github.com/user-attachments/assets/9ca9d101-85de-4bca-b7a6-e5f537524913" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1358" height="661" alt="image" src="https://github.com/user-attachments/assets/d862c898-dfac-47be-978a-875c4b0a5945" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1357" height="110" alt="image" src="https://github.com/user-attachments/assets/2d87876b-8f54-4be7-b315-c3a6b2e4c081" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1355" height="115" alt="image" src="https://github.com/user-attachments/assets/d424a0d6-7ab5-4124-bcfd-6901390ffdc0" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="336" height="113" alt="image" src="https://github.com/user-attachments/assets/12b7a9c9-ffd5-4aa8-a1f8-c2b1904949f1" />

```
x=new_data[features].values
print(x)
```

<img width="616" height="241" alt="image" src="https://github.com/user-attachments/assets/8c4f2a8c-9bd5-46e8-bac5-118a38d8254f" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="1037" height="187" alt="image" src="https://github.com/user-attachments/assets/ca929089-56f9-4e46-9e19-2aacc18d4bbc" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="567" height="147" alt="image" src="https://github.com/user-attachments/assets/332c25c2-881c-4a49-9b3b-8b2129132b39" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="613" height="105" alt="image" src="https://github.com/user-attachments/assets/40e5f84e-9b1a-4bf5-aad4-12cd88a7b9ce" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="811" height="83" alt="image" src="https://github.com/user-attachments/assets/170a09b3-8d0b-4128-8505-27aa9cf385d5" />

```
data.shape
```

<img width="232" height="90" alt="image" src="https://github.com/user-attachments/assets/4bbabe07-8951-4e05-aa12-81f1eaf3c93f" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1332" height="525" alt="image" src="https://github.com/user-attachments/assets/8133e47b-4fce-4e57-8d40-8cd8ca6db12b" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="952" height="417" alt="image" src="https://github.com/user-attachments/assets/847836fe-c2ea-432b-ba0f-92f576a9ecb8" />

```
tips.time.unique()
```

<img width="593" height="112" alt="image" src="https://github.com/user-attachments/assets/3b9e1dae-25bc-4058-aa7c-58fdbb009ea3" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="751" height="165" alt="image" src="https://github.com/user-attachments/assets/5013b4ae-ec35-4676-9a4a-3c3a8504bdcf" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="632" height="157" alt="image" src="https://github.com/user-attachments/assets/cd9ae63c-445e-4be8-a374-4d5f0b3f07d3" />




# RESULT
   Thus, Feature selection and Feature scaling has been used on thegiven dataset.
