## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df

![1](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/9ba44dd5-4788-42e1-9535-466f9e61328d)
# OrdinalEncoder

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
`
![2](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/065f15eb-f821-45a8-a235-900aa9043f89)

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

![3](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/11717fbf-5df6-4e58-ae7a-0a911cd7b8d9)
# LabelEncoder

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc

![4](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/08bfaf7f-0334-4c7f-8c36-1f9e1bc2ef91)
# OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2

![5](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/1efa0cd0-a6be-4678-8e1c-db27ef989712)

pd.get_dummies(df2,columns=["nom_0"])

![6](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/a84369ac-98ba-4018-9cb0-06381bd69796)
# BinaryEncoder

pip install --upgrade category_encoders

![7](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/d928a16f-05a5-4c2b-b62c-be60b164f1d1)

from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv')
df

![8](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/10d820e3-d836-4c02-99de-93739b5e62c3)

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb

![9](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/2f25e266-1fa8-4596-8a48-9604e0058a5a)
# Target Encoder

from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

![10](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/f0961ea6-d2d3-4ca5-9736-48adca400346)
# Data Transformation

import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('/content/Data_to_Transform.csv')
df

![11](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/1e2cab6e-0310-490f-bfb2-d5298a4ae22d)

df.skew()

![12](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/5a0e0204-b9a4-4e30-a945-73dfabfa3237)

np.log(df["Highly Positive Skew"])

![13](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/b720cc16-4fe5-4bde-b3a7-d3d959381d6b)

np.reciprocal(df["Moderate Positive Skew"])

![14](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/3c838503-07d7-4a1e-b0bf-f9ea2f063093)


np.sqrt(df["Highly Positive Skew"])

![15](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/ab4cee66-802a-4367-bde4-6ce122e8e57d)


np.square(df["Highly Positive Skew"])

![16](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/6788dcfb-b1cc-4611-8d9a-dba35c6e0554)


df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

![17](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/6dcefcd2-2285-429d-bcdf-87d72b57ce9a)

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

![18](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/92a5f023-6c24-42fa-9670-d5ba017d7e45)


df.skew()

![19](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/e2fbdf97-257e-424d-9e8f-1c6364d08fc5)


df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

![20](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/d6191ad7-12ce-407b-a516-8b88065bba6e)


import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![21](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/54176218-5940-4a95-b720-a44d1fd60e63)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

![22](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/d7ebfd01-454f-4347-81de-e72a8c3c06eb)


from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

![24](https://github.com/chgeethika/EXNO-3-DS/assets/142209368/f7443531-3456-4018-8b84-09d1d506d7ca)

# RESULT:
      Finally,perform Feature Encoding and Transformation process is executed successfully.
