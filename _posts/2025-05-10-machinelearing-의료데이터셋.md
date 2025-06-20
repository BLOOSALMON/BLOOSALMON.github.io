---
title: "의료데이터셋 따라하기"
layout: single
---
데이터를 로드하는 법부터 배웠다.    
내가 아는 방식은 csv파일로 불러오는 방식밖에 없었기 때문에 파일을 파이썬 객체로 직접 로드하는 방식을 배웠다. 

```
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)


# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
print(heart_disease.metadata)

# variable information
print(heart_disease.variables)

X.head(10)
X_cp = X.copy()
y.head()
```

y값이 중간에 어디서 자꾸 int하나의 숫자로 바뀌는 오류가 발생한다.    
정확한 원인이 무엇인 지 몰라서 y.head(), y.info(), y.value_counts()로 확인도 해보고,    
series를 데이터 프레임으로 변환해서 columns 확인도 해봤는데 못 찾았다.    
그래서 그냥 y값이 살아있는 곳에서부터 pairplot을 그리는 연습을 했다.   
```
X_cp['target'] = y["num"]
import seaborn as sns
sns.pairplot (X_cp, hue = 'target', palette = 'mako')

 # hue는 x 변수 안에 있는 열을 기준으로 그려주는 것
 # vars :  옵션에 원하는 컬럼만
```
![download](https://github.com/user-attachments/assets/77942a90-8c6d-471a-81e0-6dc86971a1f4)

pairplot은 변수간 관계를 확인하는 것이다.    
이를 이용하여 다중공신성을 발견하거나, 값 구분에 적합한 변수를 찾을 수 있다.   
그래프의 해석은 두 변수간의 관계가 기준 target 값(성별 등)에 따라 어떻게 다른지 알아보는 것이다.    
즉, 어떤 변수 조합이 그룹을 잘 나누는 가를 나타내는 것이다. 

개별 변수들과 y값의 상관관계를 알 수도 있다.
```
X_cp.corrwith(y).plot.bar(
        figsize = (12, 4), title = "Correlation with Diabetes", fontsize = 12,
        rot = 90 , grid = True)
```

[상관관계]
age         : 0.32     
cholesterol : -0.05    
thalach    : -0.41 

[함수 및 변수 설명]

`figsize=(12, 4)` : 그래프 크기 (가로 12, 세로 4)   
`title="..."`     : 그래프 제목    
`fontsize=12`     : x축, y축 라벨 폰트 크기           
`rot=90`          : x축 라벨 90도 회전 (읽기 편하게)     
`grid=True`       : 그래프에 눈금선 표시               

상관관계가 높은 데이터를 사용할 수록 모델 성능이 좋아질 가능성이 높다.


![download](https://github.com/user-attachments/assets/d6f9ba02-6b18-4250-bd62-e1a471d48873)

```
sns.pairplot(X_cp, hue = 'sex', vars = ['age', 'restecg', 'fbs']) # vars :  옵션에 원하는 컬럼만
```

```
sns.scatterplot(x = 'age', y = 'thalach', hue = 'target', data = X_cp) # 색이 섞여있으면 잘 구분 안된것
```
![download](https://github.com/user-attachments/assets/638ba762-e366-4538-b323-c4f252b751d8)

```
X_cp.corrwith(X_cp['target']).plot.bar() # 각 열과 y 값의 상관계수
    figsize = (12,4), title = "Correlation with Diabetes", fontsize = 12,
        rot = 90, grid = True)
```


![download](https://github.com/user-attachments/assets/56a4da83-2e5e-412b-a334-4d0a7896e89a)

X_cp의 변수 탐색을 위해 수치형 변수를 카테고리형 변수로 바꾸는 작업이다.   
의미를 설명하자면,   
df['chest_pain_type'] == 0 =>	chest_pain_type 열의 값이 0인 행을 필터링한다.   
df.loc[조건, '열이름']	.loc[] => 행과 열을 명시적으로 지정하여 값을 읽거나 쓸 수 있게 한다.   
'typical angina' => 조건을 만족하는 행들의 'chest_pain_type' 값을 'typical angina'로 바꾼다.   

```
X.describe(include=['object'])
```
수치형 변수들을 우선 확인해준다.



```
X_cp["sex"] = X.sex.apply(lambda x:'male' if x==1 else 'female')
X_cp["fbs"] = X.fbs.apply(lambda x: 'yes' if x==1 else "No")
X_cp["exang"] = X.exang.apply(lambda x: 'yes' if x==1 else "No")

X_cp.loc[X["cp"] == 1, "cp"] = "전형적 협심증 "
X_cp.loc[X["cp"] == 2, "cp"] = "비정형 협심증 "
X_cp.loc[X["cp"] == 3, "cp"] = "비협심증 통증 "
X_cp.loc[X["cp"] == 4, "cp"] = "무증상 "

X_cp.loc[X["restecg"] == 0, "restecg"] = "정상"
X_cp.loc[X["restecg"] == 1, "restecg"] = "ST-T 이상"
X_cp.loc[X["restecg"] == 2, "restecg"] = "좌심실 비대"

# ST 분절 기울기 (slope)
X_cp.loc[X_cp['slope'] == 1, 'slope'] = 'upsloping'
X_cp.loc[X_cp['slope'] == 2, 'slope'] = 'flat'
X_cp.loc[X_cp['slope'] == 3, 'slope'] = 'downsloping'

# 혈류 검사 결과 (thal)
X_cp.loc[X_cp['thal'] == 3, 'thal'] = 'normal'
X_cp.loc[X_cp['thal'] == 6, 'thal'] = 'fixed defect'
X_cp.loc[X_cp['thal'] == 7, 'thal'] = 'reversible defect'
```


잘되었는 지 확인해준다.
```
X_cp['restecg'].value_counts()
```
restecg	
정상	151   
좌심실 비대	148   
ST-T 이상	4   

dtype: int64


 y값의 분포를 확인하기 위한 표를 그린다.
```
 import matplotlib.pyplot as plt
import seaborn as sns

# y 값 0~4분포 확인
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(10,5), facecolor=(.94, .94, .94))
# 파라미터 설명 : 1행 2열의 서브플롯에 그림.

ax1 = y.value_counts().plot.pie(autopct = "%1.0f%%", labels = ["Normal", "Heart Disease_lower","Heart Disease_low","Heart Disease_high","Heart Disease_higher"], startangle = 60, ax=ax1, colors = sns.color_palette("crest"));
# ax 그래프 그릴 위치

ax1.set(title ='Percentage of Heart disease patients in Dataset' ) # set :  속성 설정 함수


ax2 = y.value_counts().plot(kind = "barh", ax = ax2)
for x, y in enumerate(y.value_counts().values): # 값만 ndarray로 추출 [165, 138] (1번이 165명, 0번이 138명)
  ax2.text(.5,x, y, fontsize=12 ) # x좌표, y좌표, 표시될 내용
ax2.set(title = 'No. of Heart disease patients in Dataset')
plt.show()
```

![download](https://github.com/user-attachments/assets/93bb879d-2d80-4c76-810e-911f3d8a325e)
```
plt.figure(figsize=(16,10))
plt.subplot(221)
X_cp["sex"].value_counts().plot.pie(autopct = "%1.1f%%", colors = sns.color_palette("crest", 5), startangle = 60, labels = ["Male", "Female"],wedgeprops = {"linewidth" : 2, "edgecolor" : "k"}, explode = [.1, .1],  shadow = False)
plt.title("Distribution of Gender") # explode = 중심에서 얼마나 떨어져 있을지
plt.subplot(222)
ax = sns.displot(X_cp['age'], rug = True, kde=True)
# plt.subplot()은 axes-level plot 을 위한 공간을 정의하고,
# displot은 자체적으로 새로운 객체 생성해, subplot 안에 그려지지 않음
plt.title("Age wise distribution")
#rug 개별데이터(hist 내장 안됨), kde 데이터 부드러운 곡선
plt.show()
```
![download](https://github.com/user-attachments/assets/b72fde3a-d6c6-4a64-9c62-cc7a14078bac)
```
df_1 = X_cp[y["num"] != 0]
df_0 = X_cp[y["num"] == 0] # DataFrame 인 경우, column 이름 지정해 추출

fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1, 2), (0,0))
sns.distplot(df_0['age'])
plt.title('AGE DISTRIBUTION OF NORMAL PATIENTS', fontsize=15, weight='bold')

ax1 = plt.subplot2grid((1, 2), (0, 1))
sns.countplot(x = df_0['sex'], palette = 'viridis')
plt.title('GEDNER DISTRIBUTION OF  NORMAL PATIENTS PATIENTS', fontsize = 15, weight = 'bold')
plt.show()

fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.histplot(df_1['age'])
ax1.set_title('AGE DISTRIBUTION OF HEART DISEASE PATIENTS', fontsize=15, weight='bold')

ax2 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x=df_1['sex'], palette='viridis')
ax2.set_title('GENDER DISTRIBUTION OF HEART DISEASE PATIENTS', fontsize=15, weight='bold' )
plt.show()

```
![download](https://github.com/user-attachments/assets/c072b5e4-8e98-409c-bff2-85ff963b76b5)
![download](https://github.com/user-attachments/assets/a82155ac-a9b5-4f6b-acca-a3a79c9d07a1)


처음에 두 distplot으로 그리니 두 그래프 분리되고, age의 값이 gender 그래프에 그려짐
결과를 중심으로 왜 안돼나 사고하자.

```
fig = plt.figure(figsize = (15,5))
ax1 = plt.subplot2grid((1, 2), (0,0))
sns.countplot(x=df_0['cp'], palette ='ch:start=4.2,rot=-3.3' )
plt.title('CHEST PAIN OF NORMAL PATIENTS', fontsize=15, weight='bold')

#plotting heart patients
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x =df_1['cp'], palette='icefire') # 범주형 변수 개수 세서 그래프 그려줌   
plt.title('CHEST PAIN OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()

# 이전에서 y값이 아마 숫자 하나로 바뀌어서 경우에 따른 추출이 불가능하다.
```

subplot2grid 가 이미 객체를 생성하는데 displot이 또 객체를 만들어서 하나로 안합쳐지고 따로 그려진다.   

아래 코드는    
심장병이 있는 환자드의 cp지수와 없는 자들의 cp를비교한 것이다.    
각 타겟값 별로 변수의 분포를 확인할 수 있다니     
 비교해본 결과, 타겟값은 전형적 흉통이 많은 것으로 나타났다.   
 
```
fig = plt.figure(figsize = (15,5))
ax1 = plt.subplot2grid((1, 2), (0,0))
sns.countplot(x=df_0['cp'], palette ='ch:start=4.2,rot=-3.3' )
plt.title('CHEST PAIN OF NORMAL PATIENTS', fontsize=15, weight='bold')

#plotting heart patients
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x =df_1['cp'], palette='icefire')
plt.title('CHEST PAIN OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()
```
![download](https://github.com/user-attachments/assets/2911b07f-06a6-497a-ae9b-c3757f7ab470)

```
# plotting normal patients
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(x =df_0['restecg'], palette ='ch:start=.2,rot=-.3')
plt.title('REST ECG OF NORMAL PATIENTS', fontsize=15, weight='bold')

#plotting heart patients
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x =df_1['restecg'], palette='cubehelix')
plt.title('REST ECG OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()
```

![download](https://github.com/user-attachments/assets/0673a750-a093-4d01-b4ae-ca0edec3d667)
심장병이 있는 환자들이 오히려 정상 범주가 많다.


```

# MODELIING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   X_cp,X_cp['target'], test_size=0.2, random_state=9
)

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def GetBasedModel():
  baseModels = []
  baseModels.append(('LR_L2', LogisticRegression(panalty = '12')))
  baseModels.append(('KNN5', KNeighborsClassifier(5)))
  baseModels.append(('KNN5', KNeighborsClassifier(7)))
  baseModels.append(('KNN5', KNeighborsClassifier(9)))
  baseModels.append(('KNN5', KNeighborsClassifier(11)))
  baseModels.append(("CART", DecisionTreeClassifier()))

  # 서포트 벡터 머신공부
  baseModels.append(("SVM_Linear", SVC(kernel='linear', gamma = 'auto', probability = True)))
  # 선형분류를 하겠다는 의미
  # 확률 아는 함수 model.predict_proba 사용
```

모델링을 하고 수치화를 하려다가 값에 결측치가 있는 오류가 떴다.
```
X_train_R.isna().sum()
```
로 확인하고
```
X_train_R = X_train.dropna() # NAN 값 있는 행 삭제 how = 'any' : 한개라도 / how = 'all' : 모두
```
로 지우고
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

numeric_df = X_train_R.select_dtypes(include='number')

X_train_R[numeric_df.columns] = scaler.fit_transform(X_train_R[numeric_df.columns])
X_train_R.head() # 범주화, 정규화
```
로 정규화 및 범주화를 진행했다.

  age	 sex 	cp	 trestbps	 chol	 fbs 	restecg	 thalach 	exang 	oldpeak	 slope	 ca	 thal	 target     
132	0.000000	1.0	0.333333	0.339623	0.178082	0.0	1.0	1.000000	0.0	0.000000	0.0	0.000000	0.00	0.00    
215	0.562500	1.0	0.000000	0.245283	0.152968	0.0	1.0	0.694656	0.0	0.306452	0.5	0.000000	1.00	0.00     
213	0.770833	0.0	1.000000	0.792453	0.232877	1.0	0.0	0.717557	1.0	0.161290	0.5	0.666667	1.00	0.75     
229	0.770833	1.0	1.000000	0.169811	0.196347	0.0	1.0	0.465649	1.0	0.016129	0.0	0.333333	0.00	0.50     
286	0.604167	0.0	1.000000	0.716981	0.226027	1.0	1.0	0.572519	1.0	0.451613	0.5	0.666667	0.75	0.50     
```
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def GetBasedModel():
  baseModels = []
  baseModels.append(('LR_L2', LogisticRegression(panalty = '12')))
  baseModels.append(('KNN5', KNeighborsClassifier(5)))
  baseModels.append(('KNN5', KNeighborsClassifier(7)))
  baseModels.append(('KNN5', KNeighborsClassifier(9)))
  baseModels.append(('KNN5', KNeighborsClassifier(11)))
  baseModels.append(("CART", DecisionTreeClassifier()))
```

혹시 몰라 y값에도 결측치가 있는 지 확인했다.   
seires 객체여서 info를 볼 수 없어서 to_frame 함수로 바꿨다.

```
y_num = y_train_R.to_frame(name = 'num')
```
과정 중 x와 y 값의 개수가 일치하지 않는다는 에러가 떴다.   
그래서 결측치가 존재하는 행을 y에서도 지웠다.   
그래도 맞지 않아 결국 못했다.    

```
from sklearn.svm import SVC
svc = SVC(kernel='linear',gamma='auto',probability=True)
svc.fit(X_train_R, y_train_R)
y_pred_svc = svc.predict(X_test)
```


ValueError: Found input variables with inconsistent numbers of samples: [237, 242]







