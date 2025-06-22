우선 블로그를 참고하여 프로젝트 하나를 당장 시작했다.
EDA 과정에서 데이터의 특징을 분석함
기본적으로 케글에서 데이터 불러오는 법 부터 배움
```
!pip install kaggle # 설치
!mkdir -p ~/.kaggle
!cp kaggle.jason ~/ kaggle/
! chmod 600 ~/kaggle/kaggle.json
```
## 정리하면 선형회귀란 ?
독립변수와 종속변수의 선형적 관계를 파악하고, 다른 독립변수가 주어졌을 때, 그 일차식을 기반으로 y값을 예측하는 것. 
그냥 일차식이잖아. 뭐 이렇게 말을 어렵게 해.



# 케글에서 데이터 다운로드, read_csv로 파일 로딩
```
import kagglehub

path = kagglehub.dataset_download("salahuddinahmedshuvo/student-mental-stress-and-coping-mechanisms")
print("Path to dataset file: ", path) #dataset_download 함수사용

import numpy as np
import pandas as pd

# 데이터 로딩

dataset_path = path + '/Student_Mental_Stress_and_Coping_Mechanisms.csv' # 데이터 경로에다가 /파일이름
depression_data = pd.read_csv(dataset_path)
print(depression_data.head())
```
케글에서 흥미로워 보이는데이터셋 하나를 골라 선형 회귀 모델로 분석해봤다.

학생의 정신 건강 지수와 다른 변수들의 관계, 가장 영향이 큰 요소 파악

데이터이 일부분을 히스토그램으로 시각화

```
np.random.seed(seed)
import matplotlib.pyplot as plt # 시각화 모듈

data = depression_data['Social Media Usage (Hours per day)'] # 데이터 열 추출
bins = range(min(data), max(data)+2) # 끝 값과 range의 특성 고려해서 +2

plt.figure(figsize=(10,10)) # 그래프 프레임 (10, 10) 가로세로 사이즈
plt.subplot(3, 2, 1) # 그래프를 여러개 그릴 때, 그 위치를 정해줌 3행 2열에 1번 그래프
plt.hist(depression_data['Social Media Usage (Hours per day)'], bins=bins, color = 'skyblue', edgecolor = 'black'))
# bins = 가로 막대기 개수, 데이터의 최대값부터 최솟값까지 1칸씩 생성
plt.title('Social Media Usage (Hours per day)')
plt.xlabel('student')
plt.ylabel('Social Media Usage (Hours per day)')
```




![download](https://github.com/user-attachments/assets/d4debd63-de2d-46cc-bbab-6e25e6e9ad61)         
처음에 depression_data에서 stident_id 안 빼서 과적합 발생.
여기서 과적함 배움         
★ 과적합  : 각 학생들에게 너무 과하게 훈련되어 새로운 데이터에 대한 예측에는 적합하지 않게 되는 것!

```
depression_data # 그냥 데이터 불러온거임
depression_data = depression_data.drop(columns = ['Student ID'])
```
데이터 전처리
카테고리형 변수는 늘 수치화 해야함       
★ 원-핫 인코딩 : 상담참여, 부모의 정신 질환 여부, 성별 등의 범주 변수들을 수치화 해줌      
Y/N의 별도 열 만들고, 각각 0, 1 값 표현       
★다중공신성 : 모든 항을 y/n로 만들면 독립 변수끼리 서로 연관성이 생기는데,       
서로가 서로는 추측할 수 있기 떄문에, 반복되는 데이터 열 제거         
안 그러면 인공지능 학습에 차질 생김         
 => get_dummies와 drop_first=True로 범주형 변수 수치화하고 다중공신성 제거       

```
depression_encoded = pd.get_dummies(depression_data, drop_first = True)
# 첫번째 범주에 해당하는 더미 변수(변수 0,1변환) 제거
depression_encoded # 범주형 변수 수치화, 다중공신성 제거한 데이터
```

```
# 데이터 분리(학습/테스트)
from sklearn.model_selection import train_test_split
y_column = ['Mental Stress Level'] # 종속변수
x = depression_encoded.drop(y_column, axis=1) # 학습(특성)데이터 y빼고
# 열 빼는 함수 pd, dataframe에 .drop(["열 이름"])
y = depression_encoded[y_column] # 예측하려는 값
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
# test_size 총 데이터 분리 비율
# random_state = 42 모델이 랜덤으로 test, trian 데이터 나누는 데, 실행할 떄 마다 나누게 할 수 없으니, 고정시켜 같은 모델에 대한 결과 얻음
```

데이터 스케일링 : 서로 다른 수치 데이터 범주 맞추기. 수치 데이터에만 적용 가능

경사하강법

최소값 찾을 때          
미분계수 최소인 지점인데, 미분계수 너부 복잡하거나 데이터 많을 때           
기 > 0 : x 작아지는 쪽          
기 < 0 : x 커지는 쪽           
x(I+1) = x - 이동거리 * 기울기의 부호          
x(i+1) = x - 기울기의 크기 * 기울기의 부호          
x(i+!) = x - 기울기            
수렴속도 조절 ( 스텝사이즈, 러닝레이트)          

```
from sklearn.preprocessing import StandardScaler

encoded_colums = list(set(depression_encoded.columns) - set(depression_data.columns))
#  인코딩 후 데이터에서 원래 데이터의 가로행 뺴서, 추가된 컬럼만 확보
#  set을 왜 해...?
continuous_columns = list(set(depression_encoded.columns) - set(encode_columns) - set(y_column))
# 연속변수 추출 구문 : 타깃 변수 제외한 연속 변수만 추출

scaler = StandardScaler() # 서로 다른 수치 범위 맞춤
x_train_continuous = scaler.fit_transform(x_train[continuous_columns])
x_test_continuous = scaler.fit_transform(x_test[continuous_columns]) # 평균, 표준 편차 계산 후 표준화, 스케일링

x_train_continuous_df = pd.DataFrame(x_train_continuous, columns = continuous_columns)
x_test_continuous_df = pd.DataFrame(x_test_continuous, columns = continuous_columns)

x_train_categorical_df = x_train[encode_columns].reset_index(drop=True) # 데이터 인덱스 버리고, 컬럼으로 등록되는 것 막음
x_test_categorical_df = x_test[encode_columns].reset_index(drop=True)

x_train_final = pd.concat([x_train_continuous_df, x_train_categorical_df], axis=1)
x_test_final= pd.concat([x_test_continuous_df, x_test_categorical_df], axis=1) # 최종적으로 데이터 합침
```
# 학습 진행

```
x_train_final['bias'] = 1
x_test_final['bias'] = 1

![image](https://github.com/user-attachments/assets/aae1594e-6bf9-4da5-b6f9-b21f50bbca50)
y = w0(가중치) * x0(특성) ... b (편향)
```

```
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train_final, y_train) #학습
coefficients = linear_reg.coef_ # 계수
intercept = linear_reg.intercept_ # 절댓값

print('#'*20, '학습된 파라미터 값', '#'*20)
print(coefficients)
print('#'*20, '학습된 절편 값', '#'*20)
print(intercept)
```

여기서 학생들의 Id를 빼지 않아 학생 개인에게 너무 과적합된 모델 생성됨           

따라서 .drop으로 빼줌

# 모델 평가
 ```
from sklearn.metrics import mean_squared_error # 예측값과 실제값의 오차 제곱의 평균.
# 예측, trian, test
y_train_pred = linear_reg.predict(x_train_final)
y_test_pred = linear_reg.predict(x_test_final)

mse_train = mean_squared_error(y_train, y_train_pred) # 둘이 크기가 안맞음 test를 넣으면 어쩌니..
mse_test = mean_squared_error(y_test, y_test_pred)

print('학습 데이터를 이용한 MSE 값 :', mse_train)
print('평가 데이터를 이용한 MSE 값 :', mse_test)

# 실제 값, 예측 값 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Real Values')
plt.ylabel('Predict Values')
 # 아하 (x, y가 실제치, 이상치)
```


![download](https://github.com/user-attachments/assets/ac44cc26-2340-4a23-a0b3-929c802a5673)

#### 결과 해석
```
coeff_df = pd.DataFrame({'feature' : x_train_final.columns, 'coefficient' :linear_reg.coef_.flatten()}) # 이차원 배열 일차원으로 데이터 항목별 가중치 표 만들기

coeff_df['abs_coefficient'] = coeff_df['coefficient'].abs()
coeff_df_sorted = coeff_df.sort_values(by='abs_coefficient', ascending=False) # 기준, 내림차순
coeff_df_sorted # 가장 영향 큰 것부터
# 양수 가중치, 음수 가중치 구분, 절댓값 큰게 영향 큰것
```
예측값이 중심에 모임 => 선형 회귀로 다루기엔 변수들 간 관계 너무 복잡함            
=> 랜덤 포레스트로 전환.               


##### 잔차 분석 
잔차 이해 , 잔차 자체는 작되, 고르게 퍼져있어야 정확히 좋은 모델임                 
선형 회귀 모델의 성능을 분석하는 방법은 직선과 예측값의 잔차의 제곱을 최소로 하는 각 변수별 가중치와 절편을 찾는 과정이라고 볼 수 있다.  여러 변수에 의해 영향 받아 탄생한 종속 변수 값을 나타내는 식에        
변수와 종속 변수를 학습해 변수별 가중치를 부여하는 것이다.             

```
y_pred = linear_reg.predict(x_train_final)
residuals = y_train - y_pred # 정답 - 예측

# 잔차 시각화

plt.figure(figsize=(10,10))
plt.scatter(y_pred, residuals) #x축 y축
plt.axhline(y=0, color='red',linestyle='--')
plt.xlabel('Predict')
plt.ylabel('REesidual')
plt.title('Residual Analysis')
plt.show()
```

![download](https://github.com/user-attachments/assets/2418e69d-14d4-4170-99e6-28614a20f0f6)













