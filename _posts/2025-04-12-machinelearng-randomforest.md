선형 회귀 모델로 변수 간 관계를 파악하기에 너무 관계가 복잡하다는 판단하에
비선형 관계에 적합하다는 랜덤 포레스트를 사용해본다.
크게 두 가지 특징 중요도와 정규화를 배웠다.

정규화란, 데이터 분포가 너무 치우쳐져 있을 때, log값을 취해 중앙으로 맞추는 것. 나중에 다시 지수변환함.
np.log 사용해 종속 변수 정규화
```
# 정신 건강 지수 자연로그 반환
import seaborn as sns # 좀 더 에쁜 시각회 
from scipy import stats # 수학패키지의 함수
depression_data['Mental Stress Level_log'] = np.log(depression_data['Mental Stress Level'])

fig, axes = plt.subplots(1, 2, figsize = (12, 5))

sns.histplot(depression_data['Mental Stress Level'], bins = 10, kde = True, ax = axes[0])
axes[0].set_title('Original Data')

sns.histplot(depression_data['Mental Stress Level_log'], bins = 10, kde = True, ax = axes[1])
axes[1].set_title('Log Transformed Data')

plt.show()

# 정규성 확인

stats.probplot(depression_data['Mental Stress Level_log'], dist = "norm", plot = plt) # 정규분포 시각화 함수 problot
plt.title('Q-Q Plot of THING_AMT_LOG')
plt.show()
```



![download](https://github.com/user-attachments/assets/5c0b1b5b-283e-4d05-a37c-28cac1218f86)






![download](https://github.com/user-attachments/assets/48689b32-94cf-40ad-96e3-082026024a15)

내 데이터를 확인해보니 그렇게 치우져져 있지는 않아 큰 위미는 없을 듯 하다

# 하이퍼 파라미터 찾기
```
from sklearn.model_selection import RandomizedSearchCV # 하이퍼 파라미터 객체
dist = {
    'regressor__n_estimators': list(range(50, 201, 10)), # 나무 개수
    'regressor__max_depth' : [5, 10, 15, 20, 25], # 트리 깊이
    'regressor__min_samples_split' : [2, 4, 6, 8, 10] # 최종적으로 남게 하는 데이터 개수. 너무 적으면 과적합, 너무 많으면 과소적합.
}
rf = RandomForestRegressor(random_state=42) # 모델

cat_features = ['Counseling Attendance','Stress Coping Mechanisms','Family Mental Health History','Medical Condition','Gender']
num_features = [col for col in depression_data.columns if col not in cat_features] # 처음에 encoded 데이터를 사용하는 바람에
파이프라인에 들어가야할 기본 데이터가 전처리가 된 컬럼 명이어서,
예들 들어 counseling attendence 가 들어가야 하는데 counseling attendenc yes 로 전처리가 된 데이터가 들어감.
현재 x_trian으로 확인 후 변경
Value error 발생.

# 연속 변수 수치 변수 나누기

# 열단위 전처리 객체로 전처리 묶음
preprocessor = ColumnTransformer(
    transformers = [('num_data', StandardScaler(), num_features), # 표준화 사용
                   ('cat_data', OneHotEncoder(), cat_features) # 원핫인코더 사용
                   ]
)

pipe = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', rf) # 전처리 랑 훈련 묶음, 파이프 라인 형성
])

#RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator = pipe, # 파이프 라인( 전처리, 학습모델)
    param_distributions = dist, # 실험할 파라미터 세트 => 위에서 만들어 놓은 것
    n_iter = 30, => 그 중 30개만 조합
    cv = 5, => 5-교차검증
    scoring = 'r2', => r2 점수 기준 판단
    n_jobs = -1,
    random_state = 42
)
x = depression_data
y = depression_data['Mental Stress Level'] # 인코딩 안된 기본 데이터 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
random_search.fit(x_train, y_train) # 하이퍼 파라미터 찾기

best_model = random_search.best_estimator_ # 가장 좋은 조합

y_pred_train = best_model.predict(x_train) # 최적의 조합으로 찾은 훈련 예측값
y_pred_test = best_model.predict(x_test) # 최적의 조합으로 찾은 실제 예측값

print("Best Parameters:", random_search.best_params_) # 최적 파마미터 호출 best._params
```
Best Parameters: {'regressor__n_estimators': 60, 'regressor__min_samples_split': 8, 'regressor__max_depth': 15}



5겹 교차 검증 : 학습데이터를 5개로 나누어서 너무 훈련데이터에만 과적합 되지 않게 중간에 test 데이터를 섞어 5번 돌려가는 것이다. 
R2 점수 : 결정 계수, 실제 y값 얼마나 잘 설명 1 - (잔차 제곱합 /  전체 제곱합), 회귀에서 사용. 1일 수록 잘 예측
모델이 예측한 오차 제곱합인 분자가 0이 될 수록 좋은 것. 비선형에서는 잘 안씀.

```
# 데이터 준비 및 학습
from sklearn.preprocessing import OneHotEncoder #인코더 
from sklearn.compose import ColumnTransformer # 열 전처리
from sklearn.pipeline import Pipeline # 파이프 라인 구축
from sklearn.ensemble import RandomForestRegressor # 모델
from sklearn.model_selection import train_test_split # 데이터 분리
from sklearn.metrics import mean_squared_error # 편군 제곱 오차
from sklearn.preprocessing import StandardScaler # 스케일러
from math import sqrt  # 루트 => MSE에 루트. RMSE가 이상치 민감도 적음.
```
```
# 원본 손상 방지 .copy()
depression_encoded_copy = depression_encoded.copy()

x = depression_data_copy.drop(columns = 'Mental Stress Level')
y = depression_data_copy['Mental Stress Level']

# 번주형 변수
cat_features = ['Counseling Attendance','Stress Coping Mechanisms','Family Mental Health History','Medical Condition','Gender']

# 수치형 변수
num_features = [col for col in x.columns if col not in cat_features]

# 변수 전처리 단계
propressore = ColumnTransformer(
    transformers = [('numbers', StandardScaler(), num_features),
                    ('categories', OneHotEncoder(), cat_features)
    ]
)

# 모델 생성 - 최적화
rf = RandomForestRegressor(n_estimators=60,
                           max_depth=8,
                           min_samples_split=15,
                           random_state=42)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', rf)
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
random_search.fit(x_train, y_train)

model = pipe.fit(x_train, y_train) # 학습

# 각 데이터 예측
y_pred_train = pipe.predict(x_train)
y_pred_test = pip.predict(x_test)

# 평가 : 평균 제곱 오차
rmse_train = sqrt(mean_sqared_error(y_train, y_pred_train))
rmse_test = surt(mean_squred_srror(y_test, y_pred_test))

print("Train RMSE:", rmse_train)
print("Test RMSE:" ,rmse_test)
```
```
Train RMSE: 2.077201398340799
Test RMSE: 3.029195612207572

# 종속변수 다시 지수 변환
print("Train RMSE:", np.exp(rmse_train))
print("Test RMSE:" ,np.exp(rmse_test))
```
Train RMSE: 7.982098911276037
Test RMSE: 20.680590682328326

# 모델 성능확인
print(model.score(x_train, y_train)) # 예측값과 결과값 일치하는 정도
print(model.score(x_test, y_test))
```
0.48394799761506246
-0.06942547428034063

일단 오차값이 거의 20로 너무 크고, 훈련데이터와의 차이가 너무 발생
test 데이터에 대한 점수가 음수임
=> 훈련 데이터에 과적합 가능, 일반화 성능 낮음

내가 보기엔 y_data 로그변환으로 오히려 데이터가 불균형해져서 오히려 안하는 게 나을 듯.

+ 평균 제곱 오차 값이 크면 모델이 문제일 수도 있음


# 특징 중요도 측정
```
feature_importances = rf.feature_importances_

features = x.columns

importance_df = pd.DataFrame({'Feature' : features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False) # 중요도 기준 내립차순

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance of Each Factor on stress')
plt.gca().invert_yaxis()
plt.show()
```
![download](https://github.com/user-attachments/assets/24bff605-a3d1-45c4-9ef4-9d33b648c6de)
```
왜 과적합이 나올까?
1. 트리 단순화 : 깊이를 줄이거나, 최종 산물 수 늘리기
2. 원 핫 인코딩으로 인한 변수 너무 많음 => lebel encoding 고려, targetincoding 고려
3. 로그 변환 취소.
4. 변수 중요도에서 중요도 0인 변수 삭제 (노이즈 제거)
5. stratify : test, train 비융 고려해 균형있게

변경
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state = 42)
model = pipe_.fit(x_train, y_train)
```
변경 후 RMSE 
Train RMSE: 2.1094467641154595
Test RMSE: 2.9323860998678897

Train RMSE: 8.243679323600286
Test RMSE: 18.77236984838066

조금 성능이 향상되었다.
트리깊이와 나무의 개수는 별 차이가 없다. 

라벨 인코딩으로 변환해주어도 score 결과는 그대로이다. 
라벨 인코딩 : 원-핫 인코딩과는 다르게 0과 1이 아닌 0~n-1의 수치 변환을 지원해 특징의 수를 더 줄일 수 있다.
다만 한 줄에 대해서만 지원하니, ColumnTransfer나 pipe라인 안에는 넣지 않고ㅗ, 범주형 변수와 for문을 사용해 따로 df을 바꿔주야 한다.
```
cat_features = ['Counseling Attendance','Stress Coping Mechanisms','Family Mental Health History','Medical Condition','Gender']
label_encoders = {}
for col in cat_features:
  le = LabelEncoder()
  depression_data_copy[col] = le.fit_transform(depression_data[col])
  label_encoders[col] = le
print(pd.DataFrame(depression_data_copy))

x = depression_data_copy.drop(columns = 'Mental Stress Level')
y = depression_data_copy['Mental Stress Level']

# 그리고 전처리 순서도 유의하자. 라벨 인코딩을 사용할 경우, x와 y는 전처리 이후에 만들어 줘야 한다.
모델도 파이프라인이 아니라 모델 피팅으로 바꿈

0.2593515880863535
-0.015073660764981511
그래도 음수값이 줄어들긴 했다.
다음에는 다른 모델로 해봐야겠다. 




