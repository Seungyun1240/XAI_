# 1 데이터셋을 학습용과 테스트용으로 분리
import shap
from sklearn.model_selection import train_test_split

X,y = shap.datasets.boston()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)

print(X_train[:10])

# ----------------------------------------------------------------------------------------------------------------------
# 2 방의 개수와 집값 간의 관계를 산점도로 그리기
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig, ax1 = plt.subplots(1, 1, figsize= (12,6))

ax1.scatter(X['RM'], y, color='black', alpha=0.6)

ax1.set_title('Relation # of Rooms with MEDV')
ax1.set_xlim(2.5, 9)
ax1.set_xlabel('RM')
ax1.set_ylim(0, 55)
ax1.set_ylabel('MEDV \n Price $1,000')

# ----------------------------------------------------------------------------------------------------------------------
# 3 선형 모델을 이용해서 방 개수와 주택 가격 간의 관계 구하기
from sklearn import linear_model
import pandas as pd

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X=pd.DataFrame(X_train['RM']), y=y_train)
prediction = linear_regression.predict(X=pd.DataFrame(X_test['RM']))

print('a value: ', linear_regression.intercept_)
print('b value: ', linear_regression.coef_)
print('MEDV = {:.2f} * RM {:.2f}'.format(linear_regression.coef_[0],
                                         linear_regression.intercept_))

# ----------------------------------------------------------------------------------------------------------------------
# 4 방의 개수가 달라질 때 주택 매매 가격을 예측하는 그래프와 데이터를 한꺼번에 플롯으로 그리기
# 학습, 테스트 데이터를 산점도로 그리고 직선의 방정식 표시
fig, ax1 = plt.subplots(1, 1, figsize = (12, 6))

ax1.scatter(X_train['RM'], y_train, color='black',
            alpha=0.4, label='data')
ax1.scatter(X_test['RM'], y_test, color='#993299',
            alpha=0.6, label='data')

ax1.set_title('Relation # of Rooms with MEDV')
ax1.set_xlim(2.5, 9)
ax1.set_xlabel('RM')
ax1.set_ylim(0, 55)
ax1.set_ylabel('MEDV \n Price $1,000')

ax1.plot(X_test['RM'], prediction, color='purple', alpha=1,
         linestyle='--', label='linear regression line')

ax1.legend()

# ----------------------------------------------------------------------------------------------------------------------
# 5 모델 예측치와 실제 집값 간의 RMSE 구하기
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, prediction, squared=False))

print("RMSE: %f" %(rmse))

# ----------------------------------------------------------------------------------------------------------------------
# 6 xgboost의 선형 회귀 모델로 주택 매매 가격을 예측하는 모델을 만들고 학습
import xgboost

# XGBoost 모델 학습하기
model = xgboost.XGBRegressor(objective='reg:linear')
model.fit(X_train, y_train)

preds = model.predict(X_test)

# 7 전체 피처를 사용해서 학습시킨 모델의 RMSE 구하기
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" %(rmse))

# 8 SHAP의 설명체 정의 및 섀플리 값 계산
# JS 시각화 라이브러리 로드
shap.initjs()

# SHAP값으로 모델의 예측 설명
# 설명체는 LightGBM, CatBoost, scikit-learn 모델을 입력받을 수 있음.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 9 첫 번째 데이터에 대한 구체적 SHAP 값 시각화
shap.force_plot(explainer.expected_value,
                shap_values[0,:],
                X_train.iloc[0,:])

# 10 259번 데이터에 대해서 방의 개수(RM)와 집 가격(MEDV)이 어떤 관계가 있는지 플롯
fig, ax1 = plt.subplots(1, 1, figsize= (12,6))
idx = 259
ax1.scatter(X['RM'], y, color='black', alpha=0.6)
ax1.scatter(X_train['RM'].iloc[idx], y_train[idx], c='red', s=150)
ax1.set_title('Relation # of Rooms with MEDV')
ax1.set_xlim(2.5, 9)
ax1.set_xlabel('RM')
ax1.set_ylim(0, 55)
ax1.set_ylabel('MEDV \n Price $1,000')

# 11 데이터 259번에 대한 섀플리 영향도
shap.force_plot(explainer.expected_value,
                shap_values[259,:],
                X_train.iloc[259,:])

# 12 전체 데이터에 대한 섀플리 값을 플롯으로 그리기
# 모델이 학습 데이터를 예측한 결과에 대해 SHAP 분석한 결과 출력
shap.force_plot(explainer.expected_value, shap_values, X_train)

# 13 방 개수 피처가 집값에 미치는 섀플리 영향도 시각화 플롯
# 하나의 피처가 전체 예측에 미치는 영향력을 SHAP로 계산하고 출력
shap.dependence_plot("RM", shap_values, X_train)

# 14 전체 피처들이 섀플리 값 결정에 어떻게 관여하는지 시각화
# 모든 피처에 대해 SHAP값을 계산하고, 영향력 시각화
shap.summary_plot(shap_values, X_train)

# 15 피처별 섀플리 값을 막대 타입으로 비교
shap.summary_plot(shap_values, X_train, plot_type="bar")

# 16 xgboost의 피처 중요도 호출
xgboost.plot_importance(model)
