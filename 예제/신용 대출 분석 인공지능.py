# 1 데이터 전처리에 필요한 패키지 데이터를 불러오고 학습 데이터 읽기
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

loan_data = pd.read_csv('loanData.csv')
loan_data.tail()

# 2 특정 칼럽의 값 중 중복을 제외한(unique)값 출력
loan_data['dependents'].unique()

# 3 데이터의 성별 칼럼을 정수형 데이터로 매핑
gendr_mapping = {'Male' : 1, 'Female' : 0, np.nan : -1}
loan_data = loan_data.replace({'gender' : gendr_mapping})

# 결과 일부 확인하기
loan_data.head()

# 4 칼럽 값을 실수형 데이터로 변환
gendr_mapping = {'Male' : 1, 'Female' : 0, np.nan : -1}
married_mapping = {'No':0, 'Yes':1, np.nan: -1}
dep_mapping = {'0':0, '1':1, '2':2, '3+':3, np.nan: -1}
edu_mapping = {'Graduate':1, 'Not Graduate':0}
emp_mapping = {'No':0, 'Yes':1, np.nan: -1}
prop_mapping = {'Urban':1, 'Rural':3, 'Semiurban': 3}

loan_data = loan_data.replace({'married': married_mapping,
                               'dependents':dep_mapping,
                               'education':edu_mapping,
                               'self_employed':emp_mapping,
                               'property_area':prop_mapping})

# 결과 일부 확인.
loan_data.head()

# 5 데이터를 학습용과 테스트용으로 나누기
from sklearn.model_selection import train_test_split

# id 제외
X = loan_data.loc[:, 'gender':'loan_term']
y = loan_data.loc[:, 'loan_status']

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=20)

# 6 xgboost로 신용 대출 여부를 판별하는 모델 구축
model = XGBClassifier(booster='gbtree',
                      objective='binary:logistic')

model.fit(x_train, y_train)

# 7 모델과 테스트 데이터로 정확도 측정
from sklearn.metrics import accuracy_score

def calculate_accuracy(model, x_test, y_test):
    # 예측하기
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]

    # 평가하기
    accuracy = accuracy_score(y_test, predictions)
    print('Accuracy: %.2f%%' % (accuracy * 100.0))
    return accuracy

calculate_accuracy(model, x_test, y_test)

# 8 피처 중요도 출력
import xgboost
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10,5
xgboost.plot_importance(model)

# 9 SHAP 기법을 사용해서 사용자 한 명을 분석
import shap
idx = 13

# 13번째 사용자 데이터 출력
print(x_train.iloc[idx, :])

# JS 시각화 라이브러리 로드
shap.initjs()

# SHAP 값으로 모델의 예측 결과 설명
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)

# 설명체의 해석 결과 출력
shap.force_plot(explainer.expected_value,
                shap_values[idx,:],
                x_train.iloc[idx,:])

# 10 모든 학습 데이터에 대해 설명체 전체를 플롯으로 보이기
shap.initjs()

# 모뎅이 학습한 결과에 대한 설명체 전체 출력
shap.force_plot(explainer.expected_value, shap_values, x_train)

# 11 신용 등급별로 대출 승인 여부 영향력 출력
# 신용등급 하나의 피처에 대해 SHAP 영향력 출력
shap.dependence_plot("credit_rate", shap_values, x_train)

# 12 전체 학습 세트에 대해 섀플리 값 출력
# 모든 피처ㅔ 대해 모델에 미치는 영향력 출력
shap.summary_plot(shap_values, x_train)

# 13 막대(bar)타입으로 총괄 플롯 출력
shap.summary_plot(shap_values, x_train, plot_type="bar")


