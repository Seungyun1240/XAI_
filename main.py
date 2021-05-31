from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
features = dataset.data
label = dataset.target

# 학습은 80%, 테스트는 20%로 분할
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2,
                                                    random_state=156)
# n_stimators는 400으로 설정하고, 조기 중단 수행
lgbm_wrapper = LGBMClassifier(n_estimators=400)
evals = [(x_test, y_test)]
lgbm_wrapper.fit(x_train, y_train, early_stopping_rounds=100, eval_metric='logloss',
                 eval_set=evals, verbose=True)
pred = lgbm_wrapper.predict(x_test)
pred_proba = lgbm_wrapper.predict_proba(x_test)[:,1]

# 시각화
from lightgbm import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 11))
plot_importance(lgbm_wrapper, ax=ax)

plt.show()