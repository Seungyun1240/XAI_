# 1 당뇨병 진단 모델 구현
from numpy import loadtxt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 데이터 로드
dataset = loadtxt('diabetes.csv', delimiter=',')
# X와 Y로 데이터 분리
X = dataset[:, 0:8]
y = dataset[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# 학습 데이터로 모델 학습시키기
model = XGBClassifier(max_depth=3.




                      )
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# 평가하기
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.2f%%' %(accuracy * 100.0))

# ----------------------------------------------------------------------------------------------------------------------
# 2 모델의 의사 결정 트리 시각화
import os
os.environ['PATH'] += (os.pathsep + 'C:Program Files (x86)/Graphviz2.38/bin/')

from xgboost import plot_tree
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 100,200

#의사 결정 트리 시각화
plot_tree(model)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
#
# # 3 피처 중요도 계산 및 시각화
# from xgboost import plot_importance
#
# rcParams['figure.figsize'] = 10, 10
# plot_importance(model)              # 중요변수 시각화
#
# plt.yticks(fontsize = 15)
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
#
# # # 4 GTT피처에 대한 목표 플롯
# from pdpbox import info_plots
# import pandas as pd
#
# dataset = pd.read_csv('diabetes1.csv', delimiter=',')
# pima_data = dataset
# pima_features = dataset.columns[:8]
# pima_target = dataset.columns[8]
# fig, axes, summary_df = info_plots.target_plot(df = pima_data,
#                                                feature = 'BloodPressure',         # feature 바꾸면 원하는 그래프 볼 수 있음
#                                                feature_name = 'BloodPressure',
#                                                target = pima_target)
# plt.show()
# print(summary_df)    # 그래프 결과를 pandas의 DataFrame 포맷으로 보여줌
# # ----------------------------------------------------------------------------------------------------------------------
#
# # 5 GTT 데이터에 대한 모델의 실제 예측 분포 플롯
# from pdpbox import info_plots
# import pandas as pd
#
# dataset = pd.read_csv('diabetes1.csv', delimiter=',')
# pima_data = dataset
# pima_features = dataset.columns[:8]
# pima_target = dataset.columns[8]
# fig, axes, summary_df = info_plots.actual_plot(model= model,
#                                                X= pima_data[pima_features],
#                                                feature= 'Glucose',
#                                                feature_name='Glucose',
#                                                predict_kwds={})
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
#
# # 6 GTT 테스트 피처에 대해 부분 의존성 계산 및 플롯 그리기
# from pdpbox import info_plots, pdp
# import pandas as pd
#
# dataset = pd.read_csv('diabetes1.csv', delimiter=',')
# pima_data = dataset
# pima_features = dataset.columns[:8]
# pima_target = dataset.columns[8]
# pdp_gc = pdp.pdp_isolate(model= model,
#                          dataset = pima_data,
#                          model_features = pima_features,
#                          feature = 'Glucose')
#
# # 플롯 정보 설정
# fig, axes = pdp.pdp_plot(pdp_gc,
#                          'Glucose',
#                          plot_lines = False,        # plot_lines = False or True
#                          frac_to_plot = 0.5,
#                          plot_pts_dist = True)
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
#
# # 7 혈압과 GTT 테스트 데이터 두 피처에 대한 목표 플롯
# from pdpbox import info_plots
# import pandas as pd
#
# dataset = pd.read_csv('diabetes1.csv', delimiter=',')
# pima_data = dataset
# pima_features = dataset.columns[:8]
# pima_target = dataset.columns[8]
# fig, axes, summary_df = info_plots.target_plot_interact(df = pima_data,
#                                                         features=['BloodPressure', 'Glucose'],
#                                                         feature_names=['BloodPressure', 'Glucose'],
#                                                         target=pima_target)
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
#
# # 8 혈압과 GTT 테스트 데이터로 모델에 대한 부분 의존성 플롯
# from pdpbox import info_plots, pdp
# import pandas as pd
#
# dataset = pd.read_csv('diabetes1.csv', delimiter=',')
# pima_data = dataset
# pima_features = dataset.columns[:8]
# pima_target = dataset.columns[8]
# pdp_interaction = pdp.pdp_interact(model= model,
#                                    dataset=pima_data,
#                                    model_features=pima_features,
#                                    features=['BloodPressure', 'Glucose'])
#
# fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_interaction,
#                                   feature_names=['BloodPressure', 'Glucose'],
#                                   plot_type='contour',   # contour : 컨투어(동고선)차트, grid : 그리트차트
#                                   x_quantile=True,
#                                   plot_pdp=True)
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
#
# # 9 혈압 피처에 대한 부분 의존성 플롯
# from pdpbox import info_plots, pdp
# import pandas as pd
#
# dataset = pd.read_csv('diabetes1.csv', delimiter=',')
# pima_data = dataset
# pima_features = dataset.columns[:8]
# pima_target = dataset.columns[8]
#
# # 혈압 정보 계산
# pdp_bp = pdp.pdp_isolate(model= model,
#                          dataset=pima_data,
#                          model_features=pima_features,
#                          feature='BloodPressure')
#
# # 혈압에 대한 PDP 정보를 플롯
# fig, axes = pdp.pdp_plot(pdp_bp,
#                          'BloodPressure',
#                          plot_lines=False,
#                          frac_to_plot=0.5,
#                          plot_pts_dist=True)
# plt.show()