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
model = XGBClassifier()
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# ----------------------------------------------------------------------------------------------------------------------
# 1 최적 하이퍼파라미터 찾는 코드
import numpy as np
from sklearn.model_selection import GridSearchCV

cv_params = {'max_depth': np.arange(1, 6, 1)}
fix_params = {'booster':'gbtree',
              'objective':'binary:logistic'}

csv = GridSearchCV(XGBClassifier(**fix_params),
                   cv_params,
                   scoring= 'precision',
                   cv= 5,
                   n_jobs=5)
csv.fit(x_train, y_train)
print(csv.best_params_)

# 테스트 데이터 예측하기
y_pred = csv.predict(x_test)
predictions = [round(value) for value in y_pred]

# 정확도 평가
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" %(accuracy * 100.0))
# ----------------------------------------------------------------------------------------------------------------------
# # 2 파라미터 값 변화에 대한 최적의 모델 찾기          (pc 사양에 따라 시간이 오래걸림)
# import numpy as np
# from sklearn.model_selection import GridSearchCV
#
# cv_params = {'max_depth': np.arange(1, 6, 1),
#              'learning_rate': np.arange(0.05, 0.6, 0.05),
#              'n_estimators': np.arange(50, 300, 50)}
#
# fix_params = {'booster': 'gbtree',
#               'objective': 'binary:logistic'}
#
# csv = GridSearchCV(XGBClassifier(**fix_params),
#                    cv_params,
#                    scoring='precision',
#                    cv=5,
#                    n_jobs=5)
#
# csv.fit(x_train, y_train)
#
# # 최적의 파라미터 출력
# print(csv.best_params_)
#
# # 테스트 데이터 예측
# y_pred = csv.predict(x_test)
# predictions = [round(value) for value in y_pred]
#
# #성능 평가
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
#
# # GridSearch를 사용한 모든 조합 출력
# for parameter in csv.cv_results_["params"]:
#     print(parameter)
# ----------------------------------------------------------------------------------------------------------------------

# 3 당뇨병 진단 모델의 최적 파라미터
model = XGBClassifier(booster='gbtree',
                      objective='binary:logistic',
                      learning_rate=0.03,
                      n_estimators=150,
                      reg_alpha=0.15,
                      reg_lambda=0.7,
                      max_depth=4)
# ----------------------------------------------------------------------------------------------------------------------

# 4 컨퓨전 행렬 계산
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

import itertools

def plot_confusion_matrix(cm,
                          classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment = "center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def show_data(cm, print_res = 0):
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]
    if print_res == 1:
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR)  = {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3f}'.format(fp/(fp+tn)))
    return tp/(tp+fp), tp/(tp+fn), fp/(fp-tn)

plot_confusion_matrix(cm, ['0', '1'],)
show_data(cm, print_res=1)
