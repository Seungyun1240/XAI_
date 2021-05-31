# 1 sklearn 패키지의 20 news groups 데이터셋 가져오기
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# 클래스 이름 줄이기
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:])
               for x in newsgroups_train.target_names]

print(class_names)

class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

print(class_names)
# ----------------------------------------------------------------------------------------------------------------------

# 2 제보 기사의 카테고리 분류 모델 생성 및 F1 점수 측정
import sklearn
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB

# TF-IDF 사용하여 문서를 숫자 벡터로 변환하는 전처리 과정
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# 학습하기
nb = MultinomialNB(alpha= .01)
nb.fit(train_vectors, newsgroups_train.target)

# 테스트하기
pred = nb.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')
print(sklearn.metrics.f1_score)

# ----------------------------------------------------------------------------------------------------------------------

# 3 파이프라인 기술 사용
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vectorizer, nb)
predict_classes = pipe.predict_proba([newsgroups_test.data[0]]).round(3)[0]

print(predict_classes)

# ----------------------------------------------------------------------------------------------------------------------

# 4 데이터 분류 결과의 가독성 높이기 위해 출력 수정
rank = sorted(range(len(predict_classes)),
              key=lambda i : predict_classes[i],
              reverse= True)
for rank_index in rank:
    print('[{:>5}]\t{:<3}\t class ({:.1%})'.format(rank.index(rank_index)+1,
                                                   rank_index, predict_classes[rank_index]))
# ----------------------------------------------------------------------------------------------------------------------

# 5 LIME 텍스트 설명체 선언
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

explainer = LimeTextExplainer(class_names=class_names)

# explain_instance 메서드에 필요한 최소한의 파라미터
exp = explainer.explain_instance(newsgroups_test.data[0],
                                 pipe.predict_proba,
                                 top_labels=1)

# LIME이 잘 작동하는지 확인
exp.available_labels()
exp.show_in_notebook(text=newsgroups_test.data[0])

# ----------------------------------------------------------------------------------------------------------------------

# 6 테스트 데이터 5번 LIME 알고리즘에 입력
from lime.lime_text import LimeTextExplainer

idx = 5

explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(newsgroups_test.data[idx],
                                 pipe.predict_proba,
                                 top_labels=1)

predict_classes = pipe.predict_proba([newsgroups_test.data[idx]]).round(3)[0]
rank = sorted(range(len(predict_classes)),
              key=lambda i: predict_classes[i], reverse=True)

print('Document id: %d' % idx)
print('Predicted class: %s' %
      class_names[nb.predict(test_vectors[idx]).reshape(1, -1)[0, 0]])
print('True class: %s' % class_names[newsgroups_test.target[idx]])
print(predict_classes)
print(rank)

print('Explanation for class %s' % class_names[rank[0]])
print('\n'.join(map(str, exp.as_list(rank[0]))))

exp.show_in_notebook(text=newsgroups_test.data[idx])
