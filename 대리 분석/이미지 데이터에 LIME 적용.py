# 1 올리베티 얼굴 데이터 로드 및 확인
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage.util import montage

from sklearn.datasets import fetch_olivetti_faces


faces = fetch_olivetti_faces()

# 이미지 흑백으로 만들고 LIME이 처리할 수 있는 형태로 변환
X_vec = np.stack([gray2rgb(iimg)
                  for iimg in faces.data.reshape((-1, 64, 64))], 0)
y_vec = faces.target.astype(np.uint8)

fig, ax1 = plt.subplots(1, 1 ,figsize = (8, 8))
ax1.imshow(montage(X_vec[:,:,:,0]),
           cmap='gray', interpolation='none')
ax1.set_title('All Faces')
ax1.axis('off')

# 2 이미지 데이터 한 장을 그리는 코드
index = 93
plt.imshow(X_vec[index], cmap='gray')
plt.title('{} index face'.format(index))
plt.axis('off')

# 3 텐서플로를 이용한 분류 모델을 LIME에서 사용할 수 있게 컨벤션 맞추기
def predict_proba(image):
    return session.run(model_predict,
                       feed_dict={preprocessed_image : image})

# 4 X_vec과 y_vec으로부터 학습용과 테스트용 데이터셋 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec,
                                                    y_vec,
                                                    train_size=0.70)

# 5 MLP가 학습할 수 있게 이미지 전처리를 수행하는 파이프라인 생성
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

class PipeStep(object):
    """
        Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func = step_func

    def fit(self, *args):
        return self
    def transform(self, X):
        return self._step_func(X)

makegray_step = PipeStep(lambda img_list:
                         [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list:
                        [img.ravel() for img in img_list])

simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('MLP', MLPClassifier(
        activation='relu',
        hidden_layer_sizes=(400, 40),
        random_state=1))
])

# 학습 데이터를 MLP가 있는 파이프라인에 입력
simple_pipeline.fit(X_train, y_train)

# 6 모델 성능 테스트
pipe_pred_test = simple_pipeline.predict(X_test)
pipe_pred_prop = simple_pipeline.predict_proba(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=pipe_pred_test))

# 7 전처리 과정 추가하여 MLP 학습
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier

class PipeStep(object):
    """
        Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func = step_func
    def fit(self, *args):
        return self
    def transform(self, X):
        return self._step_func(X)

makegray_step = PipeStep(lambda img_list:
                         [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list:
                        [img.ravel() for img in img_list])
simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    ('MLP', MLPClassifier(
        activation='relu',
        hidden_layer_sizes=(400, 40),
        random_state=1))
])

simple_pipeline.fit(X_train, y_train)

# 8 최적의 파이프라인 조합
simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    ('MLP', MLPClassifier(
        activation='relu',
        alpha=1e-7,
        epsilon=1e-6,
        hidden_layer_sizes=(800, 120),
        random_state=1))
])

# 9 XAI 적용 - LIME의 이미지 설명체와 이미지 분할 알고리즘 선언
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

explainer = lime_image.LimeImageExplainer()

# 이미지 분할 알고리즘 : quickshift(기본), slic, felzenszwalb
segmenter = SegmentationAlgorithm('slic',
                                  n_segments = 100,
                                  compactness = 1,
                                  sigma = 1)

# 10 테스트 0번 이미지에 대해 설명 모델 구축
olivetti_test_index = 0

exp = explainer.explain_instance(X_test[olivetti_test_index],
                                 classifier_fn=simple_pipeline.predict_proba,
                                 top_labels=6,
                                 num_samples=1000,
                                 segmentation_fn=segmenter)

# 11 올리베티 데이터 0번을 설명체에 통과시켜 XAI 수행
from skimage.color import label2rgb

# 캔버스 설정
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

# 예측에 도움이 되는 세그먼트 출력
temp, mask = exp.get_image_and_mask(y_test[olivetti_test_index],
                                    positive_only=True,
                                    num_features=8,
                                    hide_rest=False)
ax1.imshow(label2rgb(mask, temp, bg_label=0),
           interpolation='nearest')
ax1.set_title('Positive Regions for {}'.format(y_test[olivetti_test_index]))

# 모든 세그먼트 출력
temp, mask = exp.get_image_and_mask(y_test[olivetti_test_index],
                                    positive_only=False,
                                    num_features=8,
                                    hide_rest=False)
ax2.imshow(label2rgb(4-mask, temp, bg_label=0),
           interpolation='nearest')
ax2.set_title('Positive/Negative Regions for {}'.format(y_test[olivetti_test_index]))

# 이미지만 출력
ax3.imshow(temp, interpolation='nearest')
ax3.set_title('Show output image only')

# 마스크만 출력
ax4.imshow(mask, interpolation='nearest')
ax4.set_title('Show mask only')

# 올리베티 얼굴 테스트 데이터 0번(3번 인물)으로부터 추가 설명 출력
# 하나의 인물에 대한 부가 설명 출력
olivetti_test_index = 1

fig, m_axs = plt.subplots(2,6, figsize=(12, 4))
for i, (c_ax, gt_ax) in zip(exp.top_labels, m_axs.T):
    temp, mask = exp.get_image_and_mask(i,
                                        positive_only=True,
                                        num_features=12,
                                        hide_rest=False,
                                        min_weight=0.001)
    c_ax.imshow(label2rgb(mask, temp, bg_label=0),
                interpolation='nearest')
    c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(i,
                                                           100*pipe_pred_prop[olivetti_test_index,i]))
    c_ax.axis('off')

    face_id = np.random.choice(np.where(y_train==i)[0])

    gt_ax.imshow(X_train[face_id])
    gt_ax.set_title('Example of {}'.format(i))
    gt_ax.axis('off')
