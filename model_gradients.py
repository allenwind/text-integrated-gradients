import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from pooling import MaskGlobalMaxPooling1D, MaskGlobalAveragePooling1D
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_THUCNews_title_label
from dataset import load_weibo_senti_100k
from dataset import load_simplifyweibo_4_moods
from dataset import load_hotel_comment

# 纯梯度方法

X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=77236)

num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(X_train)
X_train = tokenizer.transform(X_train)

maxlen = 48
maxlen = find_best_maxlen(X_train)

X_train = sequence.pad_sequences(
    X_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0.0
)
y_train = tf.keras.utils.to_categorical(y_train)

num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(num_words, embedding_dims,
    embeddings_initializer="normal",
    input_length=maxlen,
    mask_zero=True)(inputs)
x = Dropout(0.2)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)
x, _ = MaskGlobalMaxPooling1D()(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.2)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]
)
model.summary()

batch_size = 32
epochs = 5
callbacks = []
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_split=0.2
)

class GradientLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GradientLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        output, embedding, label = inputs
        output = tf.gather(output, label, batch_dims=1)
        grads = tf.gradients(output, [embedding])[0] * embedding
        return grads

label_input = Input(shape=(1,), dtype=tf.int32)
inputs = model.inputs + [label_input]

embedding_input = model.layers[1].output
grads = GradientLayer()([model.output, embedding_input, label_input])

gradient_model = Model(inputs, grads)

embeddings = model.layers[1].embeddings # embedding矩阵
values = tf.Variable(embeddings) # 保存embedding矩阵以便恢复

def compute_weights(x, n=25, scale=True):
    # 模型预测
    y_pred = model.predict(x)[0]
    y_pred_id = np.argmax(y_pred)
    y_pred_in = np.array([y_pred_id])
    grads = gradient_model.predict([x, y_pred_in])[0]
    grads = np.array(grads)
    weights = np.sqrt(np.square(grads).sum(axis=1))
    if scale:
        weights = (weights - weights.min()) / (weights.max() - weights.min())
    return weights

id_to_classes = {j:i for i,j in classes.items()}
from textcolor import print_color_text
def visualization():
    for sample, label in zip(X_test, y_test):
        sample_len = len(sample)
        if sample_len > maxlen:
            sample_len = maxlen

        x = np.array(tokenizer.transform([sample]))
        x = sequence.pad_sequences(
            x, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        y_pred = model.predict(x)[0]
        y_pred_id = np.argmax(y_pred)
        # 预测错误的样本跳过
        if y_pred_id != label:
            continue
            
        # 预测权重
        weights = compute_weights(x)[:sample_len]
        print_color_text(sample, weights)
        print(" =>", id_to_classes[y_pred_id])
        input() # 按回车预测下一个样本

visualization()

