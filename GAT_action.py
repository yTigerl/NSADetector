from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Activation, Conv2D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from spektral.layers.pooling import GlobalAvgPool
import tensorflow as tf
from tensorflow import keras
# from DataLoader import load_gnnsandwichTrainDataset
from gnnDataLoader import load_gnnsandwichTrainDataset
from spektral.layers import GATConv, GCNConv, EdgeConv, ECCConv
from spektral.data import Graph, Dataset, BatchLoader
from spektral.transforms import AdjToSpTensor

epochs = 10
# bundle_types = ["CSA", "MLS", "LF", "Non"]
feature_file = "./data/gcn train data_4_class_20000.txt"
label_file = "./data/gcn train label_4_class_20000.txt"
# bundle_types = ["CSA", "MLS", "LF", "MPS", "Non"]
# feature_file = "./data/gcn train data_5_class_20000.txt"
# label_file = "./data/gcn train label_5_class_20000.txt"
# feature_file = "./data/gcn fine-turn data_5_class_20000_30000 MBS.txt"
# label_file = "./data/gcn fine-turn label_5_class_20000_30000 MBS.txt"
# feature_file = "./data/gnn_action_node data 17500499to17600499.txt"
# label_file = "./data/gnn_action_node target 17500499to17600499.txt"
# feature_file = "./data/gnn_action_node data without non 17500499to17600499.txt"
# label_file = "./data/gnn_action_node target without non 17500499to17600499.txt"


num_classes = 4
classes = [0, 1, 2, 3]
# num_classes = 4
# classes = [0, 1, 2, 4]
# num_classes = 3
# classes = [0, 1, 3]

# num_classes = 5
# classes = [0, 1, 2, 3, 4]
token_embedding_dim = 1
address_embedding_dim = 1
# pretrained
initial_path = "./MODEL/2GAT 64_128 1FC 512 4 class graph2.ckpt"
checkpoint_path = "./MODEL/2GAT 64_128 1FC 512 4 class graph2.ckpt"
model_file = './MODEL/2GAT 64_128 1FC 512 data_4_class_20000 graph2.h5'
model_load_file = './MODEL/2GAT 64_128 1FC 512 data_4_class_20000 graph2.h5'
initial_path = "./MODEL/2GAT 32_64 1FC 64 4 class.ckpt"
checkpoint_path = "./MODEL/2GAT 32_64 1FC 64 4 class.ckpt"
model_file = './MODEL/2GAT 32_64 1FC 64 data_4_class_20000 graph2.h5'
model_load_file = './MODEL/2GAT 32_64 1FC 64 data_4_class_20000 graph2.h5'
# model_load_file = './MODEL/2GAT 128_128 1FC 128 data_5_class_20000_20100 LPM .h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate, Flatten
from spektral.layers import GCNConv
from spektral.layers import GlobalSumPool


# 基于GCN的模型


def create_gat_model(input_shapes, output_units, address_embedding_dim=1):
    # 输入层
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')

    # 第一层 GATConv（多头）
    X_2 = GATConv(32, attn_heads=4, concat_heads=True, activation='elu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)

    # 第二层 GATConv（通常设置为 heads=1, concat=False）
    # X_2 = GATConv(64, attn_heads=4, concat_heads=False, activation='elu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)

    X_2 = GATConv(64, attn_heads=1, concat_heads=False, activation='elu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)

    # 全局池化 + FC
    X_2 = GlobalAveragePooling1D()(X_2)
    X_2 = Dense(64, activation='relu')(X_2)

    return Model(inputs=[X_in, A_in], outputs=X_2, name="base_gat")

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# def euclidean_distance(vectors):
#     x, y = vectors
#     return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # return Activation('sigmoid')(K.sqrt(K.maximum(sum_square, K.epsilon())))
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# 孪生网络模型
def create_siamese_model(input_shapes, output_units):
    base_model = create_gat_model(input_shapes, output_units)

    X_in1 = Input(shape=(None, address_embedding_dim * 3 + 5), name='X_in1')
    A_in1 = Input(shape=(None, None), name='A_in1')
    E_in1 = Input(shape=(None, address_embedding_dim * 2 + 5),
                 name='E_in1')  # 两个地址嵌入 + 操作类型的one-hot编码 + amount0 + amount1

    X_in2 = Input(shape=(None, address_embedding_dim * 3 + 5), name='X_in2')
    A_in2 = Input(shape=(None, None), name='A_in2')
    E_in2 = Input(shape=(None, address_embedding_dim * 2 + 5),
                 name='E_in2')  # 两个地址嵌入 + 操作类型的one-hot编码 + amount0 + amount1

    GCN_out1 = base_model([X_in1, A_in1])
    GCN_out2 = base_model([X_in2, A_in2])

    # distance = Lambda(euclidean_distance)([GCN_out1, GCN_out2])
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([GCN_out1, GCN_out2])

    # 添加sigmoid激活函数，使输出在0到1之间
    distance = Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[X_in1, A_in1, X_in2, A_in2], outputs=distance)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # return Activation('sigmoid')(K.sqrt(K.maximum(sum_square, K.epsilon())))
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# def euclidean_distance(vects):
#     x, y = vects
#     difference = x - y
#     # Dense(1, activation='sigmoid')(difference)
#     return difference
    # return Activation('sigmoid')(K.sqrt(K.maximum(sum_square, K.epsilon())))
    # return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def triplet_loss_fn(anchor, positive, negative, margin=1.0):
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    loss = K.maximum(pos_dist - neg_dist + margin, 0.0)
    return K.mean(loss)

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels, dtype='float32')


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# 将数据加载为批次
def batch_generator(pairs, y_train, batch_size):
    while True:
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_labels = y_train[i:i + batch_size]
            x1, a1, e1 = [], [], []
            x2, a2, e2 = [], [], []
            for pair in batch_pairs:
                graph0, graph1 = pair
                x1.append(graph0.x)
                a1.append(graph0.a)
                # e1.append(graph0.e)
                x2.append(graph1.x)
                a2.append(graph1.a)
                # e2.append(graph1.e)
            x1 = tf.ragged.constant(x1).to_tensor()
            a1 = tf.ragged.constant(a1).to_tensor()
            # e1 = tf.ragged.constant(e1).to_tensor()
            x2 = tf.ragged.constant(x2).to_tensor()
            a2 = tf.ragged.constant(a2).to_tensor()
            # e2 = tf.ragged.constant(e2).to_tensor()
            yield ([x1, a1, x2, a2], batch_labels)

# 构建sandwichAttack pair
random.seed(918)
from sklearn.model_selection import train_test_split
tr_pairs, tr_y = load_gnnsandwichTrainDataset(feature_file, label_file, num_classes, classes)
# squeezed_array = np.squeeze(array)
tr_y = np.array(tr_y, dtype='float32')
# 将数据集分成训练集和临时集
tr_pairs, tr_pairs_temp, tr_y, tr_y_temp = train_test_split(tr_pairs, tr_y, test_size=0.3, random_state=42)

# 将临时集分成验证集和测试集
test_pairs, val_pairs, test_y, val_y = train_test_split(tr_pairs_temp, tr_y_temp, test_size=0.5, random_state=42)

# tr_pairs_0 = np.vstack((tr_pairs[0:1336, 0], tr_pairs[1670:3006, 0], tr_pairs[3340:4676, 0], tr_pairs[5010:6346, 0]))
# tr_pairs_1 = np.vstack((tr_pairs[0:1336, 1], tr_pairs[1670:3006, 1], tr_pairs[3340:4676, 1], tr_pairs[5010:6346, 1]))
# tr_y_ = np.hstack((tr_y[0:1336], tr_y[1670:3006], tr_y[3340:4676], tr_y[5010:6346]))
# # print(tr_y_.shape)
# # tr_y_ =
# test_pairs_0 = np.vstack(
#     (tr_pairs[1336:1502, 0], tr_pairs[3006:3172, 0], tr_pairs[4676:4842, 0], tr_pairs[6346:6512, 0]))
# test_pairs_1 = np.vstack(
#     (tr_pairs[1336:1502, 1], tr_pairs[3006:3172, 1], tr_pairs[4676:4842, 1], tr_pairs[6346:6512, 1]))
# test_y_ = np.hstack((tr_y[1336:1502], tr_y[3006:3172], tr_y[4676:4842], tr_y[6346:6512]))
# val_pairs_0 = np.vstack(
#     (tr_pairs[1502:1670, 0], tr_pairs[3172:3340, 0], tr_pairs[4842:5010, 0], tr_pairs[6512:6680, 0]))
# val_pairs_1 = np.vstack(
#     (tr_pairs[1502:1670, 1], tr_pairs[3172:3340, 1], tr_pairs[4842:5010, 1], tr_pairs[6512:6680, 1]))
# val_y_ = np.hstack((tr_y[1502:1670], tr_y[3172:3340], tr_y[4842:5010], tr_y[6512:6680]))

# test_pairs, test_y = load_gnnsandwichTrainDataset("./data/gnn_action_node data 17400498to17500498.txt", "./data/gnn_action_node target 17400498to17500498.txt", num_classes, classes)
# # tr_pairs, tr_y = load_gnnsandwichTrainDataset("./data/gnn data 17400498to17500498.txt", "./data/gnn target 17400498to17500498.txt", num_classes, classes)
# # test_pairs, test_y = load_gnnsandwichTrainDataset(feature_file, label_file, num_classes, classes)
# val_pairs, val_y = load_gnnsandwichTrainDataset("./data/gnn_action_node data many non normal hybrid 17500499to17600499.txt", "./data/gnn_action_node target many non normal hybrid 17500499to17600499.txt", num_classes, classes)
# tr_pairs_0 = tr_pairs[:, 0]
# tr_pairs_1 = tr_pairs[:, 1]
tr_y_ = np.array(tr_y, dtype='float32')
# test_pairs_0 = test_pairs[:, 0]
# test_pairs_1 = test_pairs[:, 1]
test_y_ = np.array(test_y, dtype='float32')
# val_pairs_0 = val_pairs[:, 0]
# val_pairs_1 = val_pairs[:, 1]
val_y_ = np.array(val_y, dtype='float32')

# 3

# tr_pairs_0 = np.squeeze(tr_pairs_0)
# tr_pairs_1 = np.squeeze(tr_pairs_1)
# test_pairs_0 = np.squeeze(test_pairs_0)
# test_pairs_1 = np.squeeze(test_pairs_1)
# val_pairs_0 = np.squeeze(val_pairs_0)
# val_pairs_1 = np.squeeze(val_pairs_1)

# print(len(tr_y))
# print(len(tr_pairs_0))
#
# print(tr_pairs_0.shape)
# input_shape = tr_pairs.shape[1:]
# input_shape = (105, 105, 1)
input_shape = (1, 35, 8)
# network definition
# base_network = create_base_network(input_shape)
#
# input_a = Input(shape=input_shape)
# input_b = Input(shape=input_shape)
#
# # because we re-use the same instance `base_network`,
# # the weights of the network
# # will be shared across the two branches
# processed_a = base_network(input_a)
# processed_b = base_network(input_b)
#
# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])



# model = Model([input_a, input_b], distance)
model = create_siamese_model(0, 0)
# model.load_weights(model_load_file)
# model.load_weights(initial_path)
# train
import matplotlib.pyplot as plt
from keras.callbacks import Callback, ModelCheckpoint
class PlotLearning(Callback):
    def on_train_begin(self, logs=None):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        plt.clf()
        plt.plot(self.acc, label='Training Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.pause(0.001)

plot_callback = PlotLearning()
plt.ion()  # 开启交互模式
plt.show()
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

rms = RMSprop()
# rms = RMSprop(learning_rate=0.001, rho=0.9)
sgd= SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
sgd= SGD()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# model.compile(optimizer=rms, loss=contrastive_loss, metrics=['accuracy'])

# model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           batch_size=128,
#           epochs=epochs,
#           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# # compute final accuracy on training and test sets
# y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy(tr_y, y_pred)
# y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(te_y, y_pred)
batch_size = 64
steps_per_epoch = len(tr_pairs) // batch_size
validation_steps = len(test_pairs) // batch_size

# print(len(tr_pairs))
# siamese_model.fit(batch_generator(pairs, y_train, batch_size), steps_per_epoch=steps_per_epoch, epochs=10)

# model.fit(batch_generator(tr_pairs, tr_y, batch_size), steps_per_epoch=steps_per_epoch,
#           epochs=epochs,
#           validation_data=(batch_generator(test_pairs, test_y, batch_size)),
#           callbacks=[cp_callback])
from keras.callbacks import ReduceLROnPlateau

# 在模型训练时添加学习率调度器
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
# model.fit(batch_generator(tr_pairs, tr_y, batch_size), steps_per_epoch=steps_per_epoch,
#           epochs=100, validation_data=(batch_generator(test_pairs, test_y, batch_size)), validation_steps=validation_steps, callbacks=[reduce_lr])
model.fit(batch_generator(tr_pairs, tr_y, batch_size), steps_per_epoch=steps_per_epoch,
          epochs=epochs, validation_data=(batch_generator(test_pairs, test_y, batch_size)), validation_steps=validation_steps,
          callbacks=[plot_callback, cp_callback])

model.save(model_file)
# compute final accuracy on training and test sets
# y_pred = model.predict(batch_generator(tr_pairs, tr_y, batch_size))
# print(y_pred)
# print(tr_y_)
# tr_acc = compute_accuracy(tr_y_, y_pred)
# y_pred = model.predict(batch_generator(val_pairs, val_y, batch_size))
# print(y_pred)
# print(val_y_)
# te_acc = compute_accuracy(val_y_, y_pred)

# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))