# from siameseNet.test import *
from siameseNet.build_dataset import read_data_num
from siameseNet.gnnDataLoader import load_gnnsandwichTrainDataset, load_data, from_feature_to_graph_actionnode
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate, Flatten
from spektral.layers import GCNConv
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Activation, Conv2D, GlobalAveragePooling1D
from keras.optimizers import RMSprop, SGD
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from siameseNet.gnnDataLoader import load_gnnsandwichTrainDataset
from keras.models import load_model
from classificationAggregator import *
import matplotlib.pyplot as plt
# from keras_siamese2_sandwich import *
model_file = './MODEL/2cnn 64_128 1FC 128 4 class_20000.h5'
type_file_path = "2cnn 64_128 1FC 128 4 class_20000 TEST 7531.txt"
num = 500000
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # return Activation('sigmoid')(K.sqrt(K.maximum(sum_square, K.epsilon())))
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


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


def create_base_network1(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(32, (3,2), padding='same', activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(64, 2, padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    # x = Conv2D(128, 2, padding='same', activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    return Model(input, x)

def create_base_network2(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(32, (3,2), padding='same', activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(64, 2, padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    # x = Conv2D(128, 2, padding='same', activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    return Model(input, x)

def create_base_network3(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Conv2D(32, (3,2), padding='same', activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(32, 2, padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, 2, padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    # x = Conv2D(128, 2, padding='same', activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    return Model(input, x)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    './MODEL/2cnn 64_128 1FC 128 4 class_20000.h5' 名字起错了 应该是'./MODEL/2cnn 64_64 1FC 128 4 class_20000.h5'
    '''
    input = Input(shape=input_shape)
    x = Conv2D(64, (3,2), padding='same', activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(64, 2, padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    # x = Conv2D(64, 2, padding='same', activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Conv2D(128, 2, padding='same', activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    return Model(input, x)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

input_shape = (1, 35, 11)
# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)
initial_path = "./MODEL/cnn_sandwich_siamese2 5 class onehot.ckpt"
initial_path = "./MODEL/cnn_sandwich_siamese3 4 class 5 onehot 300.ckpt"
model.load_weights(model_file)
# model.load_weights('./MODEL/GCN_action_node_siamese_model.h5')
# loaded_model = tf.keras.models.load_model('./MODEL/GCN_action_node_siamese_model.h5', custom_objects={'GCNConv': GCNConv, 'euclidean_distance': euclidean_distance})
# , custom_objects={'euclidean_distance': euclidean_distance}
# loaded_model = load_model('./MODEL/GCN_action_node_siamese_model.h5', custom_objects={'GCNConv': GCNConv, 'euclidean_distance': euclidean_distance})



bundle_raw_file = "./cnn_repeatData/data_input_format non onehot.txt"
normalattack_file = "./cnn_repeatData/data_input_format normal onehot.txt"
hybridattack_file = "./cnn_repeatData/data_input_format hybrid onehot.txt"
manyattack_file = './cnn_repeatData/data_input_format many onehot.txt'
sas_file = "./cnn_repeatData/data_input_format swap_add_swap onehot.txt"
sssas_file = "./cnn_repeatData/data_input_format swap_swap_swap_add_swap onehot.txt"
ssas_file = "./cnn_repeatData/data_input_format swap_swap_add_swap onehot.txt"

bundle_raw_target = './cnn_repeatData/final target non onehot.txt'
normalattack_target = './cnn_repeatData/final target normal onehot.txt'
hybridattack_target = "./cnn_repeatData/final target hybrid onehot.txt"
manyattack_target = './cnn_repeatData/final target many onehot.txt'
sas_target = "./cnn_repeatData/final target swap_add_swap onehot.txt"
sssas_target = "./cnn_repeatData/final target swap_swap_swap_add_swap onehot.txt"
ssas_target = "./cnn_repeatData/final target swap_swap_add_swap onehot.txt"

CSA_file = "E:/Ethereum Data/Attack/cnn formatting/" + "CSA" + "/merge 20000.txt"
CSA_target_file = "./cnn_repeatData/final target " + "CSA" + " merge 20000.txt"
MLS_file = "E:/Ethereum Data/Attack/cnn formatting/" + "MLS" + "/merge 20000.txt"
MLS_target_file = "./cnn_repeatData/final target " + "MLS" + " merge 20000.txt"
LF_file = "E:/Ethereum Data/Attack/cnn formatting/" + "LF" + "/merge 20000.txt"
LF_target_file = "./cnn_repeatData/final target " + "LF" + " merge 20000.txt"
Non_file = "E:/Ethereum Data/Attack/cnn formatting/" + "Non" + "/merge 20000.txt"
Non_target_file = "./cnn_repeatData/final target " + "Non" + " merge 20000.txt"

bundle_data, bundle_labels = read_data_num(Non_file, Non_target_file, num)
normalattack_data, normalattack_labels = read_data_num(CSA_file, CSA_target_file, num)
# hybridattack_data, hybridattack_labels = load_data(MPS_file, MPS_target_file)
manyattack_data, manyattack_labels = read_data_num(MLS_file, MLS_target_file, num)
LF_data, LF_labels = read_data_num(LF_file, LF_target_file, num)

# print(len(normalattack_data))




# bundle_data, bundle_labels = load_data(bundle_raw_file, bundle_raw_target)
# normalattack_data, normalattack_labels = load_data(normalattack_file, normalattack_target)
# hybridattack_data, hybridattack_labels = load_data(hybridattack_file, hybridattack_target)
# manyattack_data, manyattack_labels = load_data(manyattack_file, manyattack_target)
# sas_data, sas_labels = load_data(sas_file, sas_target)
# sssas_data, sssas_labels = load_data(sssas_file, sssas_target)
# ssas_data, ssas_labels = load_data(ssas_file, ssas_target)

# bundle_data, bundle_labels = read_data_num(bundle_raw_file, bundle_raw_target, num)
# normalattack_data, normalattack_labels = read_data_num(normalattack_file, normalattack_target, num)
# hybridattack_data, hybridattack_labels = read_data_num(hybridattack_file, hybridattack_target, num)
# manyattack_data, manyattack_labels = read_data_num(manyattack_file, manyattack_target, num)
# sas_data, sas_labels = read_data_num(sas_file, sas_target, num)
# sssas_data, sssas_labels = read_data_num(sssas_file, sssas_target, num)
# ssas_data, ssas_labels = read_data_num(ssas_file, ssas_target, num)

# hybridattack_data1 = hybridattack_data[10:110]
# bundle_data1 = bundle_data[10:110]
# normalattack_data1 = normalattack_data[10:110]
# manyattack_data1 = manyattack_data[10:110]
# LF_data1 = LF_data[10:110]

# hybridattack_data1 = LF_data
# bundle_data1 = bundle_data
# normalattack_data1 = normalattack_data
# manyattack_data1 = manyattack_data
#
# print(len(hybridattack_data1))
# print(len(bundle_data1 ))
# print(len(normalattack_data1))
# print(len(manyattack_data1))


normalattack_data2 = []
bundle_data2 = []
hybridattack_data2 = []
manyattack_data2 = []
LF_data2 = []
random.seed(918)
for i in range(10):
    normalattack_data2.append(normalattack_data[random.randint(500, 700)])
    # hybridattack_data2.append(hybridattack_data[random.randint(500, 700)])
    bundle_data2.append(bundle_data[random.randint(500, 700)])
    manyattack_data2.append(manyattack_data[random.randint(500, 700)])
    LF_data2.append(LF_data[random.randint(500, 700)])


attackSamples = []
attackSamples.append(normalattack_data2)
# attackSamples.append(hybridattack_data2)
attackSamples.append(bundle_data2)
attackSamples.append(manyattack_data2)
attackSamples.append(LF_data2)
# attackSamples.append(sas_graph[:10])
# attackSamples.append(sssas_graph[:10])
# hybridattack_data1 = hybridattack_data[10:20]
# sas_graph = sas_graph[10:]

CSA_file = "E:/Ethereum Data/Attack/cnn formatting/" + "CSA" + "/merge 42671_50202.txt"
CSA_target_file = "./cnn_repeatData/final target " + "CSA" + " merge 42671_50202.txt"
MLS_file = "E:/Ethereum Data/Attack/cnn formatting/" + "MLS" + "/merge 42671_50202.txt"
MLS_target_file = "./cnn_repeatData/final target " + "MLS" + " merge 42671_50202.txt"
LF_file = "E:/Ethereum Data/Attack/cnn formatting/" + "LF" + "/merge 42671_50202.txt"
LF_target_file = "./cnn_repeatData/final target " + "LF" + " merge 42671_50202.txt"
Non_file = "E:/Ethereum Data/Attack/cnn formatting/" + "Non" + "/merge 42671_50202.txt"
Non_target_file = "./cnn_repeatData/final target " + "Non" + " merge 42671_50202.txt"

# bundle_data1, bundle_labels = read_data_num(Non_file, Non_target_file, num)
normalattack_data1, normalattack_labels = read_data_num(CSA_file, CSA_target_file, num)
# hybridattack_data, hybridattack_labels = load_data(MPS_file, MPS_target_file)
# manyattack_data1, manyattack_labels = read_data_num(MLS_file, MLS_target_file, num)
# LF_data1, LF_labels = read_data_num(LF_file, LF_target_file, num)
# print(len(LF_data1))
# print(len(bundle_data1 ))
print(len(normalattack_data1))
# print(len(manyattack_data1))

def testAccuracy(model, threshold, major, testSamples, attackSamples, y_true, record_fire):
    # print(verifyType)
    # sasattack_graph1 = from_feature_to_graph_actionnode(hybridattack_data1)
    typeincorrect = open(record_fire, "a+")
    incorrect = 0
    for i in range(len(testSamples)):
        attackType = getCNNAttackType(testSamples[i], model, threshold, major, attackSamples)
        if attackType != y_true[i]:
            incorrect += 1
            typeincorrect.write(str(i) + '\n')
            typeincorrect.write(str(incorrect) + ' ' + str(attackType) + '\n')
            typeincorrect.flush()
    typeincorrect.close()
    return incorrect / len(y_true)


threshold = 0.5
major = 4
normalAccuracies = []
hybridAccuracies = []
bundleAccuracies = []
manyAccuracies = []
sasAccuracies = []
ssasAccuracies = []
sssasAccuracies = []
thresholds = []
majors = []
acc = open("./cnn_repeatData/accurency.txt", "a+")
testdir = "./CNN Test/"
# for major in np.arange(len(attackSamples[0]) * 1 / 2+1, len(attackSamples[0]) * 1 / 2+1, 1):
# for major in np.arange(0, len(attackSamples[0]) + 1, 1):
# for major in np.arange(5, 6, 1):
for threshold in np.arange(0.5, 0.51, 0.05):
    print(str(threshold))
    thresholds.append(threshold)
    majors.append(major)
    file = str(threshold) + ' ' + str(major) + type_file_path
    # print("swap add swap")
    # sasAccuracy = test(model, threshold, major, sas_data, attackSamples, "./cnn_repeatData/sas incorrect" + file)
    # # print("swap swap swap add swap")
    # sssasAccuracy = test(model, threshold, major, sssas_data, attackSamples, "./cnn_repeatData/sssas incorrect" + file)
    # ssasAccuracy = test(model, threshold, major, ssas_data, attackSamples, "./cnn_repeatData/ssas incorrect" + file)


    # print("hybrid")
    normalAccuracy = testAccuracy(model, threshold, major, (normalattack_data1), attackSamples, [0]*len(normalattack_data1), testdir + "CSA incorrect" + file)
    # LFAccuracy = testAccuracy(model, threshold, major, (LF_data1), attackSamples, [3]*len(LF_data1), testdir + "LF incorrect" + file)
    # bundleAccuracy = testAccuracy(model, threshold, major, (bundle_data1), attackSamples, [1]*len(bundle_data1), testdir + "Non incorrect" + file)
    # manyAccuracy = testAccuracy(model, threshold, major, (manyattack_data1), attackSamples, [2]*len(manyattack_data1), testdir + "MLS incorrect" + file)
    # hybridAccuracy = testAccuracy(model, threshold, major, (hybridattack_data1), attackSamples,
    #                               [1] * len(hybridattack_data1), "./cnn_repeatData/hybrid incorrect" + file)

    # manyAccuracies.append(manyAccuracy)
    # acc.write(f"{str(normalAccuracy)} {str(hybridAccuracy)} {str(bundleAccuracy)} {str(manyAccuracy)} {str(sasAccuracy)} {str(sssasAccuracy)} {str(ssasAccuracy)}\n")
    # acc.flush()


acc.close()
print(normalAccuracies)
print(hybridAccuracies)
print(bundleAccuracies)
print(manyAccuracies)

# print(len(LF_data1))
# print(len(bundle_data1 ))
print(len(normalattack_data1))
# print(len(manyattack_data1))

accuracy = []
for i in range(len(normalAccuracies)):
    accuracy.append((normalAccuracies[i] + hybridAccuracies[i] + bundleAccuracies[i] + manyAccuracies[i]) / 4)
print(accuracy)
plt.figure(figsize=(10, 6))
plt.plot(majors, accuracy, marker='o', linestyle='-', color='b', label='accuracy')
# plt.title('Plot of sin(2πx)')
plt.xlabel('threshold')
plt.ylabel('accuracy')
plt.grid(False)
plt.legend()
plt.show()




