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

# def buildAttackSamples(attackTypeNum, ):


def buildeCNNComparePairs(testSample, attackSamples):
    sample1 = []
    sample2 = []
    for attacks in attackSamples:
        for attack in attacks:
            sample1.append(testSample)
            sample2.append(attack)
    return sample1, sample2

def buildeComparePairs(testSample, attackSamples):
    sample1_x = []
    sample1_a = []
    sample2_x = []
    sample2_a = []
    for attacks in attackSamples:
        for attack in attacks:
            sample1_x.append(testSample.x)
            sample1_a.append(testSample.a)
            sample2_x.append(attack.x)
            sample2_a.append(attack.a)
    sample1_x = tf.ragged.constant(sample1_x).to_tensor()
    sample1_a = tf.ragged.constant(sample1_a).to_tensor()
    sample2_x = tf.ragged.constant(sample2_x).to_tensor()
    sample2_a = tf.ragged.constant(sample2_a).to_tensor()
    return sample1_x, sample1_a, sample2_x, sample2_a

# def

def getAttackType(testSample, model, threshold, major, attackSamples):
    sample1_x, sample1_a, sample2_x, sample2_a = buildeComparePairs(testSample, attackSamples)
    predict = model.predict([sample1_x, sample1_a, sample2_x, sample2_a])
    pred = predict.ravel() < threshold
    maxPossibleAttack = -1
    maxSameNum = -1
    candidate = []
    for i in range(len(attackSamples)):
        sameAttack = np.sum(pred[len(attackSamples[0])*(i):len(attackSamples[0])*(i+1)])
        # for j in range(len(attackSamples[0])):
        if sameAttack > major and (sameAttack > maxSameNum or (sameAttack == maxSameNum and np.random.rand(1) < 1)):
            maxSameNum = sameAttack
            maxPossibleAttack = i
    return maxPossibleAttack

def getCNNAttackType(testSample, model, threshold, major, attackSamples):
    sample1, sample2 = buildeCNNComparePairs(testSample, attackSamples)
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    predict = model.predict([sample1, sample2])
    pred = predict.ravel() < threshold
    maxPossibleAttack = -1
    maxSameNum = -1
    for i in range(len(attackSamples)):
        sameAttack = np.sum(pred[len(attackSamples[0])*(i):len(attackSamples[0])*(i+1)])
        # for j in range(len(attackSamples[0])):
        if sameAttack > major and sameAttack > maxSameNum:
            maxSameNum = sameAttack
            maxPossibleAttack = i
    return maxPossibleAttack

def isNewAttack(testSample, model, threshold, major, attackSamples):
    return getAttackType(testSample, model, threshold, major, attackSamples) == -1
