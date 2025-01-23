from classificationAggregator import *
import matplotlib.pyplot as plt
from spektral.layers.pooling import GlobalAvgPool

token_embedding_dim = 1
address_embedding_dim = 1
model_file = './MODEL/2GCN 64_128 1FC 128 4 class.h5'
type_file_path = "2GCN 64_128 1FC 128 4 class TEST 1000.txt"
def create_gcn_model(input_shapes, output_units):
    #  model 0
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = GCNConv(64,  activation='relu')([X_2, A_in])
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    x = Dense(64, activation='relu')(X_2)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=[X_in, A_in], outputs=x)

def create_gcn_model1(input_shapes, output_units):
    #  model 1
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')

    X_2 = GCNConv(32,  activation='relu')([X_in, A_in])
    X_2 = GCNConv(64,  activation='relu')([X_2, A_in])

    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    x = Dense(64, activation='relu')(X_2)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)

    return Model(inputs=[X_in, A_in], outputs=x)

def create_gcn_model2(input_shapes, output_units):
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    # X_2 = GCNConv(16, edge_dim=E_in.shape[-1], activation='relu')([A_in, E_in])
    X_2 = GCNConv(64,  activation='relu')([X_2, A_in])

    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    x = Dense(64, activation='relu')(X_2)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    return Model(inputs=[X_in, A_in], outputs=x)

def create_gcn_model3(input_shapes, output_units):
    #  model 0
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = GCNConv(64, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    x = Dense(64, activation='relu')(X_2)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=[X_in, A_in], outputs=x)

def create_gcn_model4(input_shapes, output_units):
    #  ./MODEL/GCN_action_node_siamese_model3 3 class pretrained.h5
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = GCNConv(64, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    x = Dense(64, activation='relu')(X_2)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=[X_in, A_in], outputs=x)

# 基于GCN的模型
def create_gcn_model5(input_shapes, output_units):
    #  './MODEL/GCN_action_node_siamese_model3 4 class fine-turning.h5'
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(64, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    X_2 = Dense(64, activation='relu')(X_2)
    X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model6(input_shapes, output_units):
    #  ./MODEL/2GCN 2FL 4 class.h5
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model7(input_shapes, output_units):
    #  './MODEL/3GCN 2FL 4 class.h5'
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model8(input_shapes, output_units):
    #  './MODEL/3GCN 2FL 256_128 4 class.h5'
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(256, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model9(input_shapes, output_units):
    #  './MODEL/2GCN 64_128 5 class.h5'
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    # # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model10(input_shapes, output_units):
    #  model 0
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    X_2 = GCNConv(128, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    # # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model11(input_shapes, output_units):
    #  model 0
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GlobalAvgPool()(X_2)  # 全局平均池化
    X_2 = GCNConv(64, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)
    # X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    X_2 = GlobalAvgPool()(X_2)
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    # # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model12(input_shapes, output_units):
    #  2GCN 64_64 1FC 128 4 class_20000
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GlobalAvgPool()(X_2)  # 全局平均池化
    X_2 = GCNConv(64, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = GlobalAvgPool()(X_2)
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    # # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(128, activation='relu')(X_2)
    return Model(inputs=[X_in, A_in], outputs=X_2)
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
    base_model = create_gcn_model10(input_shapes, output_units)

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

model = create_siamese_model(0, 0)
# base model3 './MODEL/GCN_action_node_siamese_model3GCN 5 class.h5'
model.load_weights(model_file)
acc = open("./Test/new 1000 accurency major.txt", "a+")
testdir = "./Test/"

# loaded_model = tf.keras.models.load_model('./MODEL/GCN_action_node_siamese_model.h5', custom_objects={'GCNConv': GCNConv, 'euclidean_distance': euclidean_distance})
# , custom_objects={'euclidean_distance': euclidean_distance}
# loaded_model = load_model('./MODEL/GCN_action_node_siamese_model.h5', custom_objects={'GCNConv': GCNConv, 'euclidean_distance': euclidean_distance})





CSA_file = "E:/Ethereum Data/Attack/bundle formatting/" + "CSA" + "/merge.txt"
CSA_target_file = "E:/Ethereum Data/Attack/label/" + "CSA" + " merge label.txt"
MLS_file = "E:/Ethereum Data/Attack/bundle formatting/" + "MLS" + "/merge.txt"
MLS_target_file = "E:/Ethereum Data/Attack/label/" + "MLS" + " merge label.txt"
LF_file = "E:/Ethereum Data/Attack/bundle formatting/" + "LF" + "/merge.txt"
LF_target_file = "E:/Ethereum Data/Attack/label/" + "LF" + " merge label.txt"
Non_file = "E:/Ethereum Data/Attack/bundle formatting/" + "Non" + "/merge.txt"
Non_target_file = "E:/Ethereum Data/Attack/label/" + "Non" + " merge label.txt"
MPS_file = "E:/Ethereum Data/Attack/bundle formatting/" + "MPS" + "/merge.txt"
MPS_target_file = "E:/Ethereum Data/Attack/label/" + "MPS" + " merge label.txt"

HLM_file = "E:/Ethereum Data/Attack/bundle formatting/" + "HLM" + "/merge.txt"
HLM_target_file = "E:/Ethereum Data/Attack/label/" + "HLM" + " merge label.txt"
LPM_file = "E:/Ethereum Data/Attack/bundle formatting/" + "LPM" + "/merge.txt"
LPM_target_file = "E:/Ethereum Data/Attack/label/" + "LPM" + " merge label.txt"
MBS_file = "E:/Ethereum Data/Attack/bundle formatting/" + "MBS" + "/merge.txt"
MBS_target_file = "E:/Ethereum Data/Attack/label/" + "MBS" + " merge label.txt"
LR_file = "E:/Ethereum Data/Attack/bundle formatting/" + "LR" + "/merge.txt"
LR_target_file = "E:/Ethereum Data/Attack/label/" + "LR" + " merge label.txt"
MLL_file = "E:/Ethereum Data/Attack/bundle formatting/" + "MLL" + "/merge.txt"
MLL_target_file = "E:/Ethereum Data/Attack/label/" + "MLL" + " merge label.txt"

bundle_data, bundle_labels = load_data(Non_file, Non_target_file)
normalattack_data, normalattack_labels = load_data(CSA_file, CSA_target_file)
hybridattack_data, hybridattack_labels = load_data(MPS_file, MPS_target_file)
manyattack_data, manyattack_labels = load_data(MLS_file, MLS_target_file)
LF_data, LF_labels = load_data(LF_file, LF_target_file)

HLM_data, HLM_labels = load_data(HLM_file, HLM_target_file)
LPM_data, LPM_labels = load_data(LPM_file, LPM_target_file)
MBS_data, MBS_labels = load_data(MBS_file, MBS_target_file)
LR_data, LR_labels = load_data(LR_file, LR_target_file)
MLL_data, MLL_labels = load_data(MLL_file, MLL_target_file)
# sas_data, sas_labels = load_data(sas_file, sas_target)
# sssas_data, sssas_labels = load_data(sssas_file, sssas_target)
# sas_data = lpv2_data
# sssas_data = ltv2_data
# ssas_data = ltlpv2_data


# hybridattack_data1 = hybridattack_data[10000:10100]
# bundle_data1 = bundle_data[10000:10100]
# normalattack_data1 = normalattack_data[10000:10100]
# manyattack_data1 = manyattack_data[10000:10100]
# LF_data1 = LF_data[2000:2100]
# HLM_data1 = HLM_data[:100]
# LPM_data1 = LPM_data[:100]
# MBS_data1 = MBS_data[:100]
# MLL_data1 = MLL_data[:100]
# LR_data1 = LR_data[:100]


hybridattack_data1 = hybridattack_data[10000:11000]
bundle_data1 = bundle_data[10000:11000]
normalattack_data1 = normalattack_data[10000:11000]
manyattack_data1 = manyattack_data[10000:11000]
LF_data1 = LF_data[2000:3000]
HLM_data1 = HLM_data[:1000]
LPM_data1 = LPM_data[:1000]
MBS_data1 = MBS_data[:1000]
MLL_data1 = MLL_data[:1000]
LR_data1 = LR_data[:1000]


normalattack_data2 = []
bundle_data2 = []
hybridattack_data2 = []
manyattack_data2 = []
LF_data2 = []
sas_data2 = []
sssas_data2 = []
ssas_data2 = []
random.seed(918)
for i in range(10):
    normalattack_data2.append(normalattack_data[random.randint(500, 700)])
    hybridattack_data2.append(hybridattack_data[random.randint(500, 700)])
    bundle_data2.append(bundle_data[random.randint(500, 700)])
    manyattack_data2.append(manyattack_data[random.randint(500, 700)])
    LF_data2.append(LF_data[random.randint(500, 700)])
    # sas_data2.append(sas_data[random.randint(1, 10)])
    # ssas_data2.append(ssas_data[random.randint(1, 10)])
    # sssas_data2.append(sssas_data[random.randint(1, 10)])

# sas_data2 = (sas_data[0:10])
# ssas_data2 = (ssas_data[0:10])
# sssas_data2 = (sssas_data[0:10])

normalattack_data = normalattack_data2
bundle_data = bundle_data2
hybridattack_data = hybridattack_data2
manyattack_data = manyattack_data2
LF_data = LF_data2
# sas_data = sas_data2
# sssas_data = sssas_data2
# ssas_data = ssas_data2

normalattack_graph = from_feature_to_graph_actionnode(normalattack_data)
bundle_graph = from_feature_to_graph_actionnode(bundle_data)
hybridattack_graph = from_feature_to_graph_actionnode(hybridattack_data)
manyattack_graph = from_feature_to_graph_actionnode(manyattack_data)
# sas_graph = from_feature_to_graph_actionnode(sas_data)
# sssas_graph = from_feature_to_graph_actionnode(sssas_data)
# ssas_graph = from_feature_to_graph_actionnode(ssas_data)
LF_graph = from_feature_to_graph_actionnode(LF_data)

attackSamples = []
attackSamples.append(normalattack_graph)
# attackSamples.append(hybridattack_graph)
attackSamples.append(bundle_graph)
attackSamples.append(manyattack_graph)
attackSamples.append(LF_graph)
# attackSamples.append(hybridattack_graph)
# attackSamples.append(sas_graph)
# attackSamples.append(sssas_graph)
# attackSamples.append(ssas_graph)
# hybridattack_data1 = hybridattack_data[10:20]
# sas_graph = sas_graph[10:]
# sssas_graph = sssas_graph[10:]

normalattack_graph1 = from_feature_to_graph_actionnode(normalattack_data1)
hybridattack_graph1 = from_feature_to_graph_actionnode(hybridattack_data1)
bundle_graph1 = from_feature_to_graph_actionnode(bundle_data1)
manyattack_graph1 = from_feature_to_graph_actionnode(manyattack_data1)
# sssas_graph1 = from_feature_to_graph_actionnode(sssas_data1)
# ssas_graph1 = from_feature_to_graph_actionnode(ssas_data1)
# sas_graph1 = from_feature_to_graph_actionnode(sas_data1)
LF_graph1 = from_feature_to_graph_actionnode(LF_data1)

HLM_graph1 = from_feature_to_graph_actionnode(HLM_data1)
LPM_graph1 = from_feature_to_graph_actionnode(LPM_data1)
MBS_graph1 = from_feature_to_graph_actionnode(MBS_data1)
LR_graph1 = from_feature_to_graph_actionnode(LR_data1)
MLL_graph1 = from_feature_to_graph_actionnode(MLL_data1)
# MPS_graph1 = from_feature_to_graph_actionnode(MPS_data1)

# with open("./repeatTestData/incorrect.txt", 'a') as f_out:

def testAccuracy(model, threshold, major, testSamples, attackSamples, y_true, record_fire):
    # print(verifyType)
    # sasattack_graph1 = from_feature_to_graph_actionnode(hybridattack_data1)
    typeincorrect = open(record_fire, "a+")
    incorrect = 0
    for i in range(len(testSamples)):
        attackType = getAttackType(testSamples[i], model, threshold, major, attackSamples)
        if attackType != y_true[i]:
            incorrect += 1
            typeincorrect.write(str(i) + '\n')
            typeincorrect.write(str(incorrect) + ' ' + str(attackType) + '\n')
            typeincorrect.flush()
    typeincorrect.close()
    return incorrect / len(y_true)
    # print()

def test(model, threshold, major, testSamples, attackSamples, record_fire):
    # print(verifyType)
    # sasattack_graph1 = from_feature_to_graph_actionnode(hybridattack_data1)
    typeincorrect = open(record_fire, "a+")
    incorrect = 0
    for i in range(len(testSamples)):
        attackType = getAttackType(testSamples[i], model, threshold, major, attackSamples)
        if attackType != -1:
            incorrect += 1
            typeincorrect.write(str(i) + '\n')
            typeincorrect.write(str(incorrect) + ' ' + str(attackType) + '\n')
            typeincorrect.flush()
    typeincorrect.close()


threshold = 0.7
major = len(attackSamples[0]) * 1 / 2
major = 4
normalAccuracies = []
hybridAccuracies = []
bundleAccuracies = []
manyAccuracies = []
thresholds = []
majors = []

# for major in np.arange(len(attackSamples[0]) * 1 / 2, len(attackSamples[0])+1, 1):
# for major in np.arange(8, len(attackSamples[0]), 1):
# for major in np.arange(4, 8, 1):
for threshold in np.arange(0.75, 0.76, 0.05):
# for threshold in np.arange(0.05, 1, 0.05):
    majors.append(major)
    thresholds.append(threshold)
    file = str(threshold) + ' ' + str(major) + type_file_path
    # print("swap add swap")
    # sssasAcc = testAccuracy(model, threshold, major, sssas_graph1, attackSamples, [-1]*len(sssas_data1), testdir + "sssas incorrect" + file)
    # # print("swap swap swap add swap")
    # ssasAcc = testAccuracy(model, threshold, major, ssas_graph1, attackSamples, [-1]*len(ssas_data1), testdir + "ssas incorrect" + file)
    LFAcc = testAccuracy(model, threshold, major, LF_graph1, attackSamples, [3]*len(LF_data1), testdir + "LF incorrect" + file)
    normalAccuracy = testAccuracy(model, threshold, major, normalattack_graph1, attackSamples, [0]*len(normalattack_data1), testdir + "normal incorrect" + file)
    hybridAccuracy = testAccuracy(model, threshold, major, (hybridattack_graph1), attackSamples, [4]*len(hybridattack_data1), testdir + "hybrid incorrect" + file)
    # asrAccuracy = testAccuracy(model, threshold, major, (asr_graph1), attackSamples, [4]*len(asr_data1), testdir + "asr incorrect" + file)
    manyAccuracy = testAccuracy(model, threshold, major, (manyattack_graph1), attackSamples,
                            [2] * len(manyattack_data1), testdir + "many incorrect" + file)
    bundleAccuracy = testAccuracy(model, threshold, major, (bundle_graph1), attackSamples, [1]*len(bundle_data1), testdir + "bundle incorrect" + file)

    HLMAccuracy = testAccuracy(model, threshold, major, (HLM_graph1), attackSamples,
                                  [-1] * len(HLM_data1), testdir + "HLM incorrect" + file)
    LPMAccuracy = testAccuracy(model, threshold, major, (LPM_graph1), attackSamples,
                            [-1] * len(LPM_data1), testdir + "LPM incorrect" + file)
    MBSAccuracy = testAccuracy(model, threshold, major, (MBS_graph1), attackSamples,
                                      [-1] * len(MBS_data1), testdir + "MBS incorrect" + file)
    MLLAccuracy = testAccuracy(model, threshold, major, (MLL_graph1), attackSamples,
                                      [-1] * len(MLL_data1), testdir + "MLL incorrect" + file)
    LRAccuracy = testAccuracy(model, threshold, major, (LR_graph1), attackSamples,
                                      [-1] * len(LR_data1), testdir + "LR incorrect" + file)

# acc.write(str(sasAcc) + ' ' + str(ssasAcc) + ' ' + str(sssasAcc) + ' ' + '\n')
    acc.flush()

acc.close()
print(normalAccuracies)
print(hybridAccuracies)
print(bundleAccuracies)
print(manyAccuracies)

print(len(normalattack_data1))
print(len(hybridattack_data1))
print(len(manyattack_data1))
print(len(bundle_data1))
print(len(LF_data1))
print(len(HLM_data1))
print(len(LPM_data1))
print(len(MBS_data1))
print(len(MLL_data1))
print(len(LR_data1))

# print(len(ssas_data1))
# print(len(sssas_data1))

accuracy = []
for i in range(len(normalAccuracies)):
    accuracy.append((normalAccuracies[i] + hybridAccuracies[i] + bundleAccuracies[i] + manyAccuracies[i]) / 4)
print(accuracy)
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, accuracy, marker='o', linestyle='-', color='b', label='accuracy')
# # plt.title('Plot of sin(2πx)')
# plt.xlabel('threshold')
# plt.ylabel('accuracy')
# plt.grid(False)
# plt.legend()
# plt.show()





    # incorrect = 290
    # count = 20679

# print("swap add swap")
# for graph in sas_graph:
#     gx = []
#     ga = []
#     gx.append(graph.x)
#     ga.append(graph.a)
#     gx = tf.ragged.constant(gx).to_tensor()
#     ga = tf.ragged.constant(ga).to_tensor()
#     isnormalattack = 0
#     for g in normalattack_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         isnormalattack += prediction
#     isbundle = 0
#     for g in bundle_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         isbundle += prediction
#     ishybrid = 0
#     for g in hybridattack_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         ishybrid += prediction
#     ismany = 0
#     for g in manyattack_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         ismany += prediction
#     print(isnormalattack)
#     print(ismany)
#     print(ishybrid)
#     print(isbundle)
#
# print("swap swap swap add swap")
# for graph in sssas_graph:
#     gx = []
#     ga = []
#     gx.append(graph.x)
#     ga.append(graph.a)
#     gx = tf.ragged.constant(gx).to_tensor()
#     ga = tf.ragged.constant(ga).to_tensor()
#     isnormalattack = 0
#     for g in normalattack_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         isnormalattack += prediction
#     isbundle = 0
#     for g in bundle_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         isbundle += prediction
#     ishybrid = 0
#     for g in hybridattack_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         ishybrid += prediction
#     ismany = 0
#     for g in manyattack_graph:
#         x = []
#         a = []
#         a.append(g.a)
#         x.append(g.x)
#         x = tf.ragged.constant(x).to_tensor()
#         a = tf.ragged.constant(a).to_tensor()
#         prediction = model.predict([gx, ga, x, a])
#         ismany += prediction
#     print(isnormalattack)
#     print(ismany)
#     print(ishybrid)
#     print(isbundle)
#
