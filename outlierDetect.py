import numpy as np
import matplotlib.pyplot as plt

from classificationAggregator import *
import matplotlib.pyplot as plt
from spektral.layers.pooling import GlobalAvgPool
from spektral.layers import GATConv
import untils


testdir = "./GAT triplet loss Test/"
token_embedding_dim = 1
address_embedding_dim = 1
model_file = './MODEL/2GAT 32_64 1FC 64 data_4_class_20000 triplet loss.h5'
# model_file = './MODEL/2GCN 64_128 1FC 256 data_4_class_20000 triplet loss.h5'
# # model_file = "./MODEL/2GAT 64_64 1FC 128 data_4_class_20000.h5"
# type_file_path = "2GCN 64_128 1FC 256 4 class NEW 500 outlier.txt"
model_file = './MODEL/2GAT 64_128 1FC 32 data_4_class_20000.h5'
# model_file = './MODEL/2GAT 64_128 1FC 512 data_4_class_20000 graph2.h5'
# type_file_path = "2GAT 64_128 1FC 512 data_4_class_20000 graph2 TEST 7531.txt"
type_file_path = "2GAT 64_128 1FC 32 data_4_class_20000 graph2 NEW 500 outlier.txt"

z_score_th = 3
z_mad_th = 3

# def create_gat_model(input_shapes, output_units, address_embedding_dim=1):
#     # 输入层
#     X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
#     A_in = Input(shape=(None, None), name='A_in')
#
#     # 第一层 GATConv（多头）
#     X_2 = GATConv(64, attn_heads=4, concat_heads=True, activation='elu')([X_in, A_in])
#     X_2 = Dropout(0.1)(X_2)
#
#     # 第二层 GATConv（通常设置为 heads=1, concat=False）
#     # X_2 = GATConv(64, attn_heads=4, concat_heads=False, activation='elu')([X_2, A_in])
#     # X_2 = Dropout(0.1)(X_2)
#
#     X_2 = GATConv(128, attn_heads=1, concat_heads=False, activation='elu')([X_2, A_in])
#     X_2 = Dropout(0.1)(X_2)
#
#     # 全局池化 + FC
#     X_2 = GlobalAveragePooling1D()(X_2)
#     X_2 = Dense(32, activation='relu')(X_2)
#
#     return Model(inputs=[X_in, A_in], outputs=X_2)

    # return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gat_model(input_shapes, output_units, address_embedding_dim=1):
    # 输入层
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')

    # 第一层 GATConv（多头）
    X_2 = GATConv(64, attn_heads=4, concat_heads=True, activation='elu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)

    # 第二层 GATConv（通常设置为 heads=1, concat=False）
    # X_2 = GATConv(64, attn_heads=4, concat_heads=False, activation='elu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)

    X_2 = GATConv(128, attn_heads=1, concat_heads=False, activation='elu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)

    # 全局池化 + FC
    X_2 = GlobalAveragePooling1D()(X_2)
    X_2 = Dense(32, activation='relu')(X_2)

    return Model(inputs=[X_in, A_in], outputs=X_2)

def create_gcn_model(input_shapes, output_units):
    #  2GCN 64_64 1FC 128 4 class_20000
    X_in = Input(shape=(None, address_embedding_dim * 4 + 4), name='X_in')
    A_in = Input(shape=(None, None), name='A_in')
    X_2 = GCNConv(64,  activation='relu')([X_in, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GlobalAvgPool()(X_2)  # 全局平均池化
    X_2 = GCNConv(128, activation='relu')([X_2, A_in])
    X_2 = Dropout(0.1)(X_2)
    # X_2 = GCNConv(128,  activation='relu')([X_2, A_in])
    # X_2 = Dropout(0.1)(X_2)
    X_2 = GlobalAveragePooling1D()(X_2)  # 全局平均池化
    # X_2 = GlobalAvgPool()(X_2)
    # X_2 = Dense(64, activation='relu')(X_2)
    # X_2 = Dropout(0.1)(X_2)
    # X_2 = Dense(128, activation='relu')(X_2)
    # # X_2 = Dropout(0.1)(X_2)
    X_2 = Dense(256, activation='relu')(X_2)
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

# model = create_gat_model(0, 0)
# base model3 './MODEL/GCN_action_node_siamese_model3GCN 5 class.h5'
siamese_model = create_siamese_model(0, 0)
siamese_model.load_weights(model_file)
submodels = [l for l in siamese_model.layers if isinstance(l, tf.keras.Model)]
print([m.name for m in submodels], len(submodels))

model = submodels[0]   # 通常就是它
# model =
print(model)
acc = open("./GAT Test/graph2 NEW 500 outlier.txt", "a+")

# 模拟每类 bundle 的样本数和特征维度
num_classes = 4
samples_per_class = 50
feature_dim = 16

# 模拟 base model 输出的 embedding（假设为 16 维）
# np.random.seed(42)
# class_embeddings = [np.random.randn(samples_per_class, feature_dim) + i*3 for i in range(num_classes)]

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
# LR_data, LR_labels = load_data(LR_file, LR_target_file)

HLM_data, HLM_labels = load_data(HLM_file, HLM_target_file)
LPM_data, LPM_labels = load_data(LPM_file, LPM_target_file)
MBS_data, MBS_labels = load_data(MBS_file, MBS_target_file)
LR_data, LR_labels = load_data(LR_file, LR_target_file)
MLL_data, MLL_labels = load_data(MLL_file, MLL_target_file)
MPS_data, MPS_labels = load_data(MPS_file, MPS_target_file)

def get_embeddings_from_graphs(graphs, base_model):
    embeddings = []
    for g in graphs:
        x = np.expand_dims(np.array(g.x), axis=0)  # shape: (1, N, d)
        a = np.expand_dims(np.array(g.a), axis=0)  # shape: (1, N, N)
        emb = base_model.predict([x, a])
        # print(emb)
        embeddings.append(emb[0])  # remove batch dimension
    return np.array(embeddings)

normalattack_graph = from_feature_to_graph_actionnode(normalattack_data[500:1000])
bundle_graph = from_feature_to_graph_actionnode(bundle_data[500:1000])
manyattack_graph = from_feature_to_graph_actionnode(manyattack_data[500:1000])
LF_graph = from_feature_to_graph_actionnode(LF_data[500:1000])

CSA_embeddings = get_embeddings_from_graphs(normalattack_graph, model)
Non_embeddings = get_embeddings_from_graphs(bundle_graph, model)
MLS_embeddings = get_embeddings_from_graphs(manyattack_graph, model)
LF_embeddings = get_embeddings_from_graphs(LF_graph, model)
class_embeddings = []
class_embeddings.append(CSA_embeddings)
class_embeddings.append(Non_embeddings)
class_embeddings.append(MLS_embeddings)
class_embeddings.append(LF_embeddings)

normalattack_graph1 = from_feature_to_graph_actionnode(normalattack_data[:500])

bundle_graph1 = from_feature_to_graph_actionnode(bundle_data[:500])
manyattack_graph1 = from_feature_to_graph_actionnode(manyattack_data[:500])
LF_graph1 = from_feature_to_graph_actionnode(LF_data[:500])


HLM_graph1 = from_feature_to_graph_actionnode(HLM_data[:500])
LPM_graph1 = from_feature_to_graph_actionnode(LPM_data[:500])
MBS_graph1 = from_feature_to_graph_actionnode(MBS_data[:500])
LR_graph1 = from_feature_to_graph_actionnode(LR_data[:500])
MLL_graph1 = from_feature_to_graph_actionnode(MLL_data[:500])
MPS_graph1 = from_feature_to_graph_actionnode(MPS_data[:500])

CSA_embeddings1 = get_embeddings_from_graphs(normalattack_graph1, model)
Non_embeddings1 = get_embeddings_from_graphs(bundle_graph1, model)
MLS_embeddings1 = get_embeddings_from_graphs(manyattack_graph1, model)
LF_embeddings1 = get_embeddings_from_graphs(LF_graph1, model)

HLM_embeddings1 = get_embeddings_from_graphs(HLM_graph1, model)
LPM_embeddings1 = get_embeddings_from_graphs(LPM_graph1, model)
MBS_embeddings1 = get_embeddings_from_graphs(MBS_graph1, model)
LR_embeddings1 = get_embeddings_from_graphs(LR_graph1, model)
MLL_embeddings1 = get_embeddings_from_graphs(MLL_graph1, model)
MPS_embeddings1 = get_embeddings_from_graphs(MPS_graph1, model)



# 计算每类的类中心 μi、类内距离均值 ui、标准差 σi
class_centers = []
class_distance_mean = []
class_distance_std = []
class_dist_median = []
class_dist_mad = []
eps=1e-12
use_robust_scale=True

for emb in class_embeddings:
    center = np.mean(emb, axis=0)
    dists = np.linalg.norm(emb - center, axis=1)
    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    class_centers.append(center)
    class_distance_mean.append(np.mean(dists))
    class_distance_std.append(np.std(dists))
    class_dist_median.append(float(med))
    class_dist_mad.append(float(max(mad, eps)))

def classBundle(test_embedding, class_centers, class_distance_mean, class_distance_std):

    # 计算该样本与每类中心的距离 ux
    ux_all = [np.linalg.norm(test_embedding - center) for center in class_centers]

    # 判断是否为新攻击
    is_unknown = True
    min_z = float("inf")
    predicted_class = -1

    for i, ux in enumerate(ux_all):
        ui = class_distance_mean[i]
        sigma = class_distance_std[i]
        z_score = (ux - ui) / sigma
        if z_score <= z_score_th and z_score < min_z:
            is_unknown = False
            min_z = z_score
            predicted_class = i

    return predicted_class


def classBundle_MAD(
        test_embedding,
        class_centers,
        class_distance_mean,
        class_distance_std,
):
    """
    基于 MAD 的鲁棒离群判断 + 最近类判定（取 z_mad 最小者）

    参数:
      test_embedding: np.ndarray shape=(D,)
      class_centers: list[np.ndarray], 每类中心 shape=(D,)
      class_distance_mean: list[float]
          推荐：传每类 distance 的 median（不是 mean）
          兼容：如果你还没算 median，就传 mean 也能跑，但鲁棒性差
      class_distance_std: list[float]
          推荐：传每类 distance 的 MAD（不是 std）
          兼容：如果你还没算 MAD，就传 std 也能跑，但鲁棒性差
      z_mad_th: float, 阈值（类似你 z_score_th）
      use_robust_scale: True 时使用 0.6745 缩放，False 则不缩放

    返回:
      predicted_class: int, -1 表示未知新类
      (可选扩展：你也可以返回 min_z 等信息)
    """

    # 计算该样本与每类中心的距离 ux
    ux_all = [np.linalg.norm(test_embedding - center) for center in class_centers]

    is_unknown = True
    min_z = float("inf")
    predicted_class = -1

    c = 0.6745 if use_robust_scale else 1.0

    for i, ux in enumerate(ux_all):
        # 推荐：ui=median, scale=MAD
        ui = float(class_distance_mean[i])
        scale = float(class_distance_std[i])

        # 防止 MAD=0（或传入为0）
        scale = max(scale, eps)

        # 鲁棒 z 分数：越小越“像”该类
        z_mad = c * (ux - ui) / scale

        # 你可以选择用 |z_mad|，取决于你怎么定义“离群”
        # 如果你的 ux 通常 >= ui（距离更大才离群），不取绝对值更符合你的原逻辑
        # z_mad = abs(z_mad)

        if z_mad <= z_mad_th and z_mad < min_z:
            is_unknown = False
            min_z = z_mad
            predicted_class = i

    return predicted_class

def testAccuracy(test_embeddings, class_centers, class_distance_mean, class_distance_std, y_true, record_fire):
    # print(verifyType)
    # sasattack_graph1 = from_feature_to_graph_actionnode(hybridattack_data1)
    typeincorrect = open(record_fire, "a+")
    incorrect = 0
    for i in range(len(test_embeddings)):
        attackType = classBundle(test_embeddings[i], class_centers, class_distance_mean, class_distance_std)
        if attackType != y_true[i]:
            incorrect += 1
            typeincorrect.write(str(i) + '\n')
            typeincorrect.write(str(incorrect) + ' ' + str(attackType) + '\n')
            typeincorrect.flush()
    typeincorrect.close()
    return incorrect / len(y_true)

    # result = {
    #     "test_embedding": test_embedding.flatten(),
    #     "ux_per_class": ux_all,
    #     "class_centers": [c.tolist() for c in class_centers],
    #     "class_distance_mean": class_distance_mean,
    #     "class_distance_std": class_distance_std,
    #     "is_unknown_attack": is_unknown,
    #     "predicted_class_if_known": predicted_class if not is_unknown else None
    # }

# import pandas as pd
# import ace_tools as tools; tools.display_dataframe_to_user(name="类间统计与判断结果", dataframe=pd.DataFrame(result))

normalAccuracies = []
hybridAccuracies = []
bundleAccuracies = []
manyAccuracies = []
thresholds = []
majors = []

print(len(normalattack_graph1))
print(len(manyattack_graph1))
print(len(bundle_graph1))
print(len(LF_graph1))
print(len(HLM_graph1))
print(len(LPM_graph1))
print(len(MBS_graph1))
print(len(LR_graph1))
print(len(MLL_graph1))

file = type_file_path

HLMAcc = testAccuracy(HLM_embeddings1, class_centers, class_distance_mean, class_distance_std, [-1]*len(HLM_graph1), testdir + "HLM incorrect" + file)
LPMAcc = testAccuracy(LPM_embeddings1, class_centers, class_distance_mean, class_distance_std, [-1]*len(LPM_graph1), testdir + "LPM incorrect" + file)
MBSAcc = testAccuracy(MBS_embeddings1, class_centers, class_distance_mean, class_distance_std, [-1]*len(MBS_graph1), testdir + "MBS incorrect" + file)
LRAcc = testAccuracy(LR_embeddings1, class_centers, class_distance_mean, class_distance_std, [-1]*len(LR_graph1), testdir + "LR incorrect" + file)
MLLAcc = testAccuracy(MLL_embeddings1, class_centers, class_distance_mean, class_distance_std, [-1] * len(MLL_graph1),
                      testdir + "MLL incorrect" + file)
MPSAcc = testAccuracy(MPS_embeddings1, class_centers, class_distance_mean, class_distance_std, [-1]*len(MPS_graph1), testdir + "MPS incorrect" + file)

LFAcc = testAccuracy(LF_embeddings1, class_centers, class_distance_mean, class_distance_std, [3]*len(LF_graph1), testdir + "LF incorrect" + file)
normalAccuracy = testAccuracy(CSA_embeddings1, class_centers, class_distance_mean, class_distance_std, [0]*len(normalattack_graph1), testdir + "normal incorrect" + file)
manyAccuracy = testAccuracy((MLS_embeddings1), class_centers, class_distance_mean, class_distance_std,
                        [2] * len(manyattack_graph1), testdir + "many incorrect" + file)
bundleAccuracy = testAccuracy((Non_embeddings1), class_centers, class_distance_mean, class_distance_std, [1]*len(bundle_graph1), testdir + "bundle incorrect" + file)

print(HLMAcc)
print(LPMAcc)
print(MBSAcc)
print(LRAcc)
print(MLLAcc)
print(MPSAcc)
print(LFAcc)
print(normalAccuracy)
print(manyAccuracy)
print(bundleAccuracy)




