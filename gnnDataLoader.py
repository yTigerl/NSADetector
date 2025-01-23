import random
import zipfile

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from spektral.layers import GATConv
from spektral.data import Graph, Dataset, Loader
import numpy as np
from build_dataset import *
import numpy as np
from build_dataset import *
from spektral.transforms import AdjToSpTensor
import pandas as pd
# from learningSandwich.buileBundles import normalize, process_files, pad_data, write_to_file

# bundle_raw_file = './sandwichAttack/normal data 15750000to16499999.txt'
# normalattack_file = './sandwichAttack/sandwichAttack data 15750000to16499999.txt'
# hybridattack_file = './sandwichAttack/hybrid sandwichAttack data 15750000to16499999 17245000to17300498.txt'
# manyattack_file = './sandwichAttack/many sandwichAttack data 15750000to16499999.txt'

bundle_raw_file = '../UniswapV2/sandwichAttack/bundle data without all 17500499to17600499.txt'
normalattack_file = '../UniswapV2/sandwichAttack/sandwichAttack data 17500499to17600499.txt'
hybridattack_file = '../UniswapV2/sandwichAttack/hybrid sandwichAttack data 17500499to17600499.txt'
manyattack_file = '../UniswapV2/sandwichAttack/many sandwichAttack data 17500499to17600499.txt'
add_swap_remove_file = "F:/etherem data/test/attack_type9/UniswapV3.txt"

# bundle_raw_file = '../UniswapV2/Bundles/bundle data without all 17400498to17500498.txt'
# normalattack_file = '../UniswapV2/sandwichAttack/sandwichAttack data 17400498to17500498.txt'
# hybridattack_file = '../UniswapV2/sandwichAttack/hybrid sandwichAttack data 17400498to17500498.txt'
# manyattack_file = '../UniswapV2/sandwichAttack/many sandwichAttack data 17400498to17500498.txt'

# bundle 初始处理
bundle_process_file = "./data/gnn_action_node data 17500499to17600499 index from 1.txt"
bundle_process_target = "./data/gnn_action_node target 17500499to17600499 index from 1.txt"
bundle_fianl_target = "./data/gnn_action_node final target 17500499to17600499 index from 1.txt"

# Final Bundle Matrix
bundle_matrix = "./data/data_input_format_gnn_action_node 17500499to17600499 index from 1.txt"

bundle_process_file = "./data/gnn_action_node data add 17500499to17600499.txt"
bundle_process_target = "./data/gnn_action_node target add 17500499to17600499.txt"
bundle_fianl_target = "./data/gnn_action_node final target add 17500499to17600499.txt"

# Final Bundle Matrix
bundle_matrix = "./data/data_input_format_gnn_action_node add 17500499to17600499.txt"

fileDir2 = "E:/transactiondata/"
#
files2 = ["block_transactions"]
count = 0

typeDic = {'swap': 1, 'add': 0, 'remove': 2}

import pandas as pd
def from_pair_to_token(file_path):
    # df = pd.read_excel(file_path)
    df = pd.read_csv(file_path, usecols=['pairAddress', 'tokenAddress0', 'tokenAddress1'])
    # df = pd.read_csv(file_path, usecols=[0, 1, 2], names=["'pairaddress'", "'tokenAddress0'", "'tokenAddress1'"], header=0)
    # print(df.head())
    # print(df.columns)
    # result_dict = df.set_index("'pairaddress'").T.apply(lambda x: x.tolist()).to_dict()
    # result_dict = df.set_index('pairaddress').T.apply(lambda x: x.tolist()).to_dict()
    result_dict = df.set_index('pairAddress').T.apply(lambda x: x.tolist()).to_dict()

    return result_dict


def ToInt(str):
	return None if str=="None" else int(str)

def ToFloat(str):
	return 0 if str=="None" else float(str)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def process_files(input_files, data_file, target_file):
    # pair_to_token_map = from_pair_to_token("F:/etherem data/Defi exchange/UniswapV2/UniswapV2_PairInfo.csv")
    theZIP1 = zipfile.ZipFile(fileDir2 + files2[0] + ".zip", 'r')
    theCSV1 = theZIP1.open(files2[0] + ".csv")
    head1 = theCSV1.readline()
    oneLine1 = theCSV1.readline().decode("utf-8").strip()
    pair_to_index_map = {}
    send_to_index_map = {}
    to_to_index_map = {}
    token_to_index_map = {}
    token_index = 0
    pair_index = 0
    send_index = 0
    to_index = 0
    hex_to_index_map = {}
    hex_addresses = []
    prev_block_number = None
    prev_transaction_hash = None
    global_index = 0
    min_amount = float('inf')
    max_amount = float('-inf')

    with open(data_file, 'w') as f_out, open(target_file, 'w') as f_target:
        for i in range(len(input_files)):
            oldBlockNumber = -1
            txsendtoDic = {}
            theZIP1 = zipfile.ZipFile(fileDir2 + files2[0] + ".zip", 'r')
            theCSV1 = theZIP1.open(files2[0] + ".csv")
            head1 = theCSV1.readline()
            oneLine1 = theCSV1.readline().decode("utf-8").strip()
            input_file = input_files[i]
            with open(input_file, 'r') as f_in:
                data = []
                hasPair = True
                for line in f_in:
                    if line.strip():  # 跳过空行
                        # pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, transactionHash = line.split()
                        # if i == 3:
                        #     pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, type, transactionHash = line.split()
                        # else:
                        #     pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, transactionHash = line.split()
                        #     type = 1

                        if i == 1:
                            pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, type, transactionHash = line.split()
                        elif i == 4:
                            transactionHash, blockNumber, pairAddress, type, send, to, amount0, amount1 = line.split()
                            type = typeDic[type]
                            amount0 = ToFloat(amount0)
                            amount1 = ToFloat(amount1)
                            if type == 1 and amount0 > 0:
                                amount0In, amount1In, amount0Out, amount1Out = amount0, 0, 0, -1*amount1Out
                            elif type == 1 and amount0 < 0:
                                amount0In, amount1In, amount0Out, amount1Out = 0, amount1, -1 * amount0, 0
                            elif type == 0:
                                amount0In, amount1In, amount0Out, amount1Out = amount0, amount1, 0, 0
                            elif type == 2:
                                amount0In, amount1In, amount0Out, amount1Out = 0, 0, amount0, amount1

                            if int(blockNumber) < 17500499:
                                continue
                            if int(blockNumber) > 17880000:
                                break
                            if int(blockNumber) != oldBlockNumber:
                                oldBlockNumber = int(blockNumber)
                                txsendtoDic = {}
                                while (oneLine1 != ""):
                                    oneArray1 = oneLine1.split(",")
                                    blockNumber1 = int(oneArray1[0])
                                    # print(blockNumber1)
                                    # print(oldBlockNumber)
                                    if blockNumber1 == oldBlockNumber:
                                        transactionHash1 = oneArray1[1]
                                        sender1 = oneArray1[2]
                                        to1 = oneArray1[3]
                                        txsendtoDic[transactionHash1] = (sender1, to1)
                                        oneLine1 = theCSV1.readline().decode("utf-8").strip()
                                        # print(oneLine1)
                                    elif blockNumber1 < oldBlockNumber:
                                        oneLine1 = theCSV1.readline().decode("utf-8").strip()
                                    else:
                                        break
                            send = txsendtoDic[transactionHash][0]
                            to = txsendtoDic[transactionHash][1]

                        else:
                            pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, transactionHash = line.split()
                            type = 1

                        if int(blockNumber) < 17500499:
                            continue
                        if int(blockNumber) > 17880000:
                            break



                        # 如果地址是新出现的，则将其映射为新的全局索引
                        if pairAddress not in hex_to_index_map:
                            hex_to_index_map[pairAddress] = global_index
                            global_index += 1
                        if send not in hex_to_index_map:
                            hex_to_index_map[send] = global_index
                            global_index += 1
                        if to not in hex_to_index_map:
                            hex_to_index_map[to] = global_index
                            global_index += 1



                        pairAddress_index = hex_to_index_map[pairAddress]
                        send_index = hex_to_index_map[send]
                        to_index = hex_to_index_map[to]
                        blockNumber = int(blockNumber)
                        # print(pair_to_token_map[pairAddress])


                        # print(amount0In)
                        # if amount0In
                        # 更新最小值和最大值
                        min_amount = float('inf')
                        max_amount = float('-inf')
                        min_amount = min(min_amount, ToFloat(amount0In), ToFloat(amount1In), ToFloat(amount0Out),
                                         ToFloat(amount1Out))
                        max_amount = max(max_amount, ToFloat(amount0In), ToFloat(amount1In), ToFloat(amount0Out),
                                         ToFloat(amount1Out))

                        amount0, amount1 = 0, 0
                        if type == 0:
                            amount0 = normalize(ToFloat(amount0In), min_amount, max_amount)
                            amount1 = normalize(ToFloat(amount1In), min_amount, max_amount)
                        elif type == 2:
                            amount0 = -normalize(ToFloat(amount0Out), min_amount, max_amount)
                            amount1 = -normalize(ToFloat(amount1Out), min_amount, max_amount)
                        elif ToFloat(amount0In) > 0:
                            amount0 = normalize(ToFloat(amount0In), min_amount, max_amount)
                            amount1 = -(normalize(ToFloat(amount1Out), min_amount, max_amount))
                        else:
                            amount0 = -normalize(ToFloat(amount0Out), min_amount, max_amount)
                            amount1 = normalize(ToFloat(amount1In), min_amount, max_amount)
                        # if transactionHash != prev_transaction_hash and prev_transaction_hash is not None and blockNumber == prev_block_number:
                        #     f_out.write('-1 -1 -1 -1 -1 -1 -1 -1\n')

                        data.append([pairAddress_index, send_index, to_index, type, amount0, amount1, transactionHash])
                        # f_out.write(
                        #     f"{pairAddress_index} {token0_index} {token1_index} {send_index} {to_index} {type} {amount0} {amount1} {transactionHash} \n")

                        prev_block_number = blockNumber
                        prev_transaction_hash = transactionHash
                    else:
                        if len(data) != 0:
                            for sub_array in data:
                                line = ' '.join(map(str, sub_array))  # 将子数组的每个元素转换为字符串并用逗号分隔
                                f_out.write(line + '\n')  # 写入文件并换行
                            f_out.write('\n')
                            f_target.write(str(i) + '\n')
                        prev_transaction_hash = None
                        data = []
                        # 只看一组交易的相对index
                        hex_to_index_map = {}
                        token_to_index_map = {}
                        global_index = 0
                        token_index = 0
                        min_amount = float('inf')
                        max_amount = float('-inf')
                        hasPair = True
            f_target.write(str(i) + '\n')
num = 80000



# 准备数据集
class MyDataset(Dataset):
    def read(self, transactions):
        output = []
        token_vocab = {val: idx for idx, val in enumerate(
            set(op[1] for tx in transactions for op in tx) | set(op[2] for tx in transactions for op in tx))}
        address_vocab = {val: idx for idx, val in enumerate(
            set(op[3] for tx in transactions for op in tx) | set(op[4] for tx in transactions for op in tx))}

        token_vocab_size = len(token_vocab) + 1
        address_vocab_size = len(address_vocab) + 1
        token_embedding_dim = 8
        address_embedding_dim = 8

        # 定义嵌入层
        token_embedding = tf.keras.layers.Embedding(token_vocab_size, token_embedding_dim)
        address_embedding = tf.keras.layers.Embedding(address_vocab_size, address_embedding_dim)
        for tx in transactions:
            node_features = {}
            edges = []
            edge_features = []

            for op in tx:
                pair, token0, token1, from_addr, to_addr, type_, amount0, amount1, txhash = op
                token0_idx = token_vocab[token0]
                token1_idx = token_vocab[token1]
                from_idx = address_vocab[from_addr]
                to_idx = address_vocab[to_addr]
                type_one_hot = [0, 0, 0]
                type_one_hot[type_] = 1

                if token0_idx not in node_features:
                    node_features[token0_idx] = token_embedding(token0_idx)
                if token1_idx not in node_features:
                    node_features[token1_idx] = token_embedding(token1_idx)

                edge_feature = tf.concat([address_embedding(from_idx), address_embedding(to_idx),
                                          tf.convert_to_tensor(type_one_hot, dtype=tf.float32),
                                          tf.convert_to_tensor([amount0, amount1], dtype=tf.float32)], axis=-1)
                edges.append([token0_idx, token1_idx])
                edge_features.append(edge_feature)

            node_features = tf.stack(list(node_features.values()))
            edge_index = np.array(edges).T
            edge_features = tf.stack(edge_features)

            # 邻接矩阵转换为稀疏矩阵
            adj_matrix = np.zeros((len(node_features), len(node_features)))
            for edge in edges:
                adj_matrix[edge[0], edge[1]] = 1
                adj_matrix[edge[1], edge[0]] = 1  # 无向图

            adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

            # graph = Graph(x=node_features, a=adj_matrix, e=edge_features, y=np.array([1]))  # 假设所有样本标签为1
            graph = Graph(x=node_features, a=adj_matrix, e=edge_features, y=np.array([1]))
            output.append(graph)
        return output


def from_feature_to_graph(transactions):
    output = []
    token_vocab = {val: idx for idx, val in enumerate(
        set(op[1] for tx in transactions for op in tx) | set(op[2] for tx in transactions for op in tx))}
    address_vocab = {val: idx for idx, val in enumerate(
        set(op[3] for tx in transactions for op in tx) | set(op[4] for tx in transactions for op in tx))}

    token_vocab_size = len(token_vocab) + 1
    address_vocab_size = len(address_vocab) + 1
    token_embedding_dim = 8
    address_embedding_dim = 8

    # 定义嵌入层
    # token_embedding = tf.keras.layers.Embedding(token_vocab_size, token_embedding_dim)
    # address_embedding = tf.keras.layers.Embedding(address_vocab_size, address_embedding_dim)
    for tx in transactions:
        node_features = {}
        edges = []
        edge_features = []

        for op in tx:
            pair, token0, token1, from_addr, to_addr, type_, amount0, amount1 = op
            # token0_idx = token_vocab[token0]
            # token1_idx = token_vocab[token1]
            # from_idx = address_vocab[from_addr]
            # to_idx = address_vocab[to_addr]
            token0_idx = token0
            token1_idx = token1
            from_idx = from_addr
            to_idx = to_addr
            type_one_hot = [0, 0, 0]
            type_one_hot[int(type_)] = 1

            if token0_idx not in node_features:
                node_features[token0_idx] = token0_idx
            if token1_idx not in node_features:
                node_features[token1_idx] = token1_idx

            # edge_feature = tf.concat([tf.convert_to_tensor(from_idx, dtype=tf.float32), tf.convert_to_tensor(to_idx, dtype=tf.float32), tf.convert_to_tensor(type_one_hot, dtype=tf.float32), tf.convert_to_tensor([amount0, amount1], dtype=tf.float32)], axis=-1)
            edge_feature = np.concatenate([
                np.expand_dims(np.array(from_idx, dtype=np.float32), axis=-1),
                np.expand_dims(np.array(to_idx, dtype=np.float32), axis=-1),
                np.array(type_one_hot, dtype=np.float32),
                np.array([amount0, amount1], dtype=np.float32)
            ], axis=-1)
            edges.append([token0_idx, token1_idx])
            edge_features.append(edge_feature)

        node_features = np.stack(list(node_features.values()))
        edge_index = np.array(edges).T
        edge_features = np.stack(edge_features)

        # 邻接矩阵转换为稀疏矩阵
        # adj_matrix = np.zeros((len(node_features), len(node_features)))
        # for edge in edges:
        #     adj_matrix[edge[0], edge[1]] = 1
        #     adj_matrix[edge[1], edge[0]] = 1  # 无向图
        #
        # adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

        graph = Graph(x=node_features, a=edge_index, e=edge_features, y=np.array([1]))  # 假设所有样本标签为1
        output.append(graph)
    return output

def from_feature_to_graph_actionnode(transactions):
    output = []

    # 定义嵌入层
    # token_embedding = tf.keras.layers.Embedding(token_vocab_size, token_embedding_dim)
    # address_embedding = tf.keras.layers.Embedding(address_vocab_size, address_embedding_dim)
    for tx in transactions:
        node_features = []
        edges = []
        edge_features = []

        for op in tx:
            # print(op)
            pair, from_addr, to_addr, type_, amount0, amount1, txhash = op
            pair = float(pair)
            from_addr = float(from_addr)
            to_addr = float(to_addr)
            type_ = float(type_)
            amount0 = float(amount0)
            amount1 = float(amount1)
            from_idx = from_addr
            to_idx = to_addr
            type_one_hot = [0, 0, 0, 0]
            if int(type_) == 0:
                type_one_hot[int(type_)] = 1
            elif int(type_) == 2:
                type_one_hot[int(type_)-1] = 1
            elif int(type_) == 1 and amount0 > 0:
                type_one_hot[2] = 1
            else:
                type_one_hot[3] = 1
            node_feature = np.concatenate([
                np.expand_dims(np.array(pair, dtype=np.float32), axis=-1),
                # np.expand_dims(np.array(token, dtype=np.float32), axis=-1),
                np.expand_dims(np.array(from_idx, dtype=np.float32), axis=-1),
                # np.expand_dims(np.array(to_idx, dtype=np.float32), axis=-1),
                np.array(type_one_hot, dtype=np.float32),
                np.array([amount0, amount1], dtype=np.float32)
            ], axis=-1)
            # edges.append([token0_idx, token1_idx])
            node_features.append(node_feature)
            # print(edges)


        # node_features = np.stack(list(node_features.values()))
        node_features = np.stack(node_features)
        edge_index = np.array(edges).T
        # edge_features = np.stack(edge_features)

        # 邻接矩阵转换为稀疏矩阵
        adj_matrix = np.zeros((len(node_features), len(node_features)))

        for i in range(len(tx)-1):
            for j in range(i, len(tx)):
                if tx[i][0] == tx[j][0]:
                    adj_matrix[i, j] = 1
                    # adj_matrix[j, i] = 1
                    edge_features.append([1])
        adj_matrix[len(tx)-1, len(tx)-1] = 1
        edge_features.append([1])
        for i in range(len(tx) - 1):
            if tx[i+1][-1] == tx[i][-1] and tx[i][0] != tx[j][0] and adj_matrix[i, i+1] != 1:
                edge_features.append([0])
                adj_matrix[i, i+1] = 1
        for i in range(len(tx) - 1):
            oldTxHash = tx[i][-1]
            oldCurrentTxHah = None
            # next = False
            for j in range(i+1, len(tx)):
                currentTxHah = tx[j][-1]
                if currentTxHah != oldTxHash and currentTxHah != oldCurrentTxHah and tx[i][0] != tx[j][0]:
                    oldCurrentTxHah = currentTxHah
                    adj_matrix[i, j] = 1
                    edge_features.append([0])
                if currentTxHah != oldTxHash and oldCurrentTxHah != None and currentTxHah == oldCurrentTxHah and tx[i][0] != tx[j][0]:
                    adj_matrix[i, j] = 1
                    edge_features.append([0])
                if currentTxHah != oldTxHash and oldCurrentTxHah != None and currentTxHah != oldCurrentTxHah:
                    break

        edge_features = np.array(edge_features).T
        graph = Graph(x=node_features, a=adj_matrix, e=edge_features)
        output.append(graph)
    return output

def load_data(feature_file, label_file):
    with open(feature_file, 'r') as f:
        feature_lines = f.readlines()

    with open(label_file, 'r') as f:
        label_lines = f.readlines()

    data = []
    target = []
    current_sample = []
    # print(feature_lines)
    sample = 0
    counts = [0, 0, 0, 0, 0]
    for feature_line in feature_lines:
        if feature_line.strip():  # 跳过空行
            # print(feature_line)
            a = list(map(float, feature_line.strip().split()[0:-1]))
            b = feature_line.strip().split()[-1]
            a.append(b)
            # np.array(a)
            # print(a)
            current_sample.append(np.array(a))
        else:

            data.append(np.array(current_sample))
            # print(data.__sizeof__())
            target.append(int(label_lines[sample].strip()))
            counts[int(label_lines[sample].strip())] += 1
            current_sample = []
            sample += 1

    return np.array(data), np.array(target)

def buildTrainData():
    print("siamese")
    input_files = [manyattack_file, bundle_raw_file, normalattack_file, hybridattack_file, add_swap_remove_file]
    process_files(input_files, bundle_process_file, bundle_process_target)

    # 读取bundle
    # with open(bundle_process_file, 'r') as f:
    #     lines = f.read().splitlines()

# buildTrainData()

