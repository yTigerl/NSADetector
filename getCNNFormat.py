# 把bundle转换为CNN格式
import zipfile

import numpy as np

from build_dataset import *
# from learningSandwich.buileBundles import normalize, process_files, pad_data, write_to_file

# bundle_raw_file = './sandwichAttack/normal data 15750000to16499999.txt'
# normalattack_file = './sandwichAttack/sandwichAttack data 15750000to16499999.txt'
# hybridattack_file = './sandwichAttack/hybrid sandwichAttack data 15750000to16499999 17245000to17300498.txt'
# manyattack_file = './sandwichAttack/many sandwichAttack data 15750000to16499999.txt'

bundle_raw_file = '../UniswapV2/sandwichAttack/bundle data without all 17500499to17600499.txt'
normalattack_file = '../UniswapV2/sandwichAttack/sandwichAttack data 17500499to17600499.txt'
hybridattack_file = '../UniswapV2/sandwichAttack/hybrid sandwichAttack data 17500499to17600499.txt'
manyattack_file = '../UniswapV2/sandwichAttack/many sandwichAttack data 17500499to17600499.txt'

# bundle_raw_file = '../UniswapV2/Bundles/bundle data without all 17400498to17500498.txt'
# normalattack_file = '../UniswapV2/sandwichAttack/sandwichAttack data 17400498to17500498.txt'
# hybridattack_file = '../UniswapV2/sandwichAttack/hybrid sandwichAttack data 17400498to17500498.txt'
# manyattack_file = '../UniswapV2/sandwichAttack/many sandwichAttack data 17400498to17500498.txt'

# bundle 初始处理
bundle_process_file = "./Dataset/data many non normal hybrid 17500499to17600499 index from 1 t.txt"
bundle_process_target = "./Dataset/target many non normal hybrid 17500499to17600499 index from 1 t.txt"
bundle_fianl_target = "./Dataset/final target many non normal hybrid 17500499to17600499 index from 1 t.txt"

# Final Bundle Matrix
bundle_matrix = "./Dataset/data_input_format many non normal hybrid 17500499to17600499 index from 1 t.txt"


# normal = open("./sandwichAttack/normal 16000000to16249999.txt", "a+")
# normaldata = open(bundle_raw_file, "a+")
# normalBuffer = ""
# normaldataBuffer = ""

fileDir2 = "E:/transactiondata/"
#
files2 = ["block_transactions"]

count = 0

typeDic = {'swap': 1, 'add': 0, 'remove': 2}

def ToInt(str):
	return None if str=="None" else int(str)

def ToFloat(str):
	return 0 if str=="None" else float(str)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def process_files(input_files, data_file, target_file):
    theZIP1 = zipfile.ZipFile(fileDir2 + files2[0] + ".zip", 'r')
    theCSV1 = theZIP1.open(files2[0] + ".csv")
    head1 = theCSV1.readline()
    oneLine1 = theCSV1.readline().decode("utf-8").strip()
    hex_to_index_map = {}
    hex_addresses = []
    prev_block_number = None
    prev_transaction_hash = None
    global_index = 1
    min_amount = float('inf')
    max_amount = float('-inf')
    oldBlockNumber = -1
    txsendtoDic = {}
    with open(data_file, 'w') as f_out, open(target_file, 'w') as f_target:
        count = 0
        for i in range(len(input_files)):
            oldBlockNumber = -1
            txsendtoDic = {}
            input_file = input_files[i]
            err = False
            with open(input_file, 'r') as f_in:
                data = []
                for line in f_in:
                    if line.strip():  # 跳过空行
                        # transactionHash, blockNumber, pairAddress, type, send, to, amount0In, amount1In, amount0Out, amount1Out = line.split()
                        # type = typeDic[type]
                        # pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, type, transactionHash = line.split()
                        transactionHash, blockNumber, pairAddress, type, send, to, amount0In, amount1In, amount0Out, amount1Out = line.split()
                        # type = typeDic[type]

                        # transactionHash, blockNumber, pairAddress, type, send, to, amount0, amount1 = line.split()
                        # type = typeDic[type]
                        # amount0 = ToFloat(amount0)
                        # amount1 = ToFloat(amount1)
                        # if type == 1 and amount0 > 0:
                        #     amount0In, amount1In, amount0Out, amount1Out = amount0, 0, 0, -1 * amount1
                        # elif type == 1 and amount0 < 0:
                        #     amount0In, amount1In, amount0Out, amount1Out = 0, amount1, -1 * amount0, 0
                        # elif type == 0:
                        #     amount0In, amount1In, amount0Out, amount1Out = amount0, amount1, 0, 0
                        # elif type == 2:
                        #     amount0In, amount1In, amount0Out, amount1Out = 0, 0, amount0, amount1

                        if ToInt(blockNumber) < 16615778:
                            err = True
                            # max_amount += 1
                            continue

                        type_one_hot = [0, 0, 0, 0]
                        if int(type) == 0:
                            type_one_hot[int(type)] = 1
                        elif int(type) == 2:
                            type_one_hot[int(type) - 1] = 1
                        elif int(type) == 1 and ToFloat(amount0In) > 0:
                            type_one_hot[2] = 1
                        else:
                            type_one_hot[3] = 1

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

                        # print(amount0In)
                        # if amount0In
                        # 更新最小值和最大值
                        min_amount = float('inf')
                        max_amount = float('-inf')
                        min_amount = min(min_amount, ToFloat(amount0In), ToFloat(amount1In), ToFloat(amount0Out),
                                         ToFloat(amount1Out))
                        max_amount = max(max_amount, ToFloat(amount0In), ToFloat(amount1In), ToFloat(amount0Out),
                                         ToFloat(amount1Out))

                        if min_amount == max_amount:
                            err = True
                            # max_amount += 1
                            continue

                        # amount0, amount1 = 0, 0
                        # if type == 0:
                        #     amount0 = normalize(ToFloat(amount0In), min_amount, max_amount)
                        #     amount1 = normalize(ToFloat(amount1In), min_amount, max_amount)
                        # elif type == 2:
                        #     amount0 = -normalize(ToFloat(amount0Out), min_amount, max_amount)
                        #     amount1 = -normalize(ToFloat(amount1Out), min_amount, max_amount)
                        # elif ToFloat(amount0In) > 0:
                        #     amount0 = normalize(ToFloat(amount0In), min_amount, max_amount)
                        #     amount1 = -(normalize(ToFloat(amount1Out), min_amount, max_amount))
                        # else:
                        #     amount0 = -normalize(ToFloat(amount0Out), min_amount, max_amount)
                        #     amount1 = normalize(ToFloat(amount1In), min_amount, max_amount)
                        # if blockNumber != prev_block_number and prev_block_number is not None:
                        #     f_out.write('\n')
                        #     f_target.write(str(i) + '\n')
                        if transactionHash != prev_transaction_hash and prev_transaction_hash is not None and blockNumber == prev_block_number:
                            # f_out.write('-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n')
                            data.append([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                        data.append([pairAddress_index, send_index, to_index,
                                     normalize(ToFloat(amount0In), min_amount, max_amount),
                                     normalize(ToFloat(amount1In), min_amount, max_amount),
                                     normalize(ToFloat(amount0Out), min_amount, max_amount),
                                     normalize(ToFloat(amount1Out), min_amount, max_amount), type_one_hot[0],
                                     type_one_hot[1], type_one_hot[2], type_one_hot[3]])

                        prev_block_number = blockNumber
                        prev_transaction_hash = transactionHash
                    else:
                        if len(data) != 0 and err == False:
                            for sub_array in data:
                                line = ' '.join(map(str, sub_array))  # 将子数组的每个元素转换为字符串并用逗号分隔
                                f_out.write(line + '\n')  # 写入文件并换行
                            f_out.write('\n')
                            f_target.write(str(i) + '\n')
                        data = []
                        # f_out.write('\n')
                        # f_target.write(str(i) + '\n')
                        prev_transaction_hash = None
                        # 只看一组交易的相对index
                        hex_to_index_map = {}
                        global_index = 1
                        err = False
            f_target.write(str(i) + '\n')

def pad_data(data):
    with open(bundle_process_target, 'r') as f:
        lines = f.read().splitlines()
    target_index = 0
    final_targets = []
    padded_data = []
    part_len = []
    sample_len = []
    for sample in data:
        padded_sample = []
        for part in sample:
            while len(part) < 5:
                part.append([-1.0] * 11)
            part_len.append(len(part))
            if len(part) > 5:
                part = part[:5]
                # continue
            padded_sample.append(part)

        while len(padded_sample) < 7:
            padded_sample.append([[-1.0] * 11] * 5)
        sample_len.append(len(padded_sample))
        if len(padded_sample) > 7:
            # continue
            padded_sample = padded_sample[0:7]
        padded_data.append(padded_sample)
        final_targets.append(lines[target_index])
        target_index += 1
    print(max(part_len))
    print(max(sample_len))
        # print()
    return padded_data, final_targets

def write_to_file(data, output_file):
    with open(output_file, 'w') as f_out:
        for sample in data:
            for part in sample:
                for row in part:
                    f_out.write(' '.join(map(str, row)) + '\n')
                # f_out.write('-1 -1 -1 -1 -1 -1 -1 -1\n')
            f_out.write('\n')

def write_to_file1(data, output_file):
    with open(output_file, 'w') as f_out:
        for sample in data:
            for part in sample:
                f_out.write(' '.join(map(str, part)) + '\n')
                # for row in part:
                #     f_out.write(' '.join(map(str, row)) + '\n')
                # f_out.write('-1 -1 -1 -1 -1 -1 -1 -1\n')
            f_out.write('\n')

def process_data(input_file):
    with open(input_file, 'r') as file:
        data = file.read()

    # Split data into samples using double newline as the delimiter
    raw_samples = data.strip().split('\n\n')

    samples = []
    fromto = []
    type = []
    pairaddress = []
    amount = []
    for raw_sample in raw_samples:
        sample = []
        lines = raw_sample.split('\n')
        oldvalues = None
        for line in lines:
            # Split line by commas and convert each element to float
            values = list(map(float, line.split(',')))
            sample.append(values)
            if values[0] != -1:
                oldvalues = values
                type.append(values[7])
                pairaddress.append(values[2])
                amount.append(values[3])
                amount.append(values[4])
                amount.append(values[5])
                amount.append(values[6])
            else:
                fromto.append(oldvalues[0])
                fromto.append(-1)
                type.append(-1)
                pairaddress.append(-1)
                amount.append(-1)
        if len(fromto) < 30:
            fromto = np.hstack(fromto, [-1]*(30-len(fromto)))
        else:
            fromto = fromto[:30]
        if len(type) < 20:
            type = np.hstack(type, [-1]*(20-len(type)))
        else:
            type = type[:20]
        if len(pairaddress) < 20:
            pairaddress = np.hstack(pairaddress, [-1]*(20-len(pairaddress)))
        else:
            pairaddress = pairaddress[:20]
        if len(amount) < 50:
            amount = np.hstack(amount, [-1]*(50-len(amount)))
        else:
            amount = amount[:50]
        # samples.append(sample)
        data = np.hstack(fromto, type, pairaddress, amount)
        samples.append(data)
    # for sample in samples:
    return samples




def buildTrainData():
    print("siamese")
    input_files = [manyattack_file, bundle_raw_file, normalattack_file, hybridattack_file]
    process_files(input_files, bundle_process_file, bundle_process_target)

    # 读取bundle
    with open(bundle_process_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    current_sample = []
    tx_sample = []
    # 将原始数据转换为样本
    for line in lines:
        if line.strip() == '':
            while len(tx_sample) < 35:
                tx_sample.append([-1.0] * 8)
            # current_sample.append(tx_sample)
            if len(tx_sample) == 35:
                data.append(tx_sample)

            # current_sample = []
            tx_sample = []

        # elif line == "-1 -1 -1 -1 -1 -1 -1 -1":
        #     tx_sample.append(list(map(float, line.split())))
        #     current_sample.append(tx_sample)
        #     tx_sample = []
        else:
            tx_sample.append(list(map(float, line.split())))
    # current_sample.append(tx_sample)
    data.append(tx_sample)
    current_sample = []


    # 填充数据并写入文件
    # padded_data = pad_data(data)
    with open(bundle_matrix, 'w') as f_out:
        for sample in data:
            for part in sample:
                f_out.write(' '.join(map(str, part)) + '\n')
            f_out.write('\n')



def buildSiameseTrainData():
    print("siamese")

    # bundle 初始处理
    # bundle_process_file = "./data/data hybrid 15750000to16499999to 17245000to17300498.txt"
    # bundle_process_target = "./data/target hybrid 15750000to16499999to 17245000to17300498.txt"
    #
    # # Final Bundle Matrix
    # bundle_matrix = "./data/data_input_format3 hybrid 15750000to16499999to 17245000to17300498.txt"

    # input_files = ['./sandwichAttack/hybrid sandwichAttack data 15750000to16499999 17245000to17300498.txt']
    # input_files = [manyattack_file, bundle_raw_file, normalattack_file, hybridattack_file]
    # process_files(input_files, bundle_process_file, bundle_process_target)
    # print(bundle_process_file)
    # 读取bundle
    with open(bundle_process_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    current_sample = []
    tx_sample = []
    data_len = []
    tx_len = []
    # 将原始数据转换为样本
    for line in lines:
        if line.strip() == '':
            # while len(tx_sample) < 35:
            #     tx_sample.append([-1.0] * 8)
            current_sample.append(tx_sample)
            data.append(current_sample)

            tx_len.append(len(tx_sample))
            data_len.append(len(current_sample))
            current_sample = []
            tx_sample = []

        elif line == "-1 -1 -1 -1 -1 -1 -1 -1":
            tx_len.append(len(tx_sample))
            tx_sample.append(list(map(float, line.split())))
            current_sample.append(tx_sample)
            tx_sample = []
        else:
            tx_sample.append(list(map(float, line.split())))
    current_sample.append(tx_sample)
    data.append(current_sample)
    data_len.append(len(current_sample))
    current_sample = []
    # print(np.median(tx_len))
    # print(np.median(data_len))
    # tx_len = np.sort(tx_len)
    # data_len = np.sort(data_len)
    # print(tx_len[-10:])
    # print(data_len[-10:])
    # print(tx_len[:20000])
    # print(data_len[:20000])


    # import matplotlib.pyplot as  plt
    # plt.hist(tx_len)
    # plt.show()
    # plt.hist(data_len)
    # plt.show()
    # # print()



    # 填充数据并写入文件
    padded_data, final_targets = pad_data(data)
    write_to_file(padded_data, bundle_matrix)
    with open(bundle_fianl_target, 'w') as f_out:
        for sample in final_targets:
            f_out.write(' '.join(map(str, sample)) + '\n')
            # f_out.write('\n')

# buildTrainData()
# buildSiameseTrainData()

# input_files = ["F:/etherem data/test/attack_lp/UniswapV2.txt"]
# process_files(input_files, "./cnn_repeatData/data swap_add_swap.txt", "./repeatTestData/target swap_add_swap.txt")
#
# input_files = ["F:/etherem data/test/attack_lt3/UniswapV2.txt"]
# process_files(input_files, "./cnn_repeatData/data swap_swap_swap_add_swap.txt", "./repeatTestData/target swap_swap_swap_add_swap.txt")
#
# input_files = ["F:/etherem data/test/attack_lt_lp/UniswapV2.txt"]
# process_files(input_files, "./cnn_repeatData/data swap_swap_add_swap.txt", "./repeatTestData/target swap_swap_add_swap.txt")
#
# input_files = ["F:/etherem data/test/attack_type9/UniswapV2.txt"]
# process_files(input_files, "./cnn_repeatData/data add_swap_remove.txt", "./repeatTestData/target add_swap_remove.txt")

# bundle_process_file = "./cnn_repeatData/data swap_add_swap.txt"
# bundle_process_target = "./repeatTestData/target swap_add_swap.txt"
# bundle_fianl_target = "./cnn_repeatData/final target swap_add_swap.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format swap_add_swap.txt"
# buildSiameseTrainData()
#
# bundle_process_file = "./cnn_repeatData/data swap_swap_add_swap.txt"
# bundle_process_target = "./repeatTestData/target swap_swap_add_swap.txt"
# bundle_fianl_target = "./cnn_repeatData/final target swap_swap_add_swap.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format swap_swap_add_swap.txt"
# buildSiameseTrainData()
#
# bundle_process_file = "./cnn_repeatData/data swap_swap_swap_add_swap.txt"
# bundle_process_target = "./repeatTestData/target swap_swap_swap_add_swap.txt"
# bundle_fianl_target = "./cnn_repeatData/final target swap_swap_swap_add_swap.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format swap_swap_swap_add_swap.txt"
# buildSiameseTrainData()
#
# bundle_process_file = "./cnn_repeatData/data add_swap_remove.txt"
# bundle_process_target = "./repeatTestData/target add_swap_remove.txt"
# bundle_fianl_target = "./cnn_repeatData/final target add_swap_remove.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format add_swap_remove.txt"
# buildSiameseTrainData()

# bundle_process_file = "./cnn_repeatData/data non.txt"
# bundle_process_target = "./cnn_repeatData/target non.txt"
# bundle_fianl_target = "./cnn_repeatData/final target non.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format non.txt"
#
# input_files = [bundle_raw_file]
# process_files(input_files, bundle_process_file, bundle_process_target)
# buildSiameseTrainData()

# bundle_process_file = "./cnn_repeatData/data many.txt"
# bundle_process_target = "./cnn_repeatData/target many.txt"
# bundle_fianl_target = "./cnn_repeatData/final target many.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format many.txt"
#
# input_files = [manyattack_file]
# process_files(input_files, bundle_process_file, bundle_process_target)
# buildSiameseTrainData()
#
#
# bundle_process_file = "./cnn_repeatData/data normal.txt"
# bundle_process_target = "./cnn_repeatData/target normal.txt"
# bundle_fianl_target = "./cnn_repeatData/final target normal.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format normal.txt"
#
# input_files = [normalattack_file]
# process_files(input_files, bundle_process_file, bundle_process_target)
# buildSiameseTrainData()
#
# bundle_process_file = "./cnn_repeatData/data hybrid.txt"
# bundle_process_target = "./cnn_repeatData/target hybrid.txt"
# bundle_fianl_target = "./cnn_repeatData/final target hybrid.txt"
#
# # Final Bundle Matrix
# bundle_matrix = "./cnn_repeatData/data_input_format hybrid.txt"
#
# input_files = [hybridattack_file]
# process_files(input_files, bundle_process_file, bundle_process_target)
# buildSiameseTrainData()
#

source_dir = "E:/Ethereum Data/Attack/collection/"
target_dir = "E:/Ethereum Data/Attack/CNN formatting/"

attack_lp = "CSA"
attack_lt = "LF"
attack_lt2 = "MLS"
# attack_lt3 = "MPS"
# attack_lt4 = "HLM"
# attack_lt_lp = "MLL"
# attack_many_lt = "LR"
# attack_multiple_lt = "LPM"
# attack_MBS = "MBS"
non = "Non"

# files = ["PancakeV3.txt", "SushiSwap.txt", "UniswapV2.txt", "UniswapV3.txt"]
files = ["SushiSwap.txt", "UniswapV2.txt"]
# files = ["PancakeV3.txt", "UniswapV3.txt"]
# files = ["merge 20000.txt", "merge 42671_50202.txt"]
files = ["merge 42671_50202.txt"]

import os
# for subdir, dirs, files in os.walk(source_dir+attack_lp):
for file in files:
    print(source_dir+attack_lp+file)
    bundle_process_file = target_dir+attack_lp + "/tmp " + file
    bundle_process_target = "./cnn_repeatData/target " + attack_lp + " " +  file
    bundle_fianl_target = "./cnn_repeatData/final target " + attack_lp + " " +  file

    # Final Bundle Matrix
    bundle_matrix = target_dir+attack_lp + "/" + file
    process_files([source_dir+attack_lp + "/" + file], bundle_process_file, bundle_process_target)
    buildSiameseTrainData()
#
# for subdir, dirs, files in os.walk(source_dir+attack_lt):
# for file in files:
#     # print(source_dir+attack_lp+file)
#     bundle_process_file = target_dir + attack_lt + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_lt + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_lt + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_lt + "/" + file
#     process_files([source_dir+attack_lt + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#
# #
# # for subdir, dirs, files in os.walk(source_dir+attack_lt2):
# for file in files:
#     # print(source_dir+attack_lt2+file)
#     bundle_process_file = target_dir + attack_lt2 + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_lt2 + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_lt2 + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_lt2+ "/" + file
#     process_files([source_dir + attack_lt2 + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
    # process_files([source_dir+attack_lt2 + "/" + file], target_dir+attack_lt2 + "/" + file, "./data/target " + attack_lt2 + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_lt3):
# for file in files:
#     print(source_dir+attack_lp+file)
#     bundle_process_file = target_dir + attack_lt3 + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_lt3 + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_lt3 + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_lt3 + "/" + file
#     process_files([source_dir + attack_lt3 + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#     # process_files([source_dir+attack_lt3 + "/" + file], target_dir+attack_lt3 + "/" + file, "./data/target " + attack_lt3 + " " +  file)
#
# # for subdir, dirs, files in os.walk(source_dir+attack_lt4):
# for file in files:
#     print(source_dir+attack_lt4+file)
#     bundle_process_file = target_dir + attack_lt4 + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_lt4 + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_lt4 + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_lt4 + "/" + file
#     process_files([source_dir + attack_lt4 + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#     # process_files([source_dir+attack_lt4 + "/" + file], target_dir+attack_lt4 + "/" + file, "./data/target " + attack_lt4 + " " +  file)
# # file = "UniswapV3.txt"
# # process_files([source_dir+attack_lt4 + "/" + file], target_dir+attack_lt4 + "/" + file, "./data/target " + attack_lt4 + " " +  file)
#
# # for subdir, dirs, files in os.walk(source_dir+attack_lt_lp):
# for file in files:
#     print(source_dir+attack_lt_lp+file)
#     bundle_process_file = target_dir + attack_lt_lp + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_lt_lp + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_lt_lp + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_lt_lp + "/" + file
#     process_files([source_dir + attack_lt_lp + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#     # process_files([source_dir+attack_lt_lp + "/" + file], target_dir+attack_lt_lp + "/" + file, "./data/target " + attack_lt_lp + " " +  file)
# #
# # for subdir, dirs, files in os.walk(source_dir+attack_many_lt):
# for file in files:
#     print(source_dir+attack_many_lt+file)
#     bundle_process_file = target_dir + attack_many_lt + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_many_lt + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_many_lt + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_many_lt + "/" + file
#     process_files([source_dir + attack_many_lt + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#     # process_files([source_dir+attack_many_lt + "/" + file], target_dir+attack_many_lt + "/" + file, "./data/target " + attack_many_lt + " " +  file)
#
# # for subdir, dirs, files in os.walk(source_dir+attack_multiple_lt):
# for file in files:
#     print(source_dir+attack_multiple_lt+file)
#     bundle_process_file = target_dir + attack_multiple_lt + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_multiple_lt + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_multiple_lt + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_multiple_lt + "/" + file
#     process_files([source_dir + attack_multiple_lt + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#     # process_files([source_dir+attack_multiple_lt + "/" + file], target_dir+attack_multiple_lt + "/" + file, "./data/target " + attack_multiple_lt + " " +  file)
#
# # for subdir, dirs, files in os.walk(source_dir+attack_MBS):
# for file in files:
#     print(source_dir+attack_MBS+file)
#     bundle_process_file = target_dir + attack_MBS + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + attack_MBS + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + attack_MBS + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + attack_MBS + "/" + file
#     process_files([source_dir + attack_MBS + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
    # process_files([source_dir+attack_MBS + "/" + file], target_dir+attack_MBS + "/" + file, "./data/target " + attack_MBS + " " +  file)


# for subdir, dirs, files in os.walk(source_dir+attack_MBS):
# for file in files:
#     print(source_dir+non+file)
#     bundle_process_file = target_dir + non + "/tmp " + file
#     bundle_process_target = "./cnn_repeatData/target " + non + " " + file
#     bundle_fianl_target = "./cnn_repeatData/final target " + non + " " + file
#
#     # Final Bundle Matrix
#     bundle_matrix = target_dir + non + "/" + file
#     process_files([source_dir + non + "/" + file], bundle_process_file, bundle_process_target)
#     buildSiameseTrainData()
#     # process_files([source_dir+non + "/" + file], target_dir+non + "/" + file, "./data/target " + non + " " +  file)
