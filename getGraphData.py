import zipfile

from build_dataset import *
import json
# bundle_raw_file = './sandwichAttack/normal data 15750000to16499999.txt'
# normalattack_file = './sandwichAttack/sandwichAttack data 15750000to16499999.txt'
# hybridattack_file = './sandwichAttack/hybrid sandwichAttack data 15750000to16499999 17245000to17300498.txt'
# manyattack_file = './sandwichAttack/many sandwichAttack data 15750000to16499999.txt'



blockNumber = 0

# Final Bundle Matrix
bundle_matrix = "./data/data_input_format_gnn_action_node 17500499to17600499.txt"
# real_from_to_file = "F:/etherem data/from_toV2.json"
# from_to_file = open(real_from_to_file, 'r', encoding='utf-8')
# file_content = from_to_file.read()
# from_to_file.close()
#
# # 解析 JSON 数据
# from_to_data = json.loads(file_content)

typeDic = {'swap': 1, 'add': 0, 'remove': 2}

def ToInt(str):
	return None if str=="None" else int(str)

def ToFloat(str):
	return 0 if str=="None" else float(str)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def process_files(input_files, data_file, target_file):
    fileDir2 = "E:/transactiondata/"
    #
    files2 = ["block_transactions"]
    theZIP1 = zipfile.ZipFile(fileDir2 + files2[0] + ".zip", 'r')
    theCSV1 = theZIP1.open(files2[0] + ".csv")
    head1 = theCSV1.readline()
    oneLine1 = theCSV1.readline().decode("utf-8").strip()
    # pair_to_token_map = from_pair_to_token("F:/etherem data/Defi exchange/UniswapV2/UniswapV2_PairInfo.csv")
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
    global_index = 1
    min_amount = float('inf')
    max_amount = float('-inf')
    oldBlockNumber = -1
    txsendtoDic = {}
    reset = False
    with open(data_file, 'w') as f_out, open(target_file, 'w') as f_target:
        # f_target.write(str(i) + '\n')
        for i in range(len(input_files)):
            input_file = input_files[i]
            err = False
            with open(input_file, 'r') as f_in:
                data = []
                hasPair = True
                for line in f_in:
                    # print(line)
                    if line.strip():  # 跳过空行
                        # pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, type, transactionHash = line.split()
                        # pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, transactionHash = line.split()
                        # type = 1

                        transactionHash, blockNumber, pairAddress, type, send, to, amount0In, amount1In, amount0Out, amount1Out = line.split()
                        type = typeDic[type]

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
                        #
                        # if int(blockNumber) > 19200000:
                        #     break

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

                        if min_amount == max_amount:
                            err = True
                            # max_amount += 1
                            continue

                            # continue

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
                        # print(data)
                        # f_out.write(
                        #     f"{pairAddress_index} {token0_index} {token1_index} {send_index} {to_index} {type} {amount0} {amount1} {transactionHash} \n")

                        prev_block_number = blockNumber
                        prev_transaction_hash = transactionHash
                    else:
                        # print(data)
                        if len(data) != 0 and err == False:
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
                        global_index = 1
                        token_index = 0
                        min_amount = float('inf')
                        max_amount = float('-inf')
                        hasPair = True
                        err = False
            # f_target.write(str(i) + '\n')


bundle_raw_file = '../UniswapV2/Bundles/bundle data without all 17500499to17600499.txt'
normalattack_file = '../UniswapV2/sandwichAttack/sandwichAttack data 17500499to17600499.txt'
hybridattack_file = '../UniswapV2/sandwichAttack/hybrid sandwichAttack data 17500499to17600499.txt'
manyattack_file = '../UniswapV2/sandwichAttack/many sandwichAttack data 17500499to17600499.txt'
add_swap_remove_file = "F:/etherem data/test/attack_type9/UniswapV3.txt"

# bundle_raw_file = '../UniswapV2/Bundles/bundle data without all 17400498to17500498.txt'
# normalattack_file = '../UniswapV2/sandwichAttack/sandwichAttack data 17400498to17500498.txt'
# hybridattack_file = '../UniswapV2/sandwichAttack/hybrid sandwichAttack data 17400498to17500498.txt'
# manyattack_file = '../UniswapV2/sandwichAttack/many sandwichAttack data 17400498to17500498.txt'


root_dir = "E:/Ethereum Data/Attack/bundle formatting/"
print("siamese")
#
# # bundle 初始处理
# bundle_process_file = "./repeatTestData/data many index 1.txt"
# bundle_process_target = "./repeatTestData/target many index 1.txt"
# bundle_fianl_target = "./repeatTestData/final target many index 1.txt"
# input_files = [manyattack_file]
# process_files(input_files, bundle_process_file, bundle_process_target)
#
#
# bundle_process_file = "./repeatTestData/data normal index 1.txt"
# bundle_process_target = "./repeatTestData/target normal index 1.txt"
# bundle_fianl_target = "./repeatTestData/final target normal index 1.txt"
# input_files = [normalattack_file]
# process_files(input_files, bundle_process_file, bundle_process_target)
#
# bundle_process_file = "./repeatTestData/data hybrid index 1.txt"
# bundle_process_target = "./repeatTestData/target hybrid index 1.txt"
# bundle_fianl_target = "./repeatTestData/final target hybrid index 1.txt"
# input_files = [hybridattack_file]
# process_files(input_files, bundle_process_file, bundle_process_target)

# bundle_process_file = "./repeatTestData/data non index 1.txt"
# bundle_process_target = "./repeatTestData/target non index 1.txt"
# bundle_fianl_target = "./repeatTestData/final target non index 1.txt"
# input_files = [bundle_raw_file]
# process_files(input_files, bundle_process_file, bundle_process_target)

# input_files = ["F:/etherem data/test/attack_lp/UniswapV2.txt"]
# process_files(input_files, "./repeatTestData/data swap_add_swap index 1.txt", "./repeatTestData/target swap_add_swap index 1.txt")
#
# input_files = ["F:/etherem data/test/attack_lt3/UniswapV2.txt"]
# process_files(input_files, "./repeatTestData/data swap_swap_swap_add_swap index 1.txt", "./repeatTestData/target swap_swap_swap_add_swap index 1.txt")

# input_files = ["F:/etherem data/test/attack_lt_lp/UniswapV2.txt"]
# process_files(input_files, "./repeatTestData/data swap_swap_add_swap index 1.txt", "./repeatTestData/target swap_swap_add_swap index 1.txt")
#
# input_files = ["F:/etherem data/test/attack_type9/UniswapV2.txt"]
# process_files(input_files, "./repeatTestData/data add_swap_remove index 1.txt", "./repeatTestData/target add_swap_remove index 1.txt")

# input_files = [add_swap_remove_file]
# process_files(input_files, "./repeatTestData/data add_swap_remove v3.txt", "./repeatTestData/target add_swap_remove v3.txt")

source_dir = "E:/Ethereum Data/Attack/collection/"
target_dir = "E:/Ethereum Data/Attack/bundle formatting/"

attack_lp = "CSA"
attack_lt = "LF"
attack_lt2 = "MLS"
attack_lt3 = "MPS"
attack_lt4 = "HLM"
attack_lt_lp = "MLL"
attack_many_lt = "LR"
attack_multiple_lt = "LPM"
attack_MBS = "MBS"
non = "Non"

# files = ["PancakeV3.txt", "SushiSwap.txt", "UniswapV2.txt", "UniswapV3.txt"]
files = ["SushiSwap.txt", "UniswapV2.txt"]
# files = ["PancakeV3.txt", "UniswapV3.txt"]

import os
# for subdir, dirs, files in os.walk(source_dir+attack_lp):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_lp + "/" + file], target_dir+attack_lp + "/" + file, "./data/target " + attack_lp + " " +  file)
#
# for subdir, dirs, files in os.walk(source_dir+attack_lt):
for file in files:
    # print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_lt + "/" + file], target_dir+attack_lt + "/" + file, "./data/target " + attack_lt + " " + file)
#
# for subdir, dirs, files in os.walk(source_dir+attack_lt2):
for file in files:
    # print(source_dir+attack_lt2+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_lt2 + "/" + file], target_dir+attack_lt2 + "/" + file, "./data/target " + attack_lt2 + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_lt3):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_lt3 + "/" + file], target_dir+attack_lt3 + "/" + file, "./data/target " + attack_lt3 + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_lt4):
for file in files:
    print(source_dir+attack_lt4+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_lt4 + "/" + file], target_dir+attack_lt4 + "/" + file, "./data/target " + attack_lt4 + " " +  file)
# file = "UniswapV3.txt"
# process_files([source_dir+attack_lt4 + "/" + file], target_dir+attack_lt4 + "/" + file, "./data/target " + attack_lt4 + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_lt_lp):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_lt_lp + "/" + file], target_dir+attack_lt_lp + "/" + file, "./data/target " + attack_lt_lp + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_many_lt):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_many_lt + "/" + file], target_dir+attack_many_lt + "/" + file, "./data/target " + attack_many_lt + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_multiple_lt):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_multiple_lt + "/" + file], target_dir+attack_multiple_lt + "/" + file, "./data/target " + attack_multiple_lt + " " +  file)

# for subdir, dirs, files in os.walk(source_dir+attack_MBS):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+attack_MBS + "/" + file], target_dir+attack_MBS + "/" + file, "./data/target " + attack_MBS + " " +  file)


# for subdir, dirs, files in os.walk(source_dir+attack_MBS):
for file in files:
    print(source_dir+attack_lp+file)
    if file == "UniswapV2_adjusted.txt" or file == "UniswapV2_haveToken.txt":
        continue
    process_files([source_dir+non + "/" + file], target_dir+non + "/" + file, "./data/target " + non + " " +  file)

