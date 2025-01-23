import zipfile

root_dir = "E:/Ethereum Data/Attack/collection/"
bundle_types = ["CSA", "MLS", "LF", "MPS", "Non"]
bundle_types = ["CSA", "MLS", "LF", "Non"]
bundle_types = ["CSA"]
# bundle_types = ["CSA", "MLS", "LF", "MPS", "Non", "HLM", "LPM", "MBS"' "MPS", "MLL", "LR"]
data_file = "E:/Ethereum Data/Attack/collection/" + bundle_types[0] + "/merge.txt"
target_file = "E:/Ethereum Data/Attack/label/" + bundle_types[0] + " merge label.txt"
data_file = "E:/Ethereum Data/Attack/collection/" + bundle_types[0] + "/merge 20000.txt"
target_file = "E:/Ethereum Data/Attack/label/" + bundle_types[0] + " merge label 20000.txt"
data_file = "E:/Ethereum Data/Attack/collection/" + bundle_types[0] + "/merge 42671_50202.txt"
target_file = "E:/Ethereum Data/Attack/label/" + bundle_types[0] + " merge label 42671_50202.txt"
dexs = ["PancakeV3.txt", "SushiSwap.txt", "UniswapV2.txt", "UniswapV3.txt"]
# dexs = ["merge 20000_21000.txt"]
# data_file = "./data/cnn train data_4_class_20000.txt"
# target_file = "./data/cnn train label_4_class_20000.txt"
# data_file = "./data/cnn fine-turn data_5_class_20000_21000.txt"
# target_file = "./data/cnn fine-turn label_5_class_20000_21000.txt"
MAXNum = 50202
MinNum = 42671
# MAXNum = 20000
# MinNum = -1

typeDic = {'swap': 1, 'add': 0, 'remove': 2}
def ToInt(str):
	return None if str=="None" else int(str)

def ToFloat(str):
	return 0 if str=="None" else float(str)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def process_files(types, input_files, data_file, target_file):
    count = [0] * len(types)
    with open(data_file, 'w') as f_out, open(target_file, 'w') as f_target:
        # f_target.write(str(i) + '\n')
        for i in range(len(types)):
            bundle_type = types[i]
            print(bundle_type)
            print(i)
            err = False
            for j in range(len(input_files)):
                if count[i] >= MAXNum:
                    break
                input_file = root_dir + str(bundle_type) + '/' + input_files[j]
                with open(input_file, 'r') as f_in:
                    data = []
                    hasPair = True
                    for line in f_in:
                        if count[i] >= MAXNum:
                            break
                        # print(len(f_in))
                        if line.strip():  # 跳过空行
                            # pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, type, transactionHash = line.split()
                            # pairAddress, send, to, amount0In, amount1In, amount0Out, amount1Out, blockNumber, transactionHash = line.split()
                            # type = 1
                            amount0In, amount1In, amount0Out, amount1Out = 0, 0, 0, 0
                            blockNumber = 0
                            if input_files[j] == "PancakeV3.txt" or input_files[j] == "UniswapV3.txt":
                                transactionHash, blockNumber, pairAddress, type, send, to, amount0, amount1 = line.split()
                                type = typeDic[type]
                                amount0 = ToFloat(amount0)
                                amount1 = ToFloat(amount1)
                                if type == 1 and amount0 > 0:
                                    amount0In, amount1In, amount0Out, amount1Out = amount0, 0, 0, -1 * amount1
                                elif type == 1 and amount0 < 0:
                                    amount0In, amount1In, amount0Out, amount1Out = 0, amount1, -1 * amount0, 0
                                elif type == 0:
                                    amount0In, amount1In, amount0Out, amount1Out = amount0, amount1, 0, 0
                                elif type == 2:
                                    amount0In, amount1In, amount0Out, amount1Out = 0, 0, amount0, amount1
                                data.append(
                                    [transactionHash, blockNumber, pairAddress, type, send, to, amount0In, amount1In,
                                     amount0Out, amount1Out])
                            else:
                                transactionHash, blockNumber, pairAddress, type, send, to, amount0In, amount1In, amount0Out, amount1Out = line.split()
                                type = typeDic[type]
                                data.append(
                                    [transactionHash, blockNumber, pairAddress, type, send, to, amount0In, amount1In,
                                     amount0Out, amount1Out])

                            if ToInt(blockNumber) < 16615778:
                                err = True
                                # max_amount += 1
                                continue
                            # pairAddress_index, send_index, to_index, amount0In, amount1In, amount0Out, amount1Out, type = line.split()
                            # data.append([pairAddress_index, send_index, to_index, amount0In, amount1In, amount0Out, amount1Out, type])
                        else:
                            print(count)
                            if len(data) != 0 and count[i] < MAXNum and err == False:
                                count[i] += 1
                                if count[i] > MinNum:
                                    # if len(data) != 0 and count[i] < MAXNum:
                                    for sub_array in data:
                                        line = ' '.join(map(str, sub_array))  # 将子数组的每个元素转换为字符串并用逗号分隔
                                        f_out.write(line + '\n')  # 写入文件并换行
                                    f_out.write('\n')
                                    f_target.write(str(i) + '\n')
                                    # count[i] += 1
                            prev_transaction_hash = None
                            data = []
                            if count[i] >= MAXNum:
                                break
                            err = False
                            # 只看一组交易的相对index
                # f_target.write(str(i) + '\n')


process_files(bundle_types, dexs, data_file, target_file)