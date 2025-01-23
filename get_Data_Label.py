import zipfile
root_dir = "E:/Ethereum Data/Attack/bundle formatting/"
bundle_types = ["CSA", "MLS", "LF", "MPS", "Non"]
# bundle_types = ["CSA", "MLS", "LF", "Non"]
bundle_types = ["Non"]
# bundle_types = ["CSA", "MLS", "LF", "MPS", "Non", "HLM", "LPM", "MBS"' "MPS", "MLL", "LR"]
data_file = "E:/Ethereum Data/Attack/bundle formatting/" + bundle_types[0] + "/merge.txt"
target_file = "E:/Ethereum Data/Attack/label/" + bundle_types[0] + " merge label.txt"
data_file = "E:/Ethereum Data/Attack/bundle formatting/" + bundle_types[0] + "/merge 20000_21000.txt"
target_file = "E:/Ethereum Data/Attack/label/" + bundle_types[0] + " merge label 20000_21000.txt"
data_file = "E:/Ethereum Data/Attack/bundle formatting/" + bundle_types[0] + "/merge all te.txt"
target_file = "E:/Ethereum Data/Attack/label/" + bundle_types[0] + " merge label all te.txt"
dexs = ["PancakeV3.txt", "SushiSwap.txt", "UniswapV2.txt", "UniswapV3.txt"]
# dexs = ["merge 20000_21000.txt"]
# data_file = "./data/gcn fine-turn data_5_class_20000_21000.txt"
# target_file = "./data/gcn fine-turn label_5_class_20000_21000.txt"
# data_file = "./data/gcn fine-turn data_5_class_20000_21000.txt"
# target_file = "./data/gcn fine-turn label_5_class_20000_21000.txt"
MAXNum = 50202
MinNum = 42671
MAXNum = 10
MinNum = 0
def process_files(types, input_files, data_file, target_file):
    count = [0]*len(types)
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
                            pairAddress_index, send_index, to_index, type, amount0, amount1, transactionHash = line.split()
                            data.append(
                                [pairAddress_index, send_index, to_index, type, amount0, amount1, transactionHash])
                            data.append(
                                [line])
                        else:
                            print(count)
                            if len(data) != 0 and count[i] < MAXNum:
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

                            # 只看一组交易的相对index
                # f_target.write(str(i) + '\n')

process_files(bundle_types, dexs, data_file, target_file)