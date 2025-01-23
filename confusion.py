import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 初始化5x5混淆矩阵
confusion_matrix = np.zeros((5, 5), dtype=int)

total_samples_per_class = 7531
total_samples_per_class = 7532
def update_confusion_matrix(file_name, true_class):
    """
    读取每个错误记录文件，更新混淆矩阵。
    :param file_name: 错误记录文件名
    :param true_class: 该文件对应的真实类别
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()  # 读取所有行

        i = 0
        while i < len(lines):
            # 错误样本的索引 (在此处我们不使用)
            error_sample_index = lines[i].strip()  # 错误样本索引（我们不使用）

            # 错误总次数和预测类别在同一行
            error_info = lines[i + 1].strip().split()  # 分割成两部分
            error_count = int(error_info[0])  # 错误总次数
            predicted_class = int(error_info[1])  # 错误样本的预测类别

            # 更新混淆矩阵
            confusion_matrix[true_class][predicted_class] += 1

            # 移动到下一组错误记录（每组包括两行）
            i += 2

def complete_confusion_matrix():
    """
    补全混淆矩阵：补充每个类别的正确分类样本数。
    """
    # 每个类别的正确分类样本数
    for true_class in range(4):  # 类别0、1、2、3
        correct_classifications = total_samples_per_class - np.sum(confusion_matrix[true_class, :])  # 当前类别的正确分类数
        confusion_matrix[true_class, true_class] = correct_classifications

    # -1 类被视作所有错误分类的统一位置
    # confusion_matrix[4, :] = np.sum(confusion_matrix[:4, :], axis=0)  # 对于 -1 类，记录所有被误分类到 -1 类的样本
    # confusion_matrix[:, 4] = np.sum(confusion_matrix[:, :4], axis=1)  # 对于 -1 类，记录所有被预测为 -1 类的样本


def main():
    # 读取每个类别的错误记录并更新混淆矩阵
    # testdir = "./Test/"
    # type_file_path = " NSA_CRA 2GCN 64_128 1FC 128 4 class TEST 7531.txt"
    # threshold = 0.6
    # major = 0
    # file = str(threshold) + ' ' + str(major) + type_file_path
    #
    # testdir = "./Test/"
    # type_file_path = "2GCN 64_128 1FC 128 4 class TEST 7531.txt"
    # threshold = 0.75
    # major = 4
    # file = str(threshold) + ' ' + str(major) + type_file_path
    #
    # update_confusion_matrix(testdir + "LF incorrect" + file, 3)
    # update_confusion_matrix(testdir + "bundle incorrect" + file, 1)
    # update_confusion_matrix(testdir + "many incorrect" + file, 2)
    # update_confusion_matrix(testdir + "normal incorrect" + file, 0)


    testdir = "./CNN Test/"
    type_file_path = "2cnn 64_128 1FC 128 4 class_20000 TEST 7531.txt"
    threshold = 0.5
    major = 4
    file = str(threshold) + ' ' + str(major) + type_file_path

    update_confusion_matrix(testdir + "LF incorrect" + file, 3)
    update_confusion_matrix(testdir + "Non incorrect" + file, 1)
    update_confusion_matrix(testdir + "MLS incorrect" + file, 2)
    update_confusion_matrix(testdir + "CSA incorrect" + file, 0)

    complete_confusion_matrix()
    print(confusion_matrix)
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0", "1", "2", "3", "-1"],
                yticklabels=["0", "1", "2", "3", "-1"])
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()

    # 总样本数
    total_samples = confusion_matrix.sum()

    # 准确度计算
    accuracy = np.trace(confusion_matrix) / total_samples
    print(f"Accuracy: {accuracy:.4f}")

    # 计算每类的 Recall 和 Precision
    recall = []
    precision = []
    f1_scores = []
    precision_macro = 0
    recall_macro = 0
    f1_macro = 0
    for i in range(5):  # 类别 -1, 0, 1, 2, 3
        # Recall: 真实类别 i 的样本被正确预测的比例
        recall_i = confusion_matrix[i, i] / confusion_matrix[i, :].sum()
        # recall.append(recall_i)

        # Precision: 预测为类别 i 的样本中，真实为 i 的比例
        precision_i = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
        # precision.append(precision_i)

        precision_i = 0 if np.isnan(precision_i) else precision_i
        recall_i = 0 if np.isnan(recall_i) else recall_i

        # 计算 F1-score
        if precision_i + recall_i > 0:
            f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        else:
            f1_i = 0  # 如果 Precision 和 Recall 都为 0，则 F1-score 也为 0

        precision.append(precision_i)
        recall.append(recall_i)
        f1_scores.append(f1_i)
        precision_macro += precision_i
        recall_macro += recall_i
        f1_macro += f1_i
    # 输出每类的 Recall 和 Precision
    for i in range(5):
        print(f"Class {i} Recall: {recall[i]:.4f}, Precision: {precision[i]:.4f}, F1-score: {f1_scores[i]:.4f}")

    precision_macro /= 4
    recall_macro /= 4
    f1_macro /= 4
    print(f"Overall Precision (Macro): {precision_macro:.4f}")
    print(f"Overall Recall (Macro): {recall_macro:.4f}")
    print(f"Overall F1 (Macro): {f1_macro:.4f}")
    # # 计算每类的准确度
    # class_accuracies = []
    #
    # for i in range(5):  # 类别 -1, 0, 1, 2, 3
    #     # 正确分类为该类的样本数 (C[i, i])
    #     correct_classifications = confusion_matrix[i, i]
    #
    #     # 该类样本总数 (所有真实为类别 i 的样本数)
    #     total_class_samples = confusion_matrix[i, :].sum()
    #
    #     # 该类没有被错分到该类的样本数
    #     # 真实类别为 i 的样本总数 - 正确分类为该类的样本数 - 错误分类为该类的样本数
    #     misclassified_to_class_i = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
    #     not_misclassified_to_class_i = total_samples - total_class_samples - misclassified_to_class_i
    #
    #     # 计算准确度
    #     accuracy_i = (correct_classifications + not_misclassified_to_class_i) / total_samples
    #     class_accuracies.append(accuracy_i)
    #
    # # 输出每类的准确度
    # for i in range(5):
    #     print(f"Class {i} Accuracy: {class_accuracies[i]:.4f}")

    # 总样本数
    total_samples = confusion_matrix.sum()

    # 计算总体准确度
    accuracy = np.trace(confusion_matrix) / total_samples

    # 计算总体精度和召回率
    precision_macro = 0
    recall_macro = 0
    precision_weighted = 0
    recall_weighted = 0

    # for i in range(5):  # 类别 -1, 0, 1, 2, 3
    #     # 每一类的精度 (Precision) 和召回率 (Recall)
    #     precision_i = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]) if np.sum(
    #         confusion_matrix[:, i]) > 0 else np.nan
    #     recall_i = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) if np.sum(
    #         confusion_matrix[i, :]) > 0 else np.nan
    #
    #     # 累积计算宏平均
    #     precision_macro += precision_i
    #     recall_macro += recall_i
    #
    #     # 加权计算精度和召回率
    #     precision_weighted += np.sum(confusion_matrix[i, :]) * precision_i
    #     recall_weighted += np.sum(confusion_matrix[i, :]) * recall_i
    #
    # # 宏平均计算
    # precision_macro /= 5
    # recall_macro /= 5
    #
    # # 加权平均计算
    # precision_weighted /= total_samples
    # recall_weighted /= total_samples
    #
    # # 输出结果
    # print(f"Overall Accuracy: {accuracy:.4f}")
    # print(f"Overall Precision (Macro): {precision_macro:.4f}")
    # print(f"Overall Recall (Macro): {recall_macro:.4f}")
    # print(f"Overall Precision (Weighted): {precision_weighted:.4f}")
    # print(f"Overall Recall (Weighted): {recall_weighted:.4f}")


if __name__ == "__main__":
    main()
