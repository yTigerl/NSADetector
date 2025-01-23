import matplotlib.pyplot as plt
import numpy as np

# acc.write(
#     f"{str(normalAccuracy)} {str(hybridAccuracy)} {str(bundleAccuracy)} {str(manyAccuracy)} {str(sasAccuracy)} {str(sssasAccuracy)} {str(ssasAccuracy)}\n")
def read_last_line(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        else:
            return "0 0"


# y1 = [[0.148,0.991,0.995,0.58,0.6641509433962264,0.0,0.2549019607843137],
# [0.088,0.117,0.212,0.112,0.7660377358490567,0.0,0.8235294117647058],
# [0.05,0.058,0.159,0.084,0.7811320754716982,0.0,0.9215686274509803],
# [0.026,0.027,0.138,0.069,0.8641509433962264,0.03571428571428571,0.9607843137254902],
# [0.018,0.018,0.122,0.054,1.0,0.07142857142857142,0.9803921568627451]
#       [],
#       [],
#       [],
#       [],
#       [],
#       [],
#       [],
#       [],
#       [],
#       [],
#
#       ]
y1 = [[], [], [], [], [], []]
y2 = [[], [], [], [], [], []]
x = np.arange(5, 11, 1)
y1 = [[], [], [], [], [], [], [], [], [], []]
y2 = [[], [], [], [], [], [], [], [], [], []]
# y1 = [[], [], []]
# y2 = [[], [], []]
x = np.arange(4, 8, 1)
y1 = [[], [], [], []]
y2 = [[], [], [], []]
threshold = 0.6
major = 6
count = 0
# f"{str(normalAccuracy)} {str(hybridAccuracy)} {str(bundleAccuracy)} {str(manyAccuracy)} {str(sasAccuracy)} {str(sssasAccuracy)} {str(ssasAccuracy)}\n")
testdir = "./Result/"
type_file_path = " model3GCN 5 class sample random new+old 1000.txt"
for major in np.arange(4, 8, 1):
# for major in np.arange(3.5, 6, 1):
    # thresholds.append(threshold)
    # print(float(major))、
    # major = float(major)
    file = str(threshold) + ' ' + str(major) + type_file_path
    print(file)
    sas_file = testdir + "sas incorrect" + file
    sssas_file = testdir + "sssas incorrect" + file
    ssas_file = testdir + "ssas incorrect" + file
    normal_file = testdir + "normal incorrect" + file
    hybrid_file = testdir + "hybrid incorrect" + file
    bundle_file = testdir + "bundle incorrect" + file
    many_file = testdir + "many incorrect" + file
    y1[count].append(int(read_last_line(normal_file).split()[0]))
    y1[count].append(int(read_last_line(hybrid_file).split()[0]))
    y1[count].append(int(read_last_line(bundle_file).split()[0]))
    y1[count].append(int(read_last_line(many_file).split()[0]))
    y2[count].append(int(read_last_line(sas_file).split()[0]))
    y2[count].append(int(read_last_line(sssas_file).split()[0]))
    y2[count].append(int(read_last_line(ssas_file).split()[0]))
    count += 1

print(y1)
print(y2)
acc1 = []
acc2 = []
for i in range(len(y1)):
    print(np.sum(y1[i]))
    acc1.append( 1 - np.sum(y1[i]) / (1000*4))
    acc2.append(1 - np.sum(y2[i]) / (80+23+19))

# acc2[-1] = acc2[-2]
print(acc2)

figure_width = 5  # 单栏宽度，单位为英寸
figure_height = 3  # 图像高度，单位为英寸
#
# # 设置画布大小与LaTeX中插入图像的大小相同
plt.figure(figsize=(figure_width, figure_height))
plt.rcParams['font.size'] = 14

plt.plot(x, acc1, '-', alpha=0.5, linewidth=3, color='c')
plt.plot(x, acc2, '-', alpha=0.5, linewidth=3, color='r')
# plt.plot(x3, y3, '-', alpha=0.5, linewidth=3, color='slateblue')
# plt.plot(x1, y1, '-', alpha=0.5, linewidth=3, color='r')
# plt.plot(x0, y0, '-', alpha=0.5, linewidth=3, color='b')


from pylab import *
#支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']

## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

plt.legend(['old', 'new'], frameon=False)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
# plt.legend(['Cyclic Arbitrage (ND)', 'Cyclic Arbitrage (DFS)', 'Delayed Arbitrage (DFS)'])
# plt.legend(['Cyclic Arbitrage', 'Delayed Arbitrage'])  # 显示上面的label
# plt.legend(['Bribery Slot N+3 Attesters', 'Bribery Slot N+2 Attesters'])

plt.xlabel('Major')  # x_label
plt.ylabel('Accuracy')  # y_label
# plt.xlabel('区块数', fontsize=16)  # x_label
# plt.ylabel('利润 (ETH)', fontsize=16)  # y_label
plt.tight_layout()

# plt.ylim(-5,250)#仅设置y轴坐标范围
# plt.savefig('./result/profits camparision.pdf')
# plt.savefig('./result/profits camparision.png', dpi=600)
plt.show()