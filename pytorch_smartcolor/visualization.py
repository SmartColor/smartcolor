from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


def visualization(filename):
    color_map = []  # [n,5] 其中每个元素是(r,g,b)元组

    # 没什么用，就是好看
    col_names = [str(i) + j for i in range(5) for j in ['r', 'g', 'b']]

    # 数据读取
    df = pd.read_csv(filename + '.csv', header=None, names=col_names)

    df = df[-100:]  # 取100条可视化

    for idx, row in df.iterrows():
        one_design = []
        for i in range(0, 15, 3):
            one_design.append((row[i], row[i + 1], row[i + 2]))
        color_map.append(one_design)

    # 画图
    fig, axs = plt.subplots(len(color_map), len(color_map[0]), figsize=(5, 100))
    for i, design in enumerate(color_map):
        for j, color in enumerate(design):
            img = Image.new('RGB', (3, 1), color)
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])  # 去掉坐标轴# ax.axis('off')
            ax.imshow(img)
    plt.savefig(filename + '.png')
    plt.show()

if __name__ == "__main__":
    visualization('rgb')