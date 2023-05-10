import matplotlib.pyplot as plt
import numpy as np


def show_color(color_bar_target, color_bar_predict, save_path):
    '''
    把数据集中的target颜色与模型预测出来的颜色绘制出来，做一个对比，上面一条是target，下面一条是predict
    
    params
    ------
        color_bar_target: 期望的颜色，list类型
        color_bar_predict: 预测出的颜色，list类型
        save_path: 绘制出的图片存储位置
    '''
    plt.cla()
    color_bar = np.concatenate((color_bar_target[None ,...], color_bar_predict[None, ...]), axis = 0)
    print(color_bar.shape)
    color_bar = np.clip(color_bar, 0, 255)
    n = len(color_bar[0])
    plt.yticks([0,1], labels = ["tgt_color", "pdt_color"])
    plt.xticks(range(n))
    plt.imshow(color_bar)
    plt.savefig(save_path)

def show_loss(loss_trace, save_path):
    '''
    绘制模型训练过程中的loss曲线
    
    params
    ------
        loss_trace: 训练过程中累积的loss数据，list类型
    '''
    plt.cla()
    plt.plot(loss_trace)
    plt.savefig(save_path)
