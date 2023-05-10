import model_c
import torch
import dataset
import numpy as np
import visualize

save_path = "./models/net_c.ptl"

net = model_c.MemoNet()
net_state_dict = torch.load(save_path)
net.load_state_dict(net_state_dict)

# train_set = dataset.train_batch_generator()

# batch_data, batch_label, batch_color = next(train_set)

mnist_data = dataset.data_generate()
batch_data = mnist_data.train.data[-100:]
batch_label = mnist_data.train.target[-100:]
batch_color = mnist_data.train.color[-100:]

n = len(batch_data)

rcd = {
    0: [[],[]],
    1: [[],[]],
    2: [[],[]],
    3: [[],[]],
    4: [[],[]],
    5: [[],[]],
    6: [[],[]],
    7: [[],[]],
    8: [[],[]],
    9: [[],[]],
}

for i in range(n):
    data = batch_data[i][None, ...]
    label = batch_label[i]
    color_target = batch_color[i]
    color_predict = net(torch.tensor(data.astype(np.float32)), label[None, ...])
    color_predict = np.array(list(map(int, color_predict.detach().numpy()[0])))
    rcd[label][0].append(color_target)
    rcd[label][1].append(color_predict)

for digit in rcd:
    visualize.show_color(np.array(rcd[digit][0]), np.array(rcd[digit][1]), "pics_c/_showdigit"+str(digit)+".png")