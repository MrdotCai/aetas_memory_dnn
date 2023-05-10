import torch
import torch.nn as nn
from torch import optim
import numpy as np
import math

import dataset
import model
import visualize

def main():

    save_path = "./models/net.ptl"
    net = model.MemoNet()
    # net_state_dict = torch.load(save_path)
    # net.load_state_dict(net_state_dict)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    batch_size = 10
    epochs = 2000
    loss_func = nn.MSELoss()
    net.train()
    loss_trace = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch_cnt = 0
        train_set = dataset.train_batch_generator(batch_size)
        for batch_data, batch_label, batch_color in train_set:
            optimizer.zero_grad()
            predict_color = net(torch.tensor(batch_data.astype(np.float32)))
            loss = loss_func(predict_color, torch.tensor(batch_color.astype(np.float32)))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_cnt % 10 == 0:
                gap_loss = math.sqrt(running_loss / 10)
                print('epoch: %d, batch: %3d loss: %.3f' % (epoch, batch_cnt, gap_loss))
                loss_trace.append(gap_loss)
                running_loss = 0.0
            batch_cnt += 1
        
        if epoch % 100 == 0:
            torch.save(net.state_dict(), save_path)

    print("train finished.")
    visualize.show_loss(loss_trace, "./show_loss2.png")


if __name__=="__main__":
    main()
    