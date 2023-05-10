#### Description
用一个简单的CNN模型来预测(图片->颜色)的对应关系。使用的数据集是MNIST，预先将MNIST图片与色环上的颜色产生对应关系，然后训练网络去过拟合记住这种对应关系。
模型1：用图片来预测颜色，即input:图片，output: 颜色；
模型2: 用图片与其所属的类来预测颜色，即input:(图片，类别)，output: 颜色。
对比这两种模型的差异。

#### Contents
```
|--data_set/            原生数据集
|--methods/             封装了一些方法，包括读取数据集的
|--models/              训练好的模型文件
|--pics/                预测时保存的一些图片
|--pics_c/              带类输入的模型保存的一些图片
|--README.md            本文件
|--dataset.py           提供数据集接口
|--model_c.py           带类输入的模型文件
|--model.py             不带类输入的模型文件
|--preidct_c.py         带类输入的模型预测
|--predict.py           不带类输入的模型预测
|--train_c.py           带类输入的模型训练
|--train.py             不带类输入的模型训练
|--visualize.py         一些可视化函数
```

#### Usage
- 训练模型
```shell
# 训练不带类输入的模型
python3 train.py
# 训练带类输入的模型
python3 train_c.py
```

- 预测模型
```shell
python3 predict.py
```
