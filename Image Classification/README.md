# Image Classification


## 更新日志
**2023.5.27更新：**
* 新增绘制F1图和混淆矩阵图
* 优化代码
**2023.5.25更新：**
* 新建项目

## 文件说明
#### <pre>train.py</pre>
训练程序
> 可选参数：
```
--data_dir: 数据集路径，默认为根目录下的datasets文件
--epochs: 训练-验证集模式的迭代次数，默认为30
--k: K折交叉验证模式的K值，默认为10
--batch_size: 每个批次中包含的样本数量，默认为16
--optimizer: 优化器，可选['SGD', 'Adam', 'RMSprop']，默认为Adam
--lr: 学习率，可选['Fixed', 'Cosine']，默认为Fixed，具体数值在根目录下config.yaml中
--loss: 损失函数，可选['CrossEntropyLoss', 'NLLLoss', 'BCEWithLogitsLoss', 'BCELoss']，默认为CrossEntropyLoss
--num_workers: 数据载入线程数量，默认为cpu核心数和8之间最小值
--model: 网络类型，可选
    ['resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152']
--weights: 预训练权重，默认不设置，自动选择为使用网络的预训练权重
--no_transfer_learning: 是否使用迁移学习，默认为使用，调用则不使用迁移学习
--device: 设备类型，在支持cuda时使用gpu，否则使用cpu
--project: 训练完保存路径，默认为根目录下runs/train文件夹中
--name: 保存文件夹名称，默认为exp
```
训练模式
> 训练-验证模式
在datasets文件夹中将数据集按train、val分类，以train文件夹作为训练集，val文件夹作为验证集
> K折交叉验证模式
直接在datasets文件夹中添加数据集
#### <pre>detect.py</pre>
检测程序
可选参数：
```
--weights: 检测所用的模型
--source: 所要检测的图片或者文件夹
--data: 训练时的参数，默认为所指定weights的上一级目录的cfg.yaml文件
--indices: 类别标签，默认为所指定weights的上一级目录的class_indices.json文件
--device: 设备类型，在支持cuda时使用gpu，否则使用cpu
--project: 检测完保存路径，默认为根目录下runs/detect文件夹中
--name: 保存文件夹名称，默认为exp
```
#### <pre>config.yaml</pre>
有关训练的相关参数
#### <pre>models</pre>
网络实现
#### <pre>utils</pre>
辅助函数和工具函数