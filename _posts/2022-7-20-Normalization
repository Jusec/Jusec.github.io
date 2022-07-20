---
layout:     post
title:      18. Normalization
subtitle:   深度学习
date:       2022-07-20
author:     Mo
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - Deepling   
---

# Normalization(归一化层)

***Normalization的目的是为了把输入转化成均值为0方差为1的数据。***

归一化层：**Batch Normalization**(2015年)、**Layer Normalization**(2016年)、**Instance Normalization**(2017年)、**Group Normalization**(2018年)、**Switchable Normalization**(2018年)

![img](https://s2.loli.net/2022/07/20/n6YfPkLpRtFo2hv.jpg)

> 输入数据Shape：[N, C, H, W]:
>
> 1. ==Batch Norm==是在**batch**上，对NHW做归一化，就是对每个单一通道输入进行归一化，这样做对小batchsize效果不好；(==BN取的是不同样本的同一个特征==)
> 2. ==Layer Norm==在**通道方向**上，对CHW归一化，就是对每个深度上的输入进行归一化，主要对RNN作用明显；(==LN取的是同一个样本的不同特征==)
> 3. ==Instance Norm==在图像像素上，对HW做归一化，对一个图像的长宽即对一个像素进行归一化，用在风格化迁移；
> 4. ==Group Norm==将**channel分组**，有点类似于LN，只是GN把channel也进行了划分，细化，然后再做归一化；（**GN介于LN和IN之间，其首先将channel分为许多组（group），对每一组做归一化，及先将feature的维度由[N, C, H, W]reshape为[N, G，C//G , H, W]，归一化的维度为[C//G , H, W]**）
> 5. ==Switchable Norm==是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

#### 1.Batch Norm

​	可以理解为是一种数据预处理技术，使得每层网络的输入都服从（0，1）0均值，1方差分布，如果不进行BN，那么每次输入的数据分布不一致，网络训练精度自然也受影响。

​	Batch Norm即批规范化，目的是为了解决每批数据训练时的不规则分布给训练造成的困难，对批数据进行规范化，还可以在梯度反传时，解决梯度消失的问题。BN的提出，就是要解决在训练过程中，中间层数据分布发生改变的情况。



​	神经网络在训练的时候随着网络层数的加深,激活函数的输入值的整体分布逐渐往激活函数的取值区间上下限靠近,从而导致在反向传播时低层的神经网络的梯度消失。而Batch Normalization的作用是==通过规范化的手段,将越来越偏的分布拉回到标准化的分布,使得激活函数的输入值落在激活函数对输入比较敏感的区域,从而使梯度变大,加快学习收敛速度,避免梯度消失的问题。==

①不仅仅极大提升了训练速度，收敛过程大大加快；②还能增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式，所以不用Dropout也能达到相当的效果；③另外调参过程也简单多了，对于初始化要求没那么高，而且可以使用大的学习率等。



​	Batch Norm也是一种正则的方式，可以代替其他正则方式如dropout，但通过这样的正则化，也消融了数据之间的许多差异信息。

> 优点：
>
> 1. 增加了模型的==泛化能力==，一定程度上取代了之前的Dropout;
> 2. ==减轻了模型对参数初始化的依赖，且可加快模型训练，提高模型精度==。
>
> 不足：
>
> 1. Batch Normalization中batch的大小，会影响实验结果，主要是因为小的batch中计算的均值和方差可能与测试集数据中的均值与方差不匹配；
> 2. 难以用于RNN。以 Seq2seq任务为例，同一个batch中输入的数据长短不一，不同的时态下需要保存不同的统计量，无法正确使用BN层，只能使用Layer Normalization。

> 关于Normalization的有效性，主流观点有：
>
> （1）主流观点，Batch Normalization调整了数据的分布，不考虑激活函数，它让每一层的输出归一化到了均值为0方差为1的分布，这保证了梯度的有效性，目前大部分资料都这样解释，比如BN的原始论文认为的缓解了Internal Covariate Shift(ICS)问题。
>
> （2）==可以使用更大的学习率==，文【3】指出BN有效是因为用上BN层之后可以使用更大的学习率，从而跳出不好的局部极值，增强泛化能力，在它们的研究中做了大量的实验来验证。
>
> （3）==损失平面平滑==。文【4】的研究提出，BN有效的根本原因不在于调整了分布，因为即使是在BN层后模拟ICS，也仍然可以取得好的结果。它们指出，==BN有效的根本原因是平滑了损失平面==。之前我们说过，Z-score标准化对于包括孤立点的分布可以进行更平滑的调整。



​	Batch Norm的算法流程：

![img](https://s2.loli.net/2022/07/20/r9QxCs7t1fND5nB.jpg)

```python
import numpy as np

def Batchnorm(x, gamma, beta, bn_param):

    # x_shape:[B, C, H, W]
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    results = 0.
    eps = 1e-5
    
	# 1.沿着通道计算每个batch的均值
    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    # 2.沿着通道计算每个batch的方差
    x_var = np.var(x, axis=(0, 2, 3), keepdims=True)
    # 3.对x做归一化
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    # 4.加入缩放和平移变量γ和β ,归一化后的值
    results = gamma * x_normalized + beta

    # 因为在测试时是单个图片测试，这里保留训练时的均值和方差，用在后面测试时用
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return results, bn_param
```

​	==加入缩放(γ)和平移变量(β)的原因是：**保证每一次数据经过归一化后还保留原有学习来的特征**，同时又能完成归一化操作，加速训练。 这两个参数是用来学习的参数。BN中的可学习参数（重构参数）：缩放(γ)和平移变量(β)==

​	==缺点==：需要注意的是在使用==小batch-size==时BN会破坏性能，当具有==分布极不平衡二分类任务==时也会出现不好的结果。因为如果小的batch-size归一化的原因，使得原本的数据的均值和方差偏离原始数据，均值和方差不足以代替整个数据分布。分布不均的分类任务也会出现这种情况！

​	BN实际使用时需要计算并且保存某一层神经网络batch的均值和方差等统计信息，对于对一个固定深度的前向神经网络（DNN，CNN）使用BN，很方便；但对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，不同的time-step需要保存不同的statics特征，可能存在一个特殊sequence比其他sequence长很多，这样training时，计算很麻烦。（不适合原因：**Normalize的对象(position)来自不同分布**）（BN不适合RNN，但RNN还是可以用BN的，只需要让每个Batch的长度相等，可以通过对每个序列做补长，截断来实现）



​	BN一般用在网络的哪个部分？==先卷积再BN==



​	Batch Norm训练时和测试时的区别:

​	训练时，我们可以对每一批的训练数据进行归一化，计算每一批数据的均值和方差。

​	但在测试时，比如进行一个样本的预测，就并没有batch的概念，因此，这个时候用的均值和方差是全量训练数据的均值和方差：使用了BN的网络，在训练完成之后，我们会保存全量数据的均值和方差，每个隐层神经元也有对应训练好的Scaling参数和Shift参数。

​	

​	先加BN还是激活，有什么区别（先激活）？目前在实践上，倾向于把BN放在ReLU后面。也有评测表明BN放ReLU后面效果更好。



​	BN层反向传播，怎么求导?

![image-20220429205618226](https://s2.loli.net/2022/07/20/wsi23Ef9lzJO4xD.png)

#### 2.Layer Normalization

与BN不同的是，LN对每一层的所有神经元进行归一化，与BN不同的是：

​	LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；

​	BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差。

​	LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。

一般情况，LN常常用于RNN网络！

![img](https://s2.loli.net/2022/07/20/cki7P6UZ38wm1T9.jpg)

```python
def Layernorm(x, gamma, beta):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```



#### 3.Instance Normalization

​	BN注重对每一个batch进行归一化，保证数据分布的一致，因为判别模型中的结果取决与数据的整体分布。在图像风格中，生成结果主要依赖某个图像实例，所以此时对整个batch归一化不适合了，需要对但像素进行归一化，可以加速模型的收敛，并且保持每个图像实例之间的独立性。

![img](https://s2.loli.net/2022/07/20/ZtWPhLO4NQa7TBs.png)

> IN本身是一个非常简单的算法，==尤其适用于批量较小且单独考虑每个像素点的场景中==，因为其计算归一化统计量时没有混合批量和通道之间的数据，对于这种场景下的应用，我们可以考虑使用IN。
>
> 另外需要注意的一点是在图像这类应用中，每个通道上的值是比较大的，因此也能够取得比较合适的归一化统计量。
>
> 但是有两个场景建议不要使用IN:
>
> 1. MLP或者RNN中：因为在MLP或者RNN中，每个通道上只有一个数据，这时会自然不能使用IN；
> 2. Feature Map比较小时：因为此时IN的采样数据非常少，得到的归一化统计量将不再具有代表性。

```python
def Instancenorm(x, gamma, beta):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```



#### 4.Group Normalization

主要是针对Batch Normalization对小batchsize效果差，GN将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值，这样与batchsize无关，不受其约束。==GN解决了BN式归一化对batch size依赖的影响。==

```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
```

#### 5.Switchable Normalization

![img](https://s2.loli.net/2022/07/20/XQcI1gRkLseGKBu.jpg)

第一，归一化虽然提高模型泛化能力，然而归一化层的操作是人工设计的。在实际应用中，解决不同的问题原则上需要设计不同的归一化操作，并没有一个通用的归一化方法能够解决所有应用问题；

第二，一个深度神经网络往往包含几十个归一化层，通常这些归一化层都使用同样的归一化操作，因为手工为每一个归一化层设计操作需要进行大量的实验。

因此作者提出自适配归一化方法——Switchable Normalization（SN）来解决上述问题。与强化学习不同，SN使用可微分学习，为一个深度网络中的每一个归一化层确定合适的归一化操作。

> SN具有以下三个有点：
>
> 1. ==鲁棒性==：无论batchsize的大小如何，SN均能取得非常好的效果；（SN也能根据batchsize的大小自动调整不同归一化策略的比重，如果batchsize的值比较小，SN学到的BN的权重就会很小，反之BN的权重就会很大）
> 2. ==通用性==：SN可以直接应用到各种类型的应用中，减去了人工选择归一化策略的繁琐；（SN通过根据不同的任务调整不同归一化策略的权值使其可以直接应用到不同的任务中。）
> 3. ==多样性==：由于网络的不同层在网络中起着不同的作用，SN能够为每层学到不同的归一化策略，这种自适应的归一化策略往往要优于单一方案人工设定的归一化策略。

![img](https://s2.loli.net/2022/07/20/omUeZx2fSC56pMc.jpg)

![image-20220720181316496](https://s2.loli.net/2022/07/20/FKrqWDubjXQMNpC.png)

![img](https://s2.loli.net/2022/07/20/aLpYyTJkUMD8Zl5.jpg)

```python
def SwitchableNorm(x, gamma, beta, w_mean, w_var):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    mean_in = np.mean(x, axis=(2, 3), keepdims=True)
    var_in = np.var(x, axis=(2, 3), keepdims=True)

    mean_ln = np.mean(x, axis=(1, 2, 3), keepdims=True)
    var_ln = np.var(x, axis=(1, 2, 3), keepdims=True)

    mean_bn = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var_bn = np.var(x, axis=(0, 2, 3), keepdims=True)

    mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
    var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn

    x_normalized = (x - mean) / np.sqrt(var + eps)
    results = gamma * x_normalized + beta
    return results
```