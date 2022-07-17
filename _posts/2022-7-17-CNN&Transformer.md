---
layout:     post
title:      17. CNN & Transformer
subtitle:   深度学习
date:       2022-07-17
author:     Mo
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - Deepling  
---

# CNN



1、卷积层本质上是个特征抽取层，应用在NLP问题中，它捕获到的是单词的n-gram片段信息。

2、CNN存在的问题：

  （1）单卷积层无法捕获远距离特征；可以通过加深网络的层数获得更大的感受野，然而实际情况是CNN做NLP问题就是做不深，做到2到3层卷积层就做不上去了；

  （2）还有一个问题，就是Max Pooling层，无法保持输入句子中单词的位置信息。

​    why？

​       1）RNN因为是线性序列结构，所以很自然它天然就会把位置信息编码进去；

​       2）CNN的卷积核是能保留特征之间的相对位置的，滑动窗口从左到右滑动，捕获到的特征也是如此顺序排列，所以它在结构上已经记录了相对位置信息了；

​       3）但是如果卷积层后面立即接上Pooling层的话，Max Pooling的操作逻辑是：从一个卷积核获得的特征向量里只选中并保留最强的那一个特征，所以到了Pooling层，位置信息就被扔掉了。

  目前的趋势是抛弃Pooling层。

3、CNN的并行度是非常自由也非常高的，这是CNN的一个优点。





# **Visual Transformer 具有如下较好的特性**：

1. **==Long Range 带来的全局特性==**。一方面，**CNN 的 Conv 算子存在感受野较局限的问题**，为扩大网络的关注区域，需多层堆叠卷积-池化结构，但随之带来的问题是 **“有效/真实” 感受野以某个中心为原点向外高斯衰减**，因此 **CNN 通常的有效 Attention 就是图中某一两个较重要的 Parts**。为解决该问题，可设计用于 CNN 的 Attention Module 来得到感受野范围更大而均衡的 Attention Map，其有效性得到了很多工作的证明。另一方面，**Transformer 天然自带的 Long Range 特性 (自注意力带来的全局感受野) 使得从浅层到深层特征图，都较能利用全局的有效信息**，**并且 Multi-Head 机制保证了网络可关注到多个 Discriminative Parts (每个 Head 都是一个独立的 Attention)**，这是 Transformer 与 CNN主要区别之一。
2. **==更好的多模态融合能力==**。一方面，**CNN 擅长解构图像的信息**，因为 **CNN 卷积核卷积运算的本质就是传统数字图像处理中的滤波操作**，但这也使得 **CNN 不擅长融合其他模态的信息**，例如文本、标签、语音、时间等。通常需要用 CNN 提取图像特征 (Feature Embedding)，再用其他模型对其他信息进行 Embedding (如文本的 Token Embedding)，最后在网络末端融合多模态的 Embeddings (**后融合**)。另一方面，Transformer 可在网络的输入端融合多模态信息 (**前融合**)，例如，对于图像，可把对图像通过 Conv 或直接对像素操作得到的初始 Embeddings 馈入 Transformer 中，而 **无需始终保持 H×W×C 的 Feature Map 结构**。类似于 Position Embedding，只要能编码的信息，都可以非常轻松地利用进来。
3. **==Multiple Tasks 能力==**。不少工作证明一个 Transformer 可执行很多任务，因为其 Attention 机制可让网络对不同的 Task 进行不同的学习，一个简单的用法便是加一个 Task ID 的 Embedding。
4. **==更好的表征能力==**。不少工作显示 Transformer 可在多个 CV 任务上取得 SOTA 结果。



# CNN & Transformer

1. ==CNN 是通过不断地堆积卷积层来完成对图像从局部信息到全局信息的提取，不断堆积的卷积层慢慢地扩大了感受野直至覆盖整个图像。==但 ==Transformer== 并不假定从局部信息开始，而是 **==一开始就可以拿到全局信息==，学习难度更大一些，但 ==Transformer 学习长依赖的能力更强==**。另外，从 ViT 的分析来看，**前面层的 “感受野” (论文里是 Mean Attention Distance) 虽迥异但总体较小，后面层的 “感受野“ 越来越大，这说明 ViT 也学习到了和 CNN 相同的范式**。没有 “受限” 的 Transformer 一旦完成好学习，势必会发挥自己这种优势。

2. CNN 对图像问题有天然的 Inductive Bias，如平移不变性等以及 CNN 的仿生学特性，这让 CNN 在图像问题上更容易；相比之下，**Transformer 没有这个天然优势**，**那么学习的难度很大**，往往需要 **更大的数据集 (ViT)** 或 **更强的数据增强 (DeiT)** 来达到较好的训练效果。好在 **Transformer 的迁移效果更好**，大数据集上的 Pretrained 模型可以很好地迁移到小数据集上。还有一个就是 ViT 所说的，==Transformer 的 **Scaling 能力** 很强==，那么进一步 **提升参数量** 或许会带来更好的效果 (就像惊艳的GPT模型)。

3. 目前我们还看到很大一部分工作还是把 Transformer 和现有的 CNN 工作结合在一起，如 ViT 其实也是有 **Hybrid Architecture** (将由 ResNet 提取的特征图馈入 ViT 而非原始图像)。而对于检测和分割这类问题，CNN 方法已经很成熟，难以用 Transformer 一下子完全替换掉。目前的工作大都是 **CNN 和 Transformer 的混合体**，这其中有速度和效果的双重考虑。另外，也要考虑到 **如果输入较大分辨率的图像，Transformer 的计算量会很大，所以 ViT 的输入并不是 Pixel，而是小 Patch**。对于 DETR，其 Transformer Encoder 的输入是 1/32 特征 都有计算量的考虑，不过这肯定是有效果的影响，所以才有后面的改进工作 Deformable DETR。**短期内，CNN 和 Transformer 仍会携手同行**。论文 Rethinking Transformer-based Set Prediction for Object Detection 还是把现有的 CNN 检测模型和 Transformer 思想结合在一起实现了比 DETR 更好的效果 (训练收敛速度也更快)：



# Swin Transformer

1. 之前的ViT中，由于 Self-attention 是全局计算的，所以在图像分辨率较大时不太经济。由于 Locality 一直是视觉建模里非常有效的一种 Inductive Bias，所以我们将图片切分为 **无重合的 Windows**，然后在 Local Window 内进行 Self-attention 的计算。为了让 Windows 之间有信息交换，我们在相邻两层使用不同的 Windows 划分 (Shifted Window)。
2. 图片中的物体大小不一，而 ViT 中使用固定的 Scale 进行建模或许对下游任务例如目标检测而言不是最优的。在这里我们还是 Follow 传统 CNN 构建了一个 **层次化的 Transformer 模型**，从 4x 逐渐降分辨率到 32x，这样也可以在任意框架中无缝替代之前的 CNN 模型。