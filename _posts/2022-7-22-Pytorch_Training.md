---
layout:     post
title:      19. GPU
subtitle:   深度学习
date:       2022-07-20
author:     Mo
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - Deepling  - CUDA  
---

# 1.为什么使用自动混合精度(**AMP**)

​	默认情况下，大多数深度学习框架都采用32位浮点算法进行训练.2017年，NVIDIA研究了一种用于混合精度训练的方法，该方法在训练网络时将单精度（FP32）与半精度(FP16)结合在一起，并使用相同的超参数实现了与FP32几乎相同的精度。

​	AMP其实就是Float32与Float16的混合，那为什么不单独使用Float32或Float16，而是两种类型混合呢？原因是：在某些情况下Float32有优势，而在另外一些情况下Float16有优势。这里先介绍下FP16：

**==FP16优势：==**

１．减少显存占用；

２．加快训练和推断的计算，能带来多一倍速的体验；

３．张量核心的普及（NVIDIA　Tensor Core）,低精度计算是未来深度学习的一个重要趋势。

**==FP16问题：==**

１．溢出错误；

２．舍入误差；

１．溢出错误：由于FP16的动态范围比FP32位的狭窄很多，因此，在计算过程中很容易出现上溢出和下溢出，溢出之后就会出现"NaN"的问题。在深度学习中，由于激活函数的梯度往往要比权重梯度小，更易出现下溢出的情况
![img](https://s2.loli.net/2022/07/22/rDGp5ceqfZnAN9M.png)

2.舍入误差：指的是当梯度过小时，小于当前区间内的最小间隔时，该次梯度更新可能会失败：

![img](https://s2.loli.net/2022/07/22/fop2TD9kEqaAZS4.png)

为了消除torch.HalfTensor也就是FP16的问题，需要使用以下两种方法：

1. 混合精度训练

	在内存中用FP16做储存和乘法从而加速计算，而用FP32做累加避免舍入误差。混合精度训练的策略有效地缓解了舍入误差的问题。

2. 损失放大（Loss scaling)

​	即使了混合精度训练，还是存在无法收敛的情况，原因是激活梯度的值太小，造成了溢出。可以通过使用torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的**underflow**（下溢出）（只在BP时传递梯度信息使用，真正更新权重时还是要把放大的梯度再unscale回去）；

​	反向传播前，将损失变化手动增大2^k倍，因此反向传播时得到的中间变量（激活函数梯度）则不会溢出；

​	反向传播后，将权重梯度缩小2^k倍，恢复正常值。





# 2.如何在PyTorch中使用自动混合精度

***使用==autocast + GradScaler==*** import torch.cuda.amp.autocast  && torch.cuda.amp.GradScaler

**autocast**：

```python
from torch.cuda.amp import autocast as autocast

# 创建model，默认是torch.FloatTensor
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # 前向过程(model + loss)开启 autocast
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # 反向传播在autocast上下文之外
    loss.backward()
    optimizer.step()
```

GradScaler：

```python
from torch.cuda.amp import autocast as autocast

# 创建model，默认是torch.FloatTensor
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)
# 在训练最开始之前实例化一个GradScaler对象
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # 前向过程(model + loss)开启 autocast
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()	# 计算梯度
        scaler.step(optimizer)# 调整lr
        scaler.update()# 更新梯度
         # optimizer.zero_grad()	 # 梯度清零(不涉及复杂计算，所以不需要GradScaler参与)
```





# 3.在Pytorch中的多GPU训练

​	Pytorch多GPU训练本质上是数据并行，每个GPU上拥有整个模型的参数，将一个batch的数据均分成N份，每个GPU处理一份数据，然后将每个GPU上的梯度进行整合得到整个batch的梯度，用整合后的梯度更新所有GPU上的参数，完成一次迭代。

其中多gpu训练的方案有两种，一种是利用`nn.DataParallel`实现，这种方法是最早引入pytorch的，使用简单方便，不涉及多进程。另一种是用`nn.DistributedDataParallel.` 使用多进程实现，第二种方式效率更高，[参考](https://tankzhou.cn/2019/07/07/Pytorch-分布式训练/)，但是实现起来稍难, 第二种方式同时支持多节点分布式实现。方案二的效率要比方案一高，即使是在单运算节点上。

#### **nn.DataParallel**：

​	(支持单机多卡)很容易使用，只需要加一行代码，但是**速度慢**（主要原因是它采用parameter server 模式，一张主卡作为reducer，负载不均衡，主卡成为训练瓶颈），在主GPU上进行梯度计算和更新，再将参数给其他gpu。

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  # 将batchsize 30 分配到N个GPU上运行
  model = nn.DataParallel(model)
model.to(device)
```

#### ==nn.DistributedDataParallel==

（支持单机多卡和多机多卡）采用All-reduce模式：

![img](https://pic3.zhimg.com/80/v2-eea741e6266065d858259d2ffdbb159a_720w.jpg)

​	**复制模型到多个GPU上，每个GPU通过一个进程来控制，进程之间互相通信，只有梯度信息是需要不同进程gpu之间通信，各个进程进行梯度汇总平均，然后传给其他进程，各个进程更新参数。所以瓶颈限制没有那么严重，为pytorch推荐的多GPU方式。**

​	**在训练时，每个进程/GPU load 自己的minibatch数据（所以要用distributedsampler), 每个GPU做自己独立的前向运算，反向传播时梯度all-reduce在各个GPU之间，各个节点得到平均梯度，保证各个GPU上的模型权重同步。**

​	多进程之间同步信息通信是通过 distributed.init_process_group实现，多进程DDP有几个相关概念：

1. group: 进程组 （通过 init_process_group() 对进程组初始化）
2. world_size: 总的进程数 （通过 get_world_size() 得到）
3. rank：当前进程号，主机master节点rank=0
4. local_*rank: 当前进程对应的GPU号 （通过get_rank() 可以得到每台机器上的local_rank*）

（ps：2台8卡机器进行DDP训练，init_process_*group() 初始化之后，get_world_size() 得到的是16， get_rank() 得到0-8*）

​	使用DDP的方式：

1. 使用 torch.distributed.init_process_group 初始化进程组
2. 使用 torch.nn.parallel.DistributedDataParallel 创建分布式并行模型
3. 创建对应的 DistributedSampler，制作dataloader
4. 使用 torch.multiprocessing/torch.distributed.launch 开始训练

​	注意的问题：

​	在使用DDP时，要给dataloader 传sampler参数（torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)） 。 默认shuffle=True，但按照pytorch DistributedSampler的实现：

```
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore
```

​	产生随机indix的种子是和当前的epoch有关，所以如果需要在每个epoch shuffle数据，就要在训练的时候手动set epoch的值来实现真正的shuffle：

```python
for epoch in range(start_epoch, n_epochs):
    if is_distributed:
        sampler.set_epoch(epoch)
    train(loader)
```

​	一种使用方式是直接使用multiprocessing:

```python
# 每个进程run一次train(i, args), i在（0 到 args.gpus-1）的范围。
def train(local_rank, args):
    rank = args.nodes * args.gpus + local_rank #得到全局rank  
    # 初始化进程，join 其他进程，pytorch docs解释nccl 通讯后台 backend 是最快的。  
    # https://pytorch.org/docs/stable/distributed.html 
    # torch.distributed init_method支持3种初始化方式，分别为 tcp(tcp://xx.xx.xx.xx:xxxx)、共享文件和环境变量初始化（env://）
    dist.init_process_group(                                   
        backend='nccl',# 不同的backend 用 NCCL 进行分布式 GPU 训练
用 Gloo 进行分布式 CPU 训练                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    )                                                          
    
    torch.manual_seed(0)#设置随机种子每个进程中，使得每个进程以同样的参数做初始化；
    model = model()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    batch_size = 100
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank])

    # Data loading code
    train_dataset = xxx    
    #train_sampler 使得每个进程得到不同切分的数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=args.num_workers,
       pin_memory=True,
       sampler=train_sampler)
    …



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run’)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = ‘xx.xx.xx.xx'              #
    os.environ['MASTER_PORT'] = ‘xxxx'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
```

​	另一种方式是使用torch.distributed.launch：

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(‘—-local_rank’, type=int, default=0)
    # 此方式下，init_process_group 中无需指定任何参数
    dist.init_process_group(backend='nccl')                                                          
    
    world_size = torch.distributed.get_world_size()
    torch.manual_seed(0)
    model = model()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    model.cuda(args.local_rank)
    batch_size = 100
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                              device_ids[args.local_rank])

    # Data loading code
    train_dataset = xxx    
    #train_sampler 使得每个进程得到不同切分的数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
       batch_size=batch_size,
       num_workers=args.num_workers,
       pin_memory=True,
       sampler=train_sampler)

    model.train()
    for i in range(1, EPOCHS + 1):
        train_loader.sampler.set_epoch(i)
        #...

# python -m torch.distributed.launch --nproc_per_node=2 main.py （2GPUS）
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="xx.xx.xx.xx" \
# --master_port=xxxxx main.py （2node 4GPUS）
```

