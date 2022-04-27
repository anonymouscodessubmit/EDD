import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    #使用softmax参数对op的输出进行加权求和
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  #def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
  #cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    #由于每个cell有两个input边，因此reduction后一个节点除了考虑直接输入外，还需要下采样一下reduction cell前一个cell的输出
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
	#把输入都处理为C通道
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    #每个Cell的内部节点数量为4
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    #step默认为4,i=0,j=0-1,2;i=1,j=0-2,3;i=2,j=0-3,4;i=3,j=0-4,5
	#2+3+4+5=14
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      #def forward(self, x, weights)
      #return sum(w * op(x) for w, op in zip(weights, self._ops))
      #cell中每个mixop输出是所有op的softmax加权求和
      #一个节点对应多个输入mixop，节点的feature是所有mixop的sum直接求和
      #这里或许可以修改为加权求和，甚至自动分离新节点操作
      #整体softmax或者是两次softmax或者使用对数和增强softmax梯度
      #对于权重差异不大的操作，直接使用节点分离法而不是prune
      #一共4个step和14个mixop
      #第一个node时len(states)=2，因此sum为列表前两行的softmax
      #第二个node的len(states)=3，sum列表为前三行softmax结果
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    #默认multiplier为4,把cell中4个step的输出全部cat到一起，作为整个cell的输出
    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C	#16*3=48
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #48,48,16
    self.cells = nn.ModuleList()
    reduction_prev = False
    #layers默认为8
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr #48,16*4=64

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, inputd,temp=1):
    s0 = s1 = self.stem(inputd)
    weights_r=F.softmax(self.alphas_reduce/temp, dim=-1)
    weights_n=F.softmax(self.alphas_normal/temp, dim=-1)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        #weights = F.softmax(self.alphas_reduce/temp, dim=-1)
        s0, s1 = s1, cell(s0, s1, weights_r)
      else:
        #weights = F.softmax(self.alphas_normal/temp, dim=-1)
        s0, s1 = s1, cell(s0, s1, weights_n)
      #s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits,weights_n

  def _loss(self, inputd, target,temp=1):
    #forward(inputd)
    logits,alphas = self(inputd,temp)
    return self._criterion(logits, target),alphas

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
	#step==4,k==14,k=2+3+4+5,对应4个node
    num_ops = len(PRIMITIVES)

    #标准正态初始化arch参数
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self,temp=1):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W_o = np.around(weights[start:end].copy(),decimals=3)
        #W = W_o[::-1]
        W = W_o
		#将0到i+2按照W[x][k]中的取值进行排序，选择其中最大的两个值
        #每个node选择step中softmax出来两个最大的连接,因此每个中间node有两个前向连接
        #edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        #print(edges,end=";")
        #edges=[len(W)-1-i for i in edges]
        #print(edges)
        W=W_o
        #在选中的边上再挑出softmax最大的
        for j in edges: 
          k_best = None
          for k in range(len(W[j])):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    with torch.no_grad():
        gene_normal = _parse(F.softmax(self.alphas_normal/temp, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce/temp, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

