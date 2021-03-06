import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import math


def _concat(xs):
  #把每层参数直接展开成行向量,然后再cat在一起成为一个巨大的行向量
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, inputd, target, eta, network_optimizer):
    loss = self.model._loss(inputd, target)
    theta = _concat(self.model.parameters()).data
    try:
	  #momentum默认0.9,从优化器中获得冲量
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
	#decay默认3e-4,将网络梯度加上权重衰减
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch,unrolled=False,temp=1,warmup=0):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid,epoch,temp,warmup)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid,epoch,temp=1,warmup=0):
    outgrad=0
    def softmaxOutGrad(grad):
        nonlocal outgrad
        outgrad=grad
    loss, alphas= self.model._loss(input_valid, target_valid,temp)
    handle=alphas.register_hook(softmaxOutGrad)
    loss.backward()
    handle.remove()
    #print(self.model.alphas_normal.grad)
    #self.optimizer.zero_grad()
    #if epoch>=warmup:
    if not math.isclose(temp,1.0):
        alphas_smooth=torch.nn.functional.softmax(self.model.alphas_normal,dim=-1)
        #alphas_smooth=torch.nn.functional.softmax(self.model.alphas_normal/(temp*100),dim=-1)
        alphas_smooth.backward(outgrad)
        #pass
        #print(self.model.alphas_normal.grad)
    #exit(0)

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, inputd, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(inputd, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(inputd, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

