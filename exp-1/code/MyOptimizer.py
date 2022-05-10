import torch

class MyOptimizer(object):
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad:
        # optimize the parameters
            for p in self.params:
                if p.grad is None:
                    continue
                # write your code here
                p.add_(-self.lr * p.grad)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class MyOptimizerAdam(MyOptimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999):
        self.params = list(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.state = dict()  # 设置state用于存储梯度和梯度平方的移动平均值

    def step(self):
        # optimize the parameters
        eps = 1e-8
        for p in self.params:
            if p.grad is None:
                continue
            # write your code here
            # 建议：
            # 可以在self.state中记录每个参数的总迭代次数、累加梯度、累加梯度的平方
            # 例如self.state[p] = dict() ...
            pass

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
