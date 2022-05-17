import torch


class MyOptimizer(object):
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
    
    @torch.no_grad()
    def step(self):
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

    @torch.no_grad()
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
            if not p in self.state:
                st = dict()
                st['COUNT'] = 0
                st['SUM_dP'] = torch.zeros_like(p)
                st['SUM_SQRT_dP'] = torch.zeros_like(p)
                self.state[p] = st
            
            st = self.state[p]
            count = st['COUNT'] + 1
            sum_d_p = st['SUM_dP']
            sum_sqt_d_p = st['SUM_SQRT_dP']
            sum_d_p.mul_(self.b1).add_((1-self.b1) * p.grad)
            sum_sqt_d_p.mul_(self.b2).addcmul_((1-self.b2) * p.grad, p.grad)

            momentum = sum_d_p / (1 - self.b1 ** count)
            sigma = sum_sqt_d_p / (1 - self.b2 ** count)
            p.add_(-self.lr * momentum / (sigma.sqrt() + eps))
            st['COUNT'] = count

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
