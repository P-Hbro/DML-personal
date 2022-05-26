import math
import torch
from torch.utils.data import Dataset, Sampler


class MySampler_segementation(Sampler):
    def __init__(self, dataset: Dataset, num_replicas, rank, shuffle=True, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas  # 共有多少client参与
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed  # 把seed设置成client的rank，避免生成完全相同的随机数列
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        # 在random sampling 中，
        # num_samples 是一个client会获得的indices长度，也就是分得的样本数量
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Write your code here!
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        """
            example:
                indices=list(range(len(self.dataset)))
                return iter(indices)
        """
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class MySampler_sample(Sampler):
    def __init__(self, dataset: Dataset, num_replicas, rank, shuffle=True, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas  # 共有多少client参与
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed  # 把seed设置成client的rank，避免生成完全相同的随机数列
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        # 在random sampling 中，
        # num_samples 是一个client会获得的indices长度，也就是分得的样本数量

    def __iter__(self):
        # Write your code here!
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        t = torch.Generator()
        t.manual_seed(self.seed)
        samples = torch.randint(0, len(self.dataset), [self.num_samples], generator=t).tolist()
        indices = indices[samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples