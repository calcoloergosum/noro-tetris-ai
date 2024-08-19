import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.distributions import Categorical

from game import K_H, K_W

DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)


# board, valid, current(7), next(7), score
K_IN_CHANNEL = 1 + 1 + 7 + 7 + 1

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)

class Model(nn.Module):
    def __init__(self, ch, blk):
        super().__init__()
        self.start = nn.Sequential(
                nn.Conv2d(K_IN_CHANNEL, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                )
        self.res = nn.Sequential(*[ConvBlock(ch) for i in range(blk)])
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(ch, 4, 1),
                nn.BatchNorm2d(4),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(4 * K_H * K_W, K_H * K_W)
                )
        self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1),
                nn.BatchNorm2d(1),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(1 * K_H * K_W, 256),
                nn.ReLU(True),
                nn.Linear(256, 1)
                )

    @autocast(device_type=DEVICE_TYPE)
    def forward(self, obs: torch.Tensor):
        q = torch.zeros((obs.shape[0], K_IN_CHANNEL, K_H, K_W), dtype = torch.float32, device = obs.device)
        q[:,0:2] = obs[:,0:2]
        q.scatter_(1, (2 + obs[:,2,0,0].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, K_H, K_W), 1)
        q.scatter_(1, (9 + obs[:,2,0,1].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, K_H, K_W), 1)
        q[:,16] = (obs[:,2,0,2] / 32).view(-1, 1, 1)
        x = self.start(q)
        x = self.res(x)
        pi = self.pi_logits_head(x)
        if self.training:
            pi -= (1 - obs[:,1].view(-1, K_H * K_W)) * 20
        else: pi[obs[:,1].view(-1, K_H * K_W) == 0] = -30
        value = self.value(x).reshape(-1)
        pi_sample = Categorical(logits = torch.clamp(pi, -30, 30))
        return pi_sample, value

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.uint8, device = DEVICE)
