#!/usr/bin/env python3

import numpy as np, torch, sys, random
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from game import Game, kW
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda')
kEnvs = 2048

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels.value, c.blocks.value).to(device)
    model.load_state_dict(torch.load(sys.argv[1])[0].state_dict())
    model.eval()
    envs = [Game(random.randint(0, 2**32-1)) for i in range(kEnvs)]
    finished = [False for i in range(kEnvs)]
    score = [0. for i in range(kEnvs)]
    while not all(finished):
        obs = []
        for i in envs: obs.append(i.obs)
        with torch.no_grad():
            obs = obs_to_torch(np.stack(obs))
            pi = model(obs)[0]
            act = torch.argmax(pi.probs, 1).cpu().numpy()
            #act = pi.sample().cpu().numpy()
        x, y = act // kW, act % kW
        tb = []
        for i in range(kEnvs):
            if finished[i]: continue
            _, _, over, info = envs[i].step((x[i], y[i]))
            if over:
                score[i] = info['score']
                finished[i] = True
    score = [(i, j) for j, i in enumerate(score)]
    score.sort()
    ds = [0, 0.01, 0.05] + [i * 0.1 for i in range(1, 10)] + [0.95, 0.99, 1 - 1e-5]
    print(' '.join(['%5.2f' % i for i in ds]))
    print(' '.join(['%5.1f' % score[int(i * kEnvs)][0] for i in ds]))
