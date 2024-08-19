#!/usr/bin/env python3

import os
import sys
from collections import Counter

import numpy as np
import torch

from config import Configs
from game import Game, K_W
from model import Model, obs_to_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_ENVS = 10000

def main():
    try:
        start_seed = int(sys.argv[-1]) if len(sys.argv) >= 2 else 2234
        file_given = len(sys.argv) >= 3
    except ValueError:
        start_seed = 2234
        file_given = True

    c = Configs()
    model = Model(c.channels, c.blocks).to(device)

    model_path = sys.argv[1] if file_given else os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth')
    if model_path[-3:] == 'pkl':
        model.load_state_dict(torch.load(model_path)[0].state_dict())
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    envs = [Game(i + start_seed) for i in range(K_ENVS)]
    finished = [False for i in range(K_ENVS)]
    score = [0. for i in range(K_ENVS)]
    while not all(finished):
        obs = []
        for i in envs:
            obs.append(i.obs)
        with torch.no_grad():
            obs = obs_to_torch(np.stack(obs))
            pi = model(obs)[0]
            act = torch.argmax(pi.probs, 1).cpu().numpy()
            #act = pi.sample().cpu().numpy()
        x, y = act // K_W, act % K_W
        # tb = []
        for i in range(K_ENVS):
            if finished[i]:
                continue
            _, _, over, info = envs[i].step((x[i], y[i]))
            if over:
                score[i] = info['score']
                finished[i] = True
    score = sorted(list(dict(Counter(score)).items()))
    for i, j in score:
        print(i, j)
    #score = [(i, j) for j, i in enumerate(score)]

if __name__ == "__main__":
    main()
