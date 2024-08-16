import math
import numpy as np
import torch
import gym
from gym import spaces
from gym.utils import seeding
from ..utils import validate_CV


class HeadList(torch.nn.Module):
    def __init__(self, heads):
        super(HeadList, self).__init__()
        self.heads = torch.nn.ModuleList(heads)
    def forward(self, x):
        y = [ h(x) for h in self.heads ]
        return y

    
class Env_CV(gym.Env):
    
    def __init__(self, base_models, merge_models, data_iters, clip_features, data_scale):

        self.base_models = base_models
        self.merge_models = merge_models
        self.data_iters = data_iters
        self.clip_features = clip_features
        self.data_scale = data_scale
        
        self.num_blocks = len(self.merge_models[0].blocks)
        self.num_models = len(self.base_models) + len(self.merge_models)
        
        # [0, 1, 2, skip, back]
        self.action_space = spaces.Discrete(self.num_models + 2)
        self.observation_space = spaces.Discrete(self.num_models * self.num_blocks)
        
        self.seed()
        self.state = None
        self.cur_pos = 0
        self.blocks = []

        # save original blocks for restore
        self.merge_blocks = merge_models[0].blocks
        self.merge_models[0].head = HeadList([model.head for model in self.base_models])
        ######################################################
 
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
 
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        # select models
        if action < self.num_models: 
            block_pool = [model.blocks[self.cur_pos] for model in self.base_models] + \
                         [model.blocks[self.cur_pos] for model in self.merge_models]
            self.blocks.append(block_pool[action])
            self.state[self.cur_pos, action] += 1.0
            self.cur_pos += 1
        # select skip
        elif action == self.num_models:
            self.cur_pos += 1
        # select back
        elif action == (self.num_models+1):
            self.cur_pos -= 1

        # done
        if self.cur_pos == self.num_blocks:
            done = True
            with torch.no_grad():   #evaluate upspeed
                self.merge_models[0].blocks = torch.nn.Sequential(*self.blocks)
                
                results = []
                for iter_idx, data_iter in enumerate(self.data_iters):
                    acc, _ = validate_CV(self.merge_models[0], data_iter, self.clip_features, iter_idx, self.data_scale)
                    results.append(acc)
                    
                reward = sum(results) / len(results) * 100
                ########################################################
                results = ["Task {:d}: {:.2f}".format(i,r*100) for i,r in enumerate(results)]
                print(', '.join(results))
                print(self.state)
                ########################################################           
        # select back at first return -10 reward
        elif self.cur_pos < 0:
            done = True
            reward = -10
        else:
            done = False
            reward = 0
        
        return self.state.reshape([1, self.num_blocks*self.num_models]), reward, done, action
 
    def reset(self):
        # clear map/restore pos/clean blocks
        self.state = np.zeros([self.num_blocks, self.num_models], dtype=float)
        self.cur_pos = 0
        self.blocks = []
        self.merge_models[0].blocks = self.merge_blocks
        return self.state.reshape([1, self.num_blocks*self.num_models])