import argparse
import os
import pprint
import sys, time
import numpy as np
from tensorboardX import SummaryWriter
import ray
import torch, random


parser = argparse.ArgumentParser("Random example in RLCard")

# train config
parser.add_argument('--train_mode', type=bool, default=True, help='env.run() in step mode will be used')
parser.add_argument('--num_episodes', type=int, default=10000000, help='total episodes for collecting trajectories')
parser.add_argument('--max_episode_length', type=int, default=80, help='the maximum step of each episode')
parser.add_argument('--buffer_capacity', type=int, default=10000, help='buffer max capacity')
parser.add_argument('--on_policy_update_time', type=int, default=10, help='update time for one batch')
parser.add_argument('--batch_size', type=int, default=2, help='number of players in a game')
parser.add_argument('--worker_num', type=int, default=1, help='number of workers')

parser.add_argument('--num_turbines', type=int, default=100, help='number of turbines')
parser.add_argument('--num_UAV_per_ship', type=int, default=5, help='number of UAV')
parser.add_argument('--num_ship', type=int, default=1, help='number of ship')






