from collections import namedtuple
import gc
from .utils import *
import ray
import psutil
import sys
import numpy as np
import random
import copy


@ray.remote(num_cpus=1)
class ReplayBuffer:
    def __init__(self, args, capacity):
        self.args = args
        self.aht_agent_num = args.aht_agent_num
        self.teammate_agent_num = args.teammate_agent_num
        self.capacity = capacity

        # buffer initialization
        self.agents_buffer = {}
        for agent_id in range(self.aht_agent_num + self.teammate_agent_num):
            self.agents_buffer[agent_id] = []

        self.buffer_length = 0

    def store_trajectory(self, trajectory_dict):
        # controlled agents storage
        for agent_id in range(self.aht_agent_num + self.teammate_agent_num):
            self.agents_buffer[agent_id].append(trajectory_dict[agent_id])
            # if beyond the capacity
            if len(self.agents_buffer[agent_id]) > self.capacity:
                del self.agents_buffer[agent_id][0]
                gc.collect()

        self.buffer_length = len(self.agents_buffer[0])

        # check if all the buffer is of the same length
        assert len(self.agents_buffer[0]) == len(self.agents_buffer[self.aht_agent_num + self.teammate_agent_num - 1])

    def sample_batch(self, batch_size):
        agents_batch = {}
        assert batch_size <= self.capacity, 'batch is larger than capacity. enlarge the buffer or lower the batch.'
        assert batch_size <= self.buffer_length, 'samples are not enough for training.'

        sample_index_list = list(np.arange(0, batch_size))
        for agent_id in range(self.aht_agent_num + self.teammate_agent_num):
            agents_batch[agent_id] = [self.agents_buffer[agent_id][index] for index in sample_index_list]

        # make trajectory index for batch data
        for traj_index in range(self.args.batch_size):
            for agent_id in range(self.aht_agent_num + self.teammate_agent_num):
                agents_batch[agent_id][traj_index]['traj_index'] = [[traj_index] for _ in range(len(agents_batch[agent_id][traj_index]['obs']))]

        # fill the batch to same length
        agents_batch = self.fill_trajectory_to_same_length(agents_batch)
        return agents_batch

    def fill_trajectory_to_same_length(self, agents_batch):
        EMPTY_TRANSACTION = {
            'obs': [0 for _ in range(self.args.obs_space)],
            'action': [0],
            'action_prob': [1/self.args.action_space for _ in range(self.args.action_space)],
            'reward': [0],
            'h_n_next': list(np.zeros(16)),
            'c_n_next': list(np.zeros(16)),
            'actor_index': [0],
            'traj_index': [0],     # for MAPPO training, which indicates the traj_index even after the classification according to actor
            'terminal': [True],
            'mask': [0],
        }

        # find the largest length
        max_length = 0
        for agent_trajectories in list(agents_batch.values()):
            sub_max_length = min([len(agent_trajectory['obs']) for agent_trajectory in agent_trajectories])
            max_length = sub_max_length if max_length < sub_max_length else max_length

        # fill the trajectories
        for agent_trajectories in list(agents_batch.values()):
            for trajectory in agent_trajectories:
                length_bias = max_length - len(trajectory['obs'])
                assert length_bias >= 0, 'something wrong with max_length'

                for key in list(trajectory.keys()):
                    if key == 'actor_index' or key == 'traj_index':
                        trajectory[key].append(trajectory[key][0])
                    else:
                        trajectory[key].append(EMPTY_TRANSACTION[key])

        return agents_batch


    def get_trajectory_num(self):
        return len(self.agents_buffer[0])

    def clear_buffer(self):
        self.agents_buffer = {}
        for agent_id in range(self.aht_agent_num + self.teammate_agent_num):
            self.agents_buffer[agent_id] = []

        self.buffer_length = 0
        gc.collect()







