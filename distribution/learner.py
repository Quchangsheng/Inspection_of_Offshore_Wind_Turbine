import numpy as np
import copy
import gc
import ray, torch
from tensorboardX import SummaryWriter
import sys, time, datetime


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def init(self, args, interactive_agent, device, cur_time=datetime.datetime.now()):
        self.args = args
        self.interactive_agent = interactive_agent
        self.device = device
        print(f'learner initialized in {self.device}')

        self.writer = SummaryWriter(f'./log/scenario-{args.scenario}/algo-{args.algorithm}/{cur_time.day}_{cur_time.hour}_{cur_time.minute}/learner')
        self.update_time = 0

    def update_activated_model_list(self, activated_model_list):
        self.interactive_agent.activated_model_list = activated_model_list
        print(f'activated_model_list updated {activated_model_list}')

    def train(self, batch, params_manager, args):
        loss_dict = self.interactive_agent.algo.train(batch, self.update_time)

        agents_actor_list = self.get_actor()
        ray.get(params_manager.set.remote(agents_actor_list))

    def get_actor(self):
        return self.interactive_agent.get_latest_actor_list()

    def save_model(self, path):
        self.interactive_agent.save_model_list(path)





