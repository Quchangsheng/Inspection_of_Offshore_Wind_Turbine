import copy
import numpy as np
import ray, gc
import sys, time, datetime
from tensorboardX import SummaryWriter

from agent.interactive_agent import InteractAgent
from envs.make_env import make_env


@ray.remote(num_cpus=1)
class Worker:
    def init(self, args, worker_id, device, cur_time=datetime.datetime.now()):
        self.worker_id = worker_id
        self.args = args
        self.device = device
        print(f'worker{worker_id} initialized in ', self.device)

        self.env = make_env(args)

        self.writer = SummaryWriter(f'./log/{args.scenario}/algorithm{args.algorithm}/{cur_time.day}_{cur_time.hour}_{cur_time.minute}/{worker_id}')
        self.episode_count = 0

    def collect_a_trajectory(self, interactive_agent, params_manager, replay_buffer):
        start = time.time()
        # reset the worker
        try:
            self.env.reset()
        except:
            del self.env
            self.env = make_env(self.args)
        gc.collect()

        # prepare for a new episode
        temp_trajectory_dict = self.init_empty_trajectory(interactive_agent)
        for step in range(self.env.max_cycles):
            termination, truncation = False, False
            # choose action
            for agent in self.env.agent_iter():
                agent_index = self.env._index_map[self.env.agent_selection]
                observation, reward, termination, truncation, info = self.env.last()
                action, info_dict = interactive_agent.choose_action_train(observation, agent_index, step)

                # fill trajectory
                temp_trajectory_dict[agent_index]['obs'].append(observation)
                temp_trajectory_dict[agent_index]['action'].append([action])
                temp_trajectory_dict[agent_index]['action_prob'].append(info_dict['action_prob'])
                temp_trajectory_dict[agent_index]['reward'].append([reward])
                temp_trajectory_dict[agent_index]['h_n_next'].append(info_dict['h_n'])
                temp_trajectory_dict[agent_index]['c_n_next'].append(info_dict['c_n'])
                temp_trajectory_dict[agent_index]['actor_index'].append(info_dict['actor_index'])
                temp_trajectory_dict[agent_index]['terminal'].append([termination or truncation])
                temp_trajectory_dict[agent_index]['mask'].append([1])

                if termination or truncation:
                    action = None
                self.env.step(action)
            if termination or truncation:
                break

        # trajectory storage
        ray.get(replay_buffer.store_trajectory.remote(temp_trajectory_dict))
        print(f'worker {self.worker_id} episode {self.episode_count} finished')

        del temp_trajectory_dict
        gc.collect()





    def collect_trajectories_for_debug(self, activated_model_list, params_manager, replay_buffer):
        # interactive agent initialization
        aht_agent_actor_list, teammate_agent_actor_list = ray.get(params_manager.get.remote())
        interactive_agent = InteractAgent(self.args, activated_model_list, self.device, aht_agent_actor_list, teammate_agent_actor_list)

        self.collect_a_trajectory(interactive_agent, params_manager, replay_buffer)

    def init_empty_trajectory(self, interactive_agent):
        temp_trajectory_dict = {}
        for agent_index in range(self.args.aht_agent_num + self.args.teammate_agent_num):
            temp_trajectory_dict[agent_index] = {
                'obs': [],
                'action': [],
                'action_prob': [],
                'reward': [],
                'h_n_next': [],
                'c_n_next': [],
                'actor_index': [],
                'terminal': [],
                'mask': [],  # 1: count;  0: masked
            }
        return temp_trajectory_dict





