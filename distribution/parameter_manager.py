import torch
import ray, gc, psutil


@ray.remote(num_cpus=1)
class ParameterManager:
    def __init__(self, agents_actor_list):
        print(f"获取网络参数")
        self.aht_agent_actor_list = agents_actor_list[0]
        self.teammate_agent_actor_list = agents_actor_list[1]

    def set(self, agents_latest_actor_list):
        del self.aht_agent_actor_list, self.teammate_agent_actor_list
        gc.collect()
        self.aht_agent_actor_list = agents_latest_actor_list[0]
        self.teammate_agent_actor_list = agents_latest_actor_list[1]

    def get(self):
        return self.aht_agent_actor_list, self.teammate_agent_actor_list











