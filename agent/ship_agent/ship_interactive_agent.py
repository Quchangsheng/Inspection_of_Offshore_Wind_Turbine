import numpy as np
from agent.algorithms.mappo.mappo import MAPPO


algorithm_dict = {
    'mappo': MAPPO,
}


class InteractAgent:
    def __init__(self, args, device, in_worker=True):
        self.args = args
        self.device = device

        self.algo = algorithm_dict[args.algorithm](self.args, in_worker=in_worker)

    def step(self, obs_n, step):
        return self.random_action()

    def choose_action_train(self, obs, agent_index, step):
        action, info_dict = self.algo.opt_algo.choose_action_train(obs, agent_index, step)
        return action, info_dict

    def choose_action_test(self, obs_n, step):
        pass

    def random_action(self):
        actions_n = []
        for i in range(self.aht_agent_num + self.teammate_agent_num):
            action_index = np.random.randint(0, self.args.action_space)
            action_onehot = np.zeros(self.args.action_space)
            action_onehot[action_index] = 1
            actions_n.append(action_onehot)
        return actions_n

    def load_latest_actor_list(self, aht_agent_actor_list, teammate_agent_actor_list):
        self.algo.load_latest_actor_list(aht_agent_actor_list, teammate_agent_actor_list)

    def get_latest_actor_list(self):
        aht_agent_actor_list, teammate_agent_actor_list = self.algo.get_latest_actor_list()
        return [aht_agent_actor_list, teammate_agent_actor_list]

    def save_model_list(self, path):
        self.algo.save_model_list(path)