import copy
import torch
import datetime
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from .network import MappoActor, MappoCritic


'''
The training process of MAPPO refers to the code from https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/1.MAPPO_MPE/mappo_mpe.py
'''


class MappoArgs:
    def __init__(self, total_args):
        self.lr_1 = 0.001
        self.lr_2 = 0.001
        self.lr_3 = 0.001
        self.gamma = 0.98
        self.lamda = 0.95
        self.epsilon = 0.2
        self.entropy_coef = 0.01
        self.grad_norm_clip = 1

        self.use_adv_norm = True
        self.use_value_clip = True
        self.use_grad_clip = True
        self.grad_clip_value = 10.0
        self.target_update_cycle = 10

        self.actor_hidden_dim = 16
        self.critic_hidden_dim = 32


class MAPPO:
    def __init__(self, args, device, activated_model_list, aht_agent_actor_list, teammate_agent_actor_list, in_worker):
        self.total_args = args
        self.device = device
        self.mappo_args = MappoArgs(self.total_args)
        cur_time = datetime.datetime.now()
        self.params_time = f'{cur_time.day}_{cur_time.hour}_{cur_time.minute}'

        # load or init actors
        self.activated_model_list = activated_model_list
        if aht_agent_actor_list is None and teammate_agent_actor_list is None:
            self.aht_agent_actor_list, self.teammate_agent_actor_list = self.init_actor_list()
        elif aht_agent_actor_list is not None:
            self.aht_agent_actor_list = aht_agent_actor_list
            _, self.teammate_agent_actor_list = self.init_actor_list()
        elif teammate_agent_actor_list is not None:
            self.aht_agent_actor_list, _ = self.init_actor_list()
            self.teammate_agent_actor_list = teammate_agent_actor_list
        else:
            self.aht_agent_actor_list, self.teammate_agent_actor_list = aht_agent_actor_list, teammate_agent_actor_list

        # network
        if not in_worker:
            # copy actors
            self.old_aht_agent_actor_list = copy.deepcopy(self.aht_agent_actor_list)
            self.old_teammate_agent_actor_list = copy.deepcopy(self.teammate_agent_actor_list)

            # init critic
            self.critic = MappoCritic(self.total_args, self.mappo_args)
            self.old_critic = MappoCritic(self.total_args, self.mappo_args)
            self.old_critic.load_state_dict(self.critic.state_dict())

            # move to device & copy the params
            for i in range(self.total_args.best_response_model_num):
                self.aht_agent_actor_list[i].to(self.device)
                self.old_aht_agent_actor_list[i].to(self.device)
                self.teammate_agent_actor_list[i].to(self.device)
                self.old_teammate_agent_actor_list[i].to(self.device)

            self.critic.to(self.device)
            self.old_critic.to(self.device)

            # optimizer
            self.aht_actor_optimizor_list, self.teammate_actor_optimizor_list = [], []
            self.critic_params = list(self.critic.parameters())
            for i in range(self.total_args.best_response_model_num):
                aht_actor_param = list(self.aht_agent_actor_list[i].parameters())
                teammate_actor_param = list(self.teammate_agent_actor_list[i].parameters())
                self.aht_actor_optimizor_list.append(torch.optim.Adam(aht_actor_param, lr=self.mappo_args.lr_1))
                self.teammate_actor_optimizor_list.append(torch.optim.Adam(teammate_actor_param, lr=self.mappo_args.lr_1))
            self.optimizer_critic = torch.optim.Adam(self.critic_params, lr=self.mappo_args.lr_2)
            self.optimizer_lagrange = torch.optim.Adam([alpha_1_matrix+alpha_2_matrix], lr=self.mappo_args.lr_3)

        # hidden for each agent
        self.aht_actor_hidden_state_dict, self.teammate_actor_hidden_state_dict = {}, {}
        if not in_worker:
            for actor_index in range(self.total_args.best_response_model_num):
                self.aht_actor_hidden_state_dict[actor_index] = (
                    torch.zeros(1, self.total_args.batch_size, self.mappo_args.actor_hidden_dim).to(self.device),
                    torch.zeros(1, self.total_args.batch_size, self.mappo_args.actor_hidden_dim).to(self.device)
                )
                self.teammate_actor_hidden_state_dict[actor_index] = (
                    torch.zeros(1, self.total_args.batch_size, self.mappo_args.actor_hidden_dim).to(self.device),
                    torch.zeros(1, self.total_args.batch_size, self.mappo_args.actor_hidden_dim).to(self.device)
                )
                self.critic_hidden_state = (
                    torch.zeros(1, self.total_args.batch_size, self.mappo_args.critic_hidden_dim).to(self.device),
                    torch.zeros(1, self.total_args.batch_size, self.mappo_args.critic_hidden_dim).to(self.device)
                )
        else:
            for actor_index in range(self.total_args.best_response_model_num):
                self.aht_actor_hidden_state_dict[actor_index] = (
                    torch.zeros(1, 1, self.mappo_args.actor_hidden_dim).to(self.device),
                    torch.zeros(1, 1, self.mappo_args.actor_hidden_dim).to(self.device)
                )
                self.teammate_actor_hidden_state_dict[actor_index] = (
                    torch.zeros(1, 1, self.mappo_args.actor_hidden_dim).to(self.device),
                    torch.zeros(1, 1, self.mappo_args.actor_hidden_dim).to(self.device)
                )
                self.critic_hidden_state = None

    def choose_action_train(self, obs, agent_index, step):
        """
        :param obs: the vector of obs of current agent
        :param agent_index: the index of current agent amount the env agents
        :param step: env step
        :return: action selected
        """
        # agent index [0, aht_agent_num-1] [aht_agent_num, agent_num], select the proper actor for current agent
        if agent_index - self.total_args.aht_agent_num <= -1:
            selected_actor_index = self.activated_model_list[0][agent_index]
            selected_actor = self.aht_agent_actor_list[selected_actor_index]
            actor_hidden_state = self.aht_actor_hidden_state_dict[selected_actor_index]
        else:
            selected_actor_index = self.activated_model_list[1][agent_index-self.total_args.aht_agent_num]
            selected_actor = self.teammate_agent_actor_list[selected_actor_index]
            actor_hidden_state = self.teammate_actor_hidden_state_dict[selected_actor_index]

        # actor forward and choose action
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).view(1, 1, -1)
            action_probs, actor_hidden_state = selected_actor.forward(obs_tensor, actor_hidden_state)
            action_distribution = torch.distributions.Categorical(probs=action_probs)
            action = action_distribution.sample()

            # update self.hidden
            if agent_index - self.total_args.aht_agent_num <= -1:
                self.aht_actor_hidden_state_dict[selected_actor_index] = actor_hidden_state
            else:
                self.teammate_actor_hidden_state_dict[selected_actor_index] = actor_hidden_state

        # hidden state after decision-making process
        h_n = actor_hidden_state[0][0][0].tolist()
        c_n = actor_hidden_state[1][0][0].tolist()
        action_prob_list = action_probs[0][0].tolist()
        info_dict = {
            'action_prob': action_prob_list,
            'h_n': h_n,
            'c_n': c_n,
            'actor_index': [selected_actor_index],
        }
        return action.item(), info_dict

    def choose_action_test(self):
        pass

    def train(self, batch, update_time):
        batch_for_critic, batch_for_aht_agents_actor, batch_for_teammate_agents_actor = self.prepare_data_for_training(batch)

        # Calculate the advantage using GAE
        with torch.no_grad():
            joint_obs_tensor = torch.tensor(np.array(batch_for_critic['joint_obs']), dtype=torch.float32).to(self.device)
            sequence_length = copy.deepcopy(joint_obs_tensor.size(2))
            joint_obs_tensor = joint_obs_tensor.permute(0, 2, 1, 3)
            joint_obs_tensor = joint_obs_tensor.reshape(self.total_args.batch_size, sequence_length, -1)
            v_old, _ = self.old_critic(joint_obs_tensor, self.critic_hidden_state)
            reward_tensor = torch.tensor(np.array(batch_for_critic['reward']), dtype=torch.float32).to(self.device)
            reward_tensor = reward_tensor.squeeze(1)
            num_terminal_tensor = torch.tensor(np.array(batch_for_critic['terminal']), dtype=torch.int32).to(self.device)
            num_terminal_tensor = num_terminal_tensor.squeeze(1)
            num_mask_tensor = torch.tensor(np.array(batch_for_critic['mask']), dtype=torch.int32).to(self.device)
            num_mask_tensor = num_mask_tensor.squeeze(1)

            adv = []
            gae = 0
            deltas = reward_tensor[:, 1:, :] + self.mappo_args.gamma * v_old[:, 1:, :] * (1 - num_terminal_tensor[:, 1:, :]) - v_old[:, :-1, :]   # The choice of 1: or :-1 should be carefully concerned
            deltas = deltas * num_mask_tensor[:, 1:, :]    # GPT: deltas = deltas * num_mask_tensor[:, -1:, :]
            for t in reversed(range(deltas.size(1))):
                gae = deltas[:, t, :] + self.mappo_args.gamma * self.mappo_args.lamda * gae    # GPT: gae = deltas[:, t, :] + gamma * lam * (1 - num_terminal_tensor[:, t, :]) * next_mask[:, t, :] * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + v_old[:, :-1, :]
            if self.mappo_args.use_adv_norm:     # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # optimize the policy for K times
        actor_loss_list, critic_loss_list = [], []
        for _ in range(self.total_args.on_policy_update_time):
            # aht agent actors
            for actor_index, actor in enumerate(self.aht_agent_actor_list):
                sub_actor_batch = batch_for_aht_agents_actor[actor_index]
                actor_optimizor = self.aht_actor_optimizor_list[actor_index]
                if len(sub_actor_batch['obs']) > 0:
                    actor_loss = self.single_actor_forward(actor, sub_actor_batch, adv, actor_optimizor)
                    # actor loss storage
                    actor_loss_list.append(actor_loss)
            # teammate agent actors
            for actor_index, actor in enumerate(self.teammate_agent_actor_list):
                sub_actor_batch = batch_for_teammate_agents_actor[actor_index]
                actor_optimizor = self.teammate_actor_optimizor_list[actor_index]
                if len(sub_actor_batch['obs']) > 0:
                    actor_loss = self.single_actor_forward(actor, sub_actor_batch, adv, actor_optimizor)
                    # actor loss storage
                    actor_loss_list.append(actor_loss)

            # critic loss
            v_new, _ = self.critic(joint_obs_tensor, self.critic_hidden_state)
            values_new = v_new[:, :-1, :]
            if self.mappo_args.use_value_clip:
                values_old = v_old[:, :-1, :].detach()
                values_error_clip = torch.clamp(values_new - values_old, -self.mappo_args.epsilon, self.mappo_args.epsilon) + values_old - v_target
                values_error_original = values_new - v_target
                critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
            else:
                critic_loss = (values_new - v_target) ** 2
            critic_loss = critic_loss * num_mask_tensor[:, 1:, :]
            self.optimizer_critic.zero_grad()
            critic_loss = critic_loss.mean()
            critic_loss.backward()
            if self.mappo_args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic_params, self.mappo_args.grad_clip_value)
            self.optimizer_critic.step()
            # critic loss storage
            critic_loss_list.append(critic_loss.cpu().data.item())

        # update old network
        self.update_old_network()

        # get params latest time
        cur_time = datetime.datetime.now()
        self.params_time = f'{cur_time.day}_{cur_time.hour}_{cur_time.minute}'

        return np.mean(np.array(actor_loss_list)), np.mean(np.array(critic_loss_list)), 0

    def single_actor_forward(self, actor, sub_batch, adv, actor_optimizor):
        """
        for single actor parameter update, which holds parallelizability
        :param actor:
        :param sub_batch:
        :return:
        """
        # data process for old action log prob
        action_tensor = torch.squeeze(torch.tensor(np.array(sub_batch['action']), dtype=torch.long), dim=-1).to(self.device)
        action_tensor_one_hot = F.one_hot(action_tensor, num_classes=self.total_args.action_space).to(self.device)
        old_action_space_prob = torch.tensor(np.array(sub_batch['action_prob']), dtype=torch.float32).to(self.device)
        old_action_prob_log = torch.log(torch.sum(action_tensor_one_hot * old_action_space_prob, dim=-1, keepdim=True) + 1e-3)
        num_mask_tensor = torch.tensor(np.array(sub_batch['mask']), dtype=torch.int32).to(self.device)
        num_mask_tensor = num_mask_tensor.squeeze(1)

        # get new action log prob
        obs_tensor = torch.tensor(np.array(sub_batch['obs']), dtype=torch.float32).to(self.device)
        hidden_state = (
            torch.zeros(1, obs_tensor.size(0), self.mappo_args.actor_hidden_dim).to(self.device),
            torch.zeros(1, obs_tensor.size(0), self.mappo_args.actor_hidden_dim).to(self.device)
        )
        new_action_space_prob, _ = actor(obs_tensor, hidden_state)
        new_action_prob_log = torch.log(torch.sum(action_tensor_one_hot * new_action_space_prob, dim=-1, keepdim=True) + 1e-3)
        new_action_space_entropy = torch.sum(new_action_space_prob * torch.log(new_action_space_prob), dim=-1, keepdim=True)[:, :-1, :]

        traj_index_numpy = np.array(sub_batch['traj_index'])[:, 0, 0]    # it is not necessary to be in order.
        ratios = torch.exp(new_action_prob_log - old_action_prob_log.detach())[:, :-1, :]
        sub_adv = adv[list(traj_index_numpy), :]
        surr1 = ratios * sub_adv
        surr2 = torch.clamp(ratios, 1-self.mappo_args.epsilon, 1+self.mappo_args.epsilon) * sub_adv
        actor_loss = -torch.min(surr1, surr2) - self.mappo_args.entropy_coef * new_action_space_entropy
        actor_loss = actor_loss * num_mask_tensor[:, 1:, :]

        actor_optimizor.zero_grad()
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        if self.mappo_args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()), self.mappo_args.grad_clip_value)
        actor_optimizor.step()

        return actor_loss.cpu().data.item()

    def prepare_data_for_training(self, batch):
        """
        One batch for actor network according to actor index and another batch for critic according to trajectory.
        :param batch: original data without being classified
        :return: classified tensor
        """
        ################## make batch for critic ##################
        batch_for_critic = {
            'joint_obs': [],
            'reward': [],
            'terminal': [],
            'mask': [],
        }
        for traj_index in range(self.total_args.batch_size):
            traj_joint_obs = []
            traj_reward = []
            traj_terminal = []
            traj_mask = []
            for agent_index in range(self.total_args.aht_agent_num+self.total_args.teammate_agent_num):
                traj_joint_obs.append(batch[agent_index][traj_index]['obs'])
                assert batch[agent_index][traj_index]['reward'] == batch[0][traj_index]['reward']
                assert batch[agent_index][traj_index]['terminal'] == batch[0][traj_index]['terminal']
                assert batch[agent_index][traj_index]['mask'] == batch[0][traj_index]['mask']
            traj_reward.append(batch[0][traj_index]['reward'])
            traj_terminal.append(batch[0][traj_index]['terminal'])
            traj_mask.append(batch[0][traj_index]['mask'])

            batch_for_critic['joint_obs'].append(traj_joint_obs)
            batch_for_critic['reward'].append(traj_reward)
            batch_for_critic['terminal'].append(traj_terminal)
            batch_for_critic['mask'].append(traj_mask)

        ################## make batch for actors ##################
        # batch initialization
        batch_for_aht_agents_actor, batch_for_teammate_agents_actor = {}, {}
        for i in range(self.total_args.best_response_model_num):
            batch_for_aht_agents_actor[i], batch_for_teammate_agents_actor[i] = {}, {}
            for key in list(batch[0][0].keys()):
                batch_for_aht_agents_actor[i][key] = []
                batch_for_teammate_agents_actor[i][key] = []
        # fill the data for the two actor batches
        for traj_index in range(self.total_args.batch_size):
            for agent_index in range(self.total_args.aht_agent_num+self.total_args.teammate_agent_num):
                if agent_index < self.total_args.aht_agent_num:   # for aht agents
                    assert batch[agent_index][traj_index]['actor_index'][0][0] == batch[agent_index][traj_index]['actor_index'][-1][0]
                    actor_index = batch[agent_index][traj_index]['actor_index'][0][0]
                    for key in list(batch[agent_index][traj_index].keys()):
                        batch_for_aht_agents_actor[actor_index][key].append(batch[agent_index][traj_index][key])
                else:                                             # for teammates agents
                    assert batch[agent_index][traj_index]['actor_index'][0][0] == batch[agent_index][traj_index]['actor_index'][-1][0]
                    actor_index = batch[agent_index][traj_index]['actor_index'][0][0]
                    for key in list(batch[agent_index][traj_index].keys()):
                        batch_for_teammate_agents_actor[actor_index][key].append(batch[agent_index][traj_index][key])

        return batch_for_critic, batch_for_aht_agents_actor, batch_for_teammate_agents_actor

    def update_old_network(self):
        for i in range(self.total_args.best_response_model_num):
            # Load state dict from eval actors to old actors
            self.old_aht_agent_actor_list[i].load_state_dict(self.aht_agent_actor_list[i].state_dict())
            self.old_teammate_agent_actor_list[i].load_state_dict(self.teammate_agent_actor_list[i].state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

    def init_actor_list(self):
        mappo_args = MappoArgs(self.total_args)

        # k model initialization for aht agent
        aht_agent_actor_list = []
        for i in range(self.total_args.best_response_model_num):
            actor = MappoActor(self.total_args, mappo_args)
            aht_agent_actor_list.append(actor)

        # k model initialization for teammate agents
        teammate_agent_actor_list = []
        for i in range(self.total_args.best_response_model_num):
            actor = MappoActor(self.total_args, mappo_args)
            teammate_agent_actor_list.append(actor)

        return [aht_agent_actor_list, teammate_agent_actor_list]

    def get_latest_actor_list(self):
        return [self.aht_agent_actor_list, self.teammate_agent_actor_list]

    def load_latest_actor_list(self, aht_agent_actor_list, teammate_agent_actor_list):
        self.aht_agent_actor_list = aht_agent_actor_list
        self.teammate_agent_actor_list = teammate_agent_actor_list

    def save_model_list(self, path):
        # k model obtain for aht agent
        aht_agent_model_list = []
        for i in range(self.total_args.best_response_model_num):
            actor = self.aht_agent_actor_list[i]
            actor_net_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
            cur_time = datetime.datetime.now()
            params_time = f'{cur_time.day}_{cur_time.hour}_{cur_time.minute}'
            net_dict = {
                'param_dict': actor_net_dict,
                'save_time': params_time
            }
            aht_agent_model_list.append(net_dict)

        # k model obtain for teammate agents
        teammate_agent_model_list = []
        for i in range(self.total_args.best_response_model_num):
            actor = self.teammate_agent_actor_list[i]
            actor_net_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
            cur_time = datetime.datetime.now()
            params_time = f'{cur_time.day}_{cur_time.hour}_{cur_time.minute}'
            net_dict = {
                'param_dict': actor_net_dict,
                'save_time': params_time
            }
            teammate_agent_model_list.append(net_dict)

        model_dict = {
            'aht_agent_model_list': aht_agent_model_list,
            'teammate_agent_model_list': teammate_agent_model_list,
        }
        torch.save(model_dict, path)


