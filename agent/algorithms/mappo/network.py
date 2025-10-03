import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MappoActor(nn.Module):
    def __init__(self, total_args, args):
        super(MappoActor, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(total_args.obs_space, args.actor_hidden_dim)
        self.lstm = nn.LSTM(args.actor_hidden_dim, args.actor_hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.fc2 = nn.Linear(args.actor_hidden_dim, total_args.action_space)

    def forward(self, x, hidden_state=None):
        # decision-making
        x = torch.tanh(self.fc1(x))
        x, hidden_state = self.lstm(x, hidden_state)
        x = torch.softmax(self.fc2(x), dim=-1)
        return x, hidden_state


class MappoCritic(nn.Module):
    def __init__(self, total_args, args):
        super(MappoCritic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(total_args.obs_space * (total_args.aht_agent_num + total_args.teammate_agent_num), self.args.critic_hidden_dim)
        self.lstm = nn.LSTM(self.args.critic_hidden_dim, self.args.critic_hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.fc2 = nn.Linear(self.args.critic_hidden_dim, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x, hidden_state=None):
        x = torch.tanh(self.fc1(x))
        x, hidden_state = self.lstm(x, hidden_state)
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x, hidden_state



