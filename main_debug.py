import argparse
import os
import pprint
import sys, time
import numpy as np
from tensorboardX import SummaryWriter
import ray
import torch, random

from distribution.worker import Worker
from distribution.learner import Learner
from distribution.replay_buffer import ReplayBuffer
from distribution.parameter_manager import ParameterManager
from agent.interactive_agent import InteractAgent
from env.make_env import make_env


parser = argparse.ArgumentParser("Random example in RLCard")

# train config
parser.add_argument('--train_mode', type=bool, default=True, help='env.run() in step mode will be used')
parser.add_argument('--num_episodes', type=int, default=10000000, help='total episodes for collecting trajectories')
parser.add_argument('--max_episode_length', type=int, default=300, help='the maximum step of each episode')
parser.add_argument('--buffer_capacity', type=int, default=10000, help='buffer max capacity')
parser.add_argument('--on_policy_update_time', type=int, default=10, help='update time for one batch')
parser.add_argument('--batch_size', type=int, default=2, help='number of players in a game')
parser.add_argument('--worker_num', type=int, default=1, help='number of workers')
parser.add_argument('--use_cuda', type=bool, default=True, help='load model')

parser.add_argument('--num_turbines', type=int, default=100, help='number of turbines')
parser.add_argument('--num_UAV_per_ship', type=int, default=5, help='number of UAV')
parser.add_argument('--num_ship', type=int, default=2, help='number of ship')

parser.add_argument('--algorithm', type=str, default='mappo', help='algorithm selection')
parser.add_argument('--model_dir', type=str, default='model', help='the first level direction of saving path')

args = parser.parse_args()


def run(args):
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("CUDA is available.")
        print("Device:", torch.cuda.get_device_name())
    else:
        dev = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    print(f"--------create {args.worker_num} workers, 1 buffer and 1 learner--------")
    workers = [Worker.remote() for i in range(args.worker_num)]
    learner = Learner.remote()
    learner_device = torch.device(f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu")
    print('learner_device:', learner_device)
    worker_device = torch.device('cpu')

    print(f"--------main process preparation--------")
    env = make_env(args)
    args.action_space = list(env.action_spaces.values())[0].n
    args.obs_space = list(env.observation_spaces.values())[0].shape[0]
    args.max_episode_length = env.max_cycles

    ray.get([
        worker.init.remote(args, i, worker_device)
        for i, worker in enumerate(workers)
    ])

    interactive_agent = InteractAgent(args, learner_device, in_worker=False)
    ray.get(learner.init.remote(args, interactive_agent, learner_device))

    # buffer and parameter manager
    print('--------buffer initialization--------')
    replay_buffer = ReplayBuffer.remote(args, args.buffer_capacity)
    params_manager = ParameterManager.remote(ray.get(learner.get_actor.remote()))
    print('--------buffer initialized--------')

    # training process
    train_epoch = 0

    while True:
        print('-----------start to collect trajectories for buffer-----------')
        [worker.collect_trajectories_for_debug.remote(params_manager, replay_buffer) for worker in workers]
        # training
        buffer_length = ray.get(replay_buffer.get_trajectory_num.remote())
        if buffer_length >= args.batch_size:
            print('one batch collected')

            print('-------------------')
            print(f'buffer_length now is {buffer_length}')
            batch = ray.get(replay_buffer.sample_batch.remote(args.batch_size))
            ray.get(learner.train.remote(batch, params_manager, args))
            ray.get(replay_buffer.clear.remote())
            train_epoch += 1
            print('train episode: ', train_epoch)
            print('-------------------')

        # save model
        if train_epoch % 100 == 1:
            # model save path
            if args.train_model:
                agent_model_path = f'./agent/model/{args.scenario}/{args.algorithm}/' + str(train_epoch)
                if not os.path.exists(agent_model_path):
                    os.makedirs(agent_model_path)
                ray.get(learner.save_model.remote(agent_model_path))


def set_seed(seed=42):
    # Python 原生随机数生成器
    random.seed(seed)
    # Numpy 随机数生成器
    np.random.seed(seed)
    # PyTorch 随机数生成器
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    set_seed(42)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ray.init(num_cpus=100, num_gpus=1)
    ray.init(local_mode=True)

    run(args)




