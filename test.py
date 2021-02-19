from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging


def test(args, shared_models, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(env, args, gpu_id)
    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    prev_reward = 0
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.models[0].load_state_dict(shared_models[0].state_dict())
                    player.models[1].load_state_dict(shared_models[1].state_dict())
            else:
                player.models[0].load_state_dict(shared_models[0].state_dict())
                player.models[1].load_state_dict(shared_models[1].state_dict())
            player.models[0].eval()
            player.models[1].eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))
            ##############################################################
            with open('./results','a') as f:
                line = f"{reward_total_sum - prev_reward}\n"
                f.write(line)
                prev_reward = reward_total_sum
            player.episodic_reward = 0
            player.fire_action_next = True
            ##############################################################
            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.models[0].state_dict()
                        torch.save(state_to_save, '{0}{1}_early.dat'.format(
                            args.save_model_dir, args.env))
                        state_to_save = player.models[1].state_dict()
                        torch.save(state_to_save, '{0}{1}_late.dat'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.models[0].state_dict()
                    torch.save(state_to_save, '{0}{1}_early.dat'.format(
                        args.save_model_dir, args.env))
                    state_to_save = player.models[1].state_dict()
                    torch.save(state_to_save, '{0}{1}_late.dat'.format(
                        args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
