from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable


def train(rank, args, shared_models, optimizers, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    env.seed(args.seed + rank)
    player = Agent(env, args, gpu_id)
    player.rank = rank
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    player.models[0].train()
    player.models[1].train()
    player.eps_len += 2
#    player.test_models()
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                # player.model.load_state_dict(shared_model.state_dict())
                player.models[0].load_state_dict(shared_models[0].state_dict())
                player.models[1].load_state_dict(shared_models[1].state_dict())
        else:
            # player.model.load_state_dict(shared_model.state_dict())
            player.models[0].load_state_dict(shared_models[0].state_dict())
            player.models[1].load_state_dict(shared_models[1].state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        # if rank == 0:
        #     print(player.episodic_reward)
        player.episodic_reward = 0

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.models[player.curr_model_id]((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        # player.values.append(Variable(R))
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        # print("Length of values vector", len(player.values))
        # print("Length of rewards vector", len(player.rewards))
        # print("Length of model sequence vector", len(player.model_sequence))
        next_val = Variable(R)
        last_val = next_val
        R_vec = [Variable(R), Variable(R)]
        # last_id = player.model_sequence[-1]
        active_flags = [False, False]
        policy_loss = [0, 0]
        value_loss = [0, 0]
        for reward, value, model_id, log_prob, entropy in zip(
                reversed(player.rewards),
                reversed(player.values),
                reversed(player.model_sequence),
                reversed(player.log_probs),
                reversed(player.entropies)
        ):
            active_flags[model_id] = True
            R_vec[model_id] = args.gamma * R_vec[model_id] + reward
            R_vec[(model_id+1)%2] *= args.gamma

            advantage = R_vec[model_id] - value
            value_loss[model_id] += 0.5 * advantage.pow(2)

            delta_t = reward + args.gamma * next_val.data - value.data
            gae = gae * args.gamma * args.tau + delta_t
            policy_loss[model_id] -= (log_prob * Variable(gae) + 0.01 * entropy)

            next_val = value

        try:
            if active_flags[0] is True:
                player.models[0].zero_grad()
                (policy_loss[0] + 0.5 * value_loss[0]).backward()
                ensure_shared_grads(player.models[0], shared_models[0], gpu = gpu_id >= 0)
                optimizers[0].step()
            if active_flags[1] is True:
                player.models[1].zero_grad()
                (policy_loss[1] + 0.5 * value_loss[1]).backward()
                ensure_shared_grads(player.models[1], shared_models[1], gpu = gpu_id >= 0)
                optimizers[1].step()
        except Exception as e:
            print("Exception caught. Ignoring")
            if rank == 1:
                print(rewards)
                print(model_sequence)
        player.clear_actions()
