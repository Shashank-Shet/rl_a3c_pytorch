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
    if optimizers is None:
        if args.optimizer == 'RMSprop':
            optimizers = [
                optim.RMSprop(shared_models[0].parameters(), lr=args.lr),
                optim.RMSprop(shared_models[1].parameters(), lr=args.lr)
            ]
        if args.optimizer == 'Adam':
            optimizers = [
                optim.Adam(shared_models[0].parameters(), lr=args.lr, amsgrad=args.amsgrad),
                optim.Adam(shared_models[1].parameters(), lr=args.lr, amsgrad=args.amsgrad)
            ]
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.set_model(A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space))

#    player.model = A3Clstm(player.env.observation_space.shape[0],
#                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.early_game_model = player.early_game_model.cuda()
            player.late_game_model = player.late_game_model.cuda()
    # player.model.train()
    player.early_game_model.train()
    player.late_game_model.train()
    player.eps_len += 2
#    player.test_models()
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                # player.model.load_state_dict(shared_model.state_dict())
                player.early_game_model.load_state_dict(shared_models[0].state_dict())
                player.late_game_model.load_state_dict(shared_models[1].state_dict())
        else:
            # player.model.load_state_dict(shared_model.state_dict())
            player.early_game_model.load_state_dict(shared_models[0].state_dict())
            player.late_game_model.load_state_dict(shared_models[1].state_dict())
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

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        # player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
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
        R_vec = [Variable(R), Variable(R), Variable(R), Variable(R)]
        last_id = player.model_sequence[-1]
        P_vec, V_vec = [None, None], [None, None]
        for reward, value, model_id, log_prob, entropy in zip(
                reversed(player.rewards),
                reversed(player.values),
                reversed(player.model_sequence),
                reversed(player.log_probs),
                reversed(player.entropies)
        ):
            R_vec[model_id] = args.gamma * R_vec[model_id] + reward
            advantage = R_vec[model_id] - value
            value_loss += 0.5 * advantage.pow(2)

            delta_t = reward + args.gamma * next_val.data - value.data
            gae = gae * args.gamma * args.tau + delta_t
            policy_loss -= (log_prob * Variable(gae) + 0.01 * entropy)

            if model_id != last_id:
                if P_vec[model_id] is not None:
                    P_vec[model_id] += policy_loss
                    V_vec[model_id] += value_loss
                else:
                    P_vec[model_id] = policy_loss
                    V_vec[model_id] = value_loss
                # player.models[model_id].zero_grad()
                # (policy_loss + 0.5 * value_loss).backward(retain_graph=True)
                # ensure_shared_grads(player.models[model_id], shared_models[model_id], gpu = gpu_id >= 0)
                # optimizers[model_id].step()

            last_id = model_id
        
        if P_vec[model_id] is not None:
            P_vec[model_id] += policy_loss
            V_vec[model_id] += value_loss
        else:
            P_vec[model_id] = policy_loss
            V_vec[model_id] = value_loss

        player.models[0].zero_grad()
        player.models[1].zero_grad()
        if P_vec[0] is not None:
            (P_vec[0] + 0.5 * V_vec[0]).backward()
        if P_vec[1] is not None:
            (P_vec[1] + 0.5 * V_vec[1]).backward()
        ensure_shared_grads(player.models[0], shared_models[0], gpu = gpu_id >= 0)
        ensure_shared_grads(player.models[1], shared_models[1], gpu = gpu_id >= 0)
        optimizers[0].step()
        optimizers[1].step()

            
        # for i in reversed(range(len(player.rewards))):
        #     R = args.gamma * R + player.rewards[i]
        #     advantage = R - player.values[i]
        #     value_loss = value_loss + 0.5 * advantage.pow(2)

        #     # Generalized Advantage Estimataion
        #     delta_t = player.rewards[i] + args.gamma * \
        #         player.values[i + 1].data - player.values[i].data

        #     gae = gae * args.gamma * args.tau + delta_t

        #     policy_loss = policy_loss - \
        #         player.log_probs[i] * \
        #         Variable(gae) - 0.01 * player.entropies[i]

        # player.model.zero_grad()
        # (policy_loss + 0.5 * value_loss).backward()
        # ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        # optimizer.step()
        player.clear_actions()
