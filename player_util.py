from __future__ import division
import torch
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from time import sleep
from model import A3Clstm

TILES_REFILL_THRESHOLD_SCORE = 432
GAME_STAGE_CHANGEOVER_THRESHOLD = 300

# TILES_REFILL_THRESHOLD_SCORE = 1
# GAME_STAGE_CHANGEOVER_THRESHOLD = 1


class Agent(object):
    def __init__(self, env, args, gpu_id):
        self.results_filename = "./results"
        self.env = env
        self.models = [
            A3Clstm(self.env.observation_space.shape[0],
                    self.env.action_space),
            A3Clstm(self.env.observation_space.shape[0],
                    self.env.action_space)
        ]
        self.state = None
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = gpu_id
        self.episodic_reward = 0
        self.life_counter = 5
        self.model_sequence = []
        self.curr_model_id = 0
        self.first_time_changeover = True
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.models[0] = self.models[0].cuda()
                self.models[1] = self.models[1].cuda()
        with open(self.results_filename, 'w'):
            pass

    def action_train(self):
        value, logit, (self.hx, self.cx) = self.models[self.curr_model_id]((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())

        # Extra code to switch between models based on score
        if self.episodic_reward == 0:
            self.curr_model_id = 0
        elif self.episodic_reward == GAME_STAGE_CHANGEOVER_THRESHOLD:
            if self.first_time_changeover:
                self.late_game_model = copy.deepcopy(self.early_game_model)
                self.first_time_changeover = False
            self.curr_model_id = 1
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE:
            self.curr_model_id = 0
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE + GAME_STAGE_CHANGEOVER_THRESHOLD:
            self.curr_model_id = 1

        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.model_sequence.append(self.curr_model_id)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, (self.hx, self.cx) = self.models[self.curr_model_id]((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        # self.env.render()
        # sleep(0.005)
        # print(f"ACTION: {action[0]} DONE? {self.done}  INFO: {self.info}")

        if self.episodic_reward == 0:
            self.curr_model_id = 0
        elif self.episodic_reward == GAME_STAGE_CHANGEOVER_THRESHOLD:
            self.curr_model_id = 1
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE:
            self.curr_model_id = 0
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE + GAME_STAGE_CHANGEOVER_THRESHOLD:
            self.curr_model_id = 1


        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.model_sequence = []
        return self
