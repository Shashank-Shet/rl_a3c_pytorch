from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


SECOND_LAYOUT_SCORE = 432
LATE_GAME_SCORE_DELTA = 300


class Agent(object):
    def __init__(self, model, env, args, state):
        self.early_game_model = model
        self.late_game_model = None
        self.model = self.early_game_model
        self.env = env
        self.state = state
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
        self.gpu_id = -1
        self.episodic_reward = 0
        self.life_counter = 5

    def set_model(self, model):
        self.model = model
        self.early_game_model = self.model
        
    def action_train(self):
        value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())

        if self.done is True:
            self.life_counter -= 1
            if self.life_counter == 0:
                print("Episodic reward: ", self.episodic_reward)
                self.episodic_reward = 0
                self.life_counter = 5
        self.episodic_reward += self.reward

        if self.episodic_reward == LATE_GAME_SCORE_DELTA:
            if self.late_game_model is None:
                self.late_game_model = self.early_game_model.copy()
            self.model = self.late_game_model
        elif self.episodic_reward == SECOND_LAYOUT_SCORE:
            self.model = self.early_game_model
        elif self.episodic_reward == SECOND_LAYOUT_SCORE + LATE_GAME_SCORE_DELTA:
            self.model = self.late_game_model

        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
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
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])

        if self.done is True:
            self.life_counter -= 1
            if self.life_counter == 0:
                print("Episodic reward: ", self.episodic_reward)
                self.episodic_reward = 0
                self.life_counter = 5
        self.episodic_reward += self.reward

        if self.episodic_reward == LATE_GAME_SCORE_DELTA:
            if self.late_game_model is None:
                self.late_game_model = self.early_game_model.copy()
            self.model = self.late_game_model
        elif self.episodic_reward == SECOND_LAYOUT_SCORE:
            self.model = self.early_game_model
        elif self.episodic_reward == SECOND_LAYOUT_SCORE + LATE_GAME_SCORE_DELTA:
            self.model = self.late_game_model

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
        return self
