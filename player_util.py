from __future__ import division
import torch
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from time import sleep


TILES_REFILL_THRESHOLD_SCORE = 432
GAME_STAGE_CHANGEOVER_THRESHOLD = 300

# TILES_REFILL_THRESHOLD_SCORE = 1
# GAME_STAGE_CHANGEOVER_THRESHOLD = 1


class Agent(object):
    def __init__(self, model, env, args, state):
        self.results_filename = "./results"
        self.early_game_model = model
        self.late_game_model = None
        self.models = [self.early_game_model, self.late_game_model]
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
        self.model_sequence = []
        self.curr_model_id = 0
        self.first_time_changeover = True
        self.next_action_fire = True

    def test_models(self):
        print(self.model)
        print(self.early_game_model)
        print(self.late_game_model)

    def set_model(self, model):
        self.early_game_model = model
        self.late_game_model  = copy.deepcopy(model)
        self.model = self.early_game_model
        self.models = [self.early_game_model, self.late_game_model]

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

        # Extra code for book-keeping progress of rewards in training.
        # Note: Training allows only 1 life for the agent
        if self.done is True:
            self.life_counter = 0
        self.episodic_reward += self.reward

        # Extra code to switch between models based on score
        if self.episodic_reward == 0:
            self.model = self.early_game_model
            self.curr_model_id = 0
        elif self.episodic_reward == GAME_STAGE_CHANGEOVER_THRESHOLD:
            if self.first_time_changeover:
                self.late_game_model = copy.deepcopy(self.early_game_model)
                self.first_time_changeover = False
            self.model = self.late_game_model
            self.curr_model_id = 1
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE:
            self.model = self.early_game_model
            self.curr_model_id = 0
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE + GAME_STAGE_CHANGEOVER_THRESHOLD:
            self.model = self.late_game_model
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
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        if self.next_action_fire is False:
            action = prob.max(1)[1].data.cpu().numpy()
        else:
            action = [1]
        state, self.reward, self.done, self.info = self.env.step(action[0])
        # self.env.render()
        # sleep(0.005)
        # print(f"ACTION: {action[0]} DONE? {self.done}  INFO: {self.info}")

        if self.done is True:
            self.next_action_fire = True
            self.life_counter -= 1
            if self.life_counter == 0:
                with open(self.results_filename, 'a') as f:
                    line = f"{self.episodic_reward}\n"
                    f.write(line)
 #               print("Episodic reward: ", self.episodic_reward)
                self.episodic_reward = 0
                self.life_counter = 5
        else:
            self.next_action_fire = False

        self.episodic_reward += self.reward


        if self.episodic_reward == 0:
            self.model = self.early_game_model
            self.curr_model_id = 0
        elif self.episodic_reward == GAME_STAGE_CHANGEOVER_THRESHOLD:
            # if self.first_time_changeover:
            #     self.late_game_model = copy.deepcopy(self.early_game_model)
            #     self.first_time_changeover = False
            self.model = self.late_game_model
            self.curr_model_id = 1
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE:
            self.model = self.early_game_model
            self.curr_model_id = 0
        elif self.episodic_reward == TILES_REFILL_THRESHOLD_SCORE + GAME_STAGE_CHANGEOVER_THRESHOLD:
            self.model = self.late_game_model
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
