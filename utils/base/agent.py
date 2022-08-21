class Agent(object):
    def __init__(self):
        self.state_norm_codes = (0, 3, 4)
        self.reward_norm_codes = (1, 3)
        self.td_norm_codes = (2, 4)
        self.loss = list()
        self.save_path = ""

    def train(self, *args, **kwargs):
        pass

    def policy(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
