import pickle
import random
import time

import numpy as np
import torch


class Memory:
    def __init__(self, buffer_size=None, device="cpu"):
        self.state_current_ = list()
        self.state_next_ = list()
        self.reward_ = list()
        self.action_ = list()
        self.done_ = list()
        self.terminal_state_indices = list()
        self.buffer_size = buffer_size
        self.device = device
        return

    def append(self, state_current, state_next, reward, action, done):
        self.__append(state_current, state_next, reward, action, done)
        return

    def cast_to_tensor(self, data, dtype=torch.float32, ndims=None):
        if ndims is not None:
            assert isinstance(
                ndims, int
            ), "Argument `ndims` must be an int specifying number of dimensions!"

        # If already in Tensor, return.
        if isinstance(data, torch.Tensor):
            return self.__adjust_dims_for_tensor(ndims, data).to(self.device)
        if isinstance(data, (list, tuple)):
            if isinstance(data[0], torch.Tensor):
                return data

        # For other cases.
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, (list, tuple, float, int, np.float64, np.double)):
            data = torch.tensor(data, dtype=dtype)
        elif isinstance(data, bool):
            data = torch.tensor(int(data), dtype=dtype)
        else:
            raise TypeError(
                f"Expected states to be one of"
                f" {np.ndarray}, {torch.Tensor}, {list} and {tuple} but got "
                f"{type(data)}"
            )
        return self.__adjust_dims_for_tensor(ndims, data).to(self.device)

    def save(self, path):
        with open(path, "w") as f:
            to_pickle = {
                "state_current": self.state_current_,
                "state_next": self.state_next_,
                "reward": self.reward_,
                "action": self.action_,
                "done": self.done_,
            }
            pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def load(self, path):
        with open(path, "r") as f:
            pickled = pickle.load(f)
        self.state_current_ = pickled["state_current"]
        self.state_next_ = pickled["state_next"]
        self.reward_ = pickled["reward"]
        self.action_ = pickled["action"]
        self.done_ = pickled["done"]
        return

    def clear(self):
        self.state_current_.clear()
        self.state_next_.clear()
        self.reward_.clear()
        self.action_.clear()
        self.done_.clear()
        return

    def reserve(self, buffer_size):
        self.buffer_size = buffer_size
        return

    def get_random_terminal_sample(self):
        if not self.has_terminal_state():
            raise RuntimeError("No Terminal states yet in the Memory!")
        index = self.get_random_terminal_sample_index()
        return self[self.terminal_state_indices[index]]

    def get_random_terminal_sample_index(self):
        if not self.has_terminal_state():
            raise RuntimeError("No Terminal states yet in the Memory!")
        random.seed(time.time())
        index = random.randint(0, len(self.terminal_state_indices) - 1)
        return self.terminal_state_indices[index]

    def has_terminal_state(self):
        if len(self.terminal_state_indices) > 0:
            return True
        return False

    def stack_current_states(self):
        return torch.stack(self.state_current_, dim=0)

    def stack_next_states(self):
        return torch.stack(self.state_next_, dim=0)

    def stack_rewards(self):
        return torch.stack(self.reward_, dim=0)

    def stack_actions(self):
        return torch.stack(self.action_, dim=0)

    def stack_dones(self):
        return torch.stack(self.done_, dim=0)

    def __append(self, state_current, state_next, reward, action, done):

        if isinstance(done, bool):
            done = int(done)

        if done == 1:
            self.terminal_state_indices.append(self.__len__() - 1)

        if isinstance(state_current, np.ndarray):
            ndims = len(state_current.shape)
        elif isinstance(state_current, torch.Tensor):
            ndims = state_current.dim()
        elif isinstance(state_current, (list, tuple)):
            ndims = None
        else:
            raise TypeError(
                f"Invalid type received! Expected state "
                f"values to be of type {torch.Tensor} or {np.ndarray} but "
                f"received of list of type: {type(state_current)}."
            )

        # Cast to torch.Tensor
        state_current = self.cast_to_tensor(state_current, ndims=ndims)
        state_next = self.cast_to_tensor(state_next, ndims=ndims)
        reward = self.cast_to_tensor(
            reward, ndims=ndims - 1 if ndims is not None else None
        )
        action = self.cast_to_tensor(
            action, dtype=torch.int64, ndims=ndims - 1 if ndims is not None else None
        )
        done = self.cast_to_tensor(done, ndims=ndims - 1 if ndims is not None else None)

        delete_flag = False
        if self.buffer_size is not None:
            if self.__len__() > self.buffer_size:
                delete_flag = True

        # Append operations
        if isinstance(state_current, (np.ndarray, torch.Tensor)):
            if delete_flag:
                del self[0]
            self.__append_item(state_current, state_next, reward, action, done)

        elif isinstance(state_current, (list, tuple)):
            if delete_flag:
                del self[: len(state_current)]
            self.__append_items(state_current, state_next, reward, action, done)

        else:
            raise TypeError(
                f"Expected states to be one of"
                f" {np.ndarray}, {torch.Tensor}, {list} and {tuple} but got "
                f"{type(state_current)}"
            )
        return

    def __append_item(self, state_current, state_next, reward, action, done):
        self.state_current_.append(state_current)
        self.state_next_.append(state_next)
        self.reward_.append(reward)
        self.action_.append(action)
        self.done_.append(done)
        return

    def __append_items(self, states_current, states_next, rewards, actions, dones):
        self.state_current_.extend(states_current)
        self.state_next_.extend(states_next)
        self.reward_.extend(rewards)
        self.action_.extend(actions)
        self.done_.extend(dones)
        return

    @staticmethod
    def __adjust_dims_for_tensor(target_dim, tensor):
        if target_dim is None:
            return tensor
        curr_dim = tensor.dim()
        if target_dim != curr_dim:
            for _ in range(abs(target_dim - curr_dim)):
                tensor = torch.unsqueeze(tensor, dim=-1)

        return tensor

    def __getitem__(self, index):
        if isinstance(index, int):
            return (
                self.state_current_[index],
                self.state_next_[index],
                self.reward_[index],
                self.action_[index],
                self.done_[index],
            )
        elif isinstance(index, (list, tuple)):
            return (
                [self.state_current_[idx] for idx in index],
                [self.state_next_[idx] for idx in index],
                [self.reward_[idx] for idx in index],
                [self.action_[idx] for idx in index],
                [self.done_[idx] for idx in index],
            )
        else:
            raise TypeError(
                f"Expected one of {int}, {list} or {tuple}, but got {type(index)}!"
            )

    def __delitem__(self, index):
        del self.state_current_[index]
        del self.state_next_[index]
        del self.reward_[index]
        del self.action_[index]
        del self.done_[index]
        return

    def __len__(self):
        return len(self.reward_)
