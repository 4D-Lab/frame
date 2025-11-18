from collections import defaultdict

import torch


class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults

        self.k = k
        self.alpha = alpha
        self.state = defaultdict(dict)
        self.fast_state = base_optimizer.state

        for group in self.param_groups:
            group["counter"] = 0

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            group["counter"] += 1

            if group["counter"] >= self.k:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    if "slow_param" not in self.state[p]:
                        self.state[p]["slow_param"] = p.data.clone()

                    slow_param = self.state[p]["slow_param"]
                    p.data = slow_param + self.alpha * (p.data - slow_param)
                    self.state[p]["slow_param"] = p.data.clone()

                group["counter"] = 0

        return loss

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        state = {"base_optimizer": self.base_optimizer.state_dict(),
                 "lookahead_state": dict(self.state),
                 "k": self.k,
                 "alpha": self.alpha}

        return state

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.state.update(state_dict["lookahead_state"])
        self.k = state_dict.get("k", self.k)
        self.alpha = state_dict.get("alpha", self.alpha)

    def add_param_group(self, param_group):
        self.base_optimizer.add_param_group(param_group)
        self.param_groups = self.base_optimizer.param_groups
