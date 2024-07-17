"""
by Zhaofan Qiu, Copyright 2022.
"""

import torch


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def backward(self, loss, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)

    def update(self, optimizer):
        self._scaler.unscale_(optimizer)
        self._scaler.step(optimizer)
        self._scaler.update()
        optimizer.zero_grad()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)