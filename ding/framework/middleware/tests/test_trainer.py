import pytest
import torch
import copy
from unittest.mock import Mock, patch
from ding.framework import Context
from ding.framework.middleware.functional.trainer import trainer, multistep_trainer
from ding.framework.middleware.tests.mock_for_test import CONFIG

class MockPolicy(Mock):
    # MockPolicy class for train mode
    def forward(self, train_data, **kwargs):
        res = {'total_loss': 1.0}
        return res

class MultiStepMockPolicy(Mock):
    # MockPolicy class for multi-step train mode
    def forward(self, train_data, **kwargs):
        res = [{'total_loss': 1.0}, {'total_loss': 0.1}]
        return res
    

@pytest.mark.lxl
def test_trainer():
    cfg = copy.deepcopy(CONFIG)
    ctx = Context(CONFIG.ctx)
    ctx.train_data = {'data': torch.rand(4, 4)}
    with patch("ding.policy.Policy",  MockPolicy):
        policy = MockPolicy()
        for _ in range(100):
            trainer(cfg, policy)(ctx)
    assert ctx.train_iter == 100
    assert "total_loss" in ctx.train_output


@pytest.mark.lxl
def test_multistep_trainer():
    cfg = copy.deepcopy(CONFIG)
    ctx = Context(CONFIG.ctx)
    ctx.train_data = {'data': torch.rand(4, 4)}
    with patch("ding.policy.Policy",  MultiStepMockPolicy):
        policy = MultiStepMockPolicy()
        for _ in range(100):
            multistep_trainer(cfg, policy)(ctx)
    assert ctx.train_iter == 200
    assert "total_loss" in ctx.train_output[0]
    
