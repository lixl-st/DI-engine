from copy import deepcopy
from time import sleep
import time
import pytest
import logging

from ding.data import DequeBuffer
from ding.framework import EventEnum
from ding.framework.context import BattleContext
from ding.framework.task import task, Parallel
from ding.framework.middleware import data_pusher, OffPolicyLearner, LeagueLearnerCommunicator
from ding.framework.middleware.functional.actor_data import *
from ding.league.v2 import BaseLeague
from ding.utils import log_every_n
from dizoo.distar.config import distar_cfg
from dizoo.distar.envs import fake_rl_data_batch_with_last
from dizoo.distar.policy.distar_policy import DIStarPolicy


def coordinator_mocker(cfg):
    task.on(EventEnum.LEARNER_SEND_META, lambda x: logging.info("test: {}".format(x)))
    task.on(EventEnum.LEARNER_SEND_MODEL, lambda x: logging.info("test: send model success"))

    def _coordinator_mocker(ctx):
        sleep(1)

    return _coordinator_mocker


def actor_mocker(cfg, league):

    def _actor_mocker(ctx):
        players = league.active_players
        for _ in range(6):
            for player in players:
                log_every_n(logging.INFO, 13, "[Actor 1]: actor player: {}".format(player.player_id))
                meta = ActorDataMeta(player_total_env_step=0, actor_id=0, send_wall_time=time.time())
                data = fake_rl_data_batch_with_last()
                actor_data = ActorData(meta=meta, train_data=[ActorEnvTrajectories(env_id=0, trajectories=[data])])
                task.emit(EventEnum.ACTOR_SEND_DATA.format(player=player.player_id), actor_data)
        time.sleep(5.0)

    return _actor_mocker


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = deepcopy(distar_cfg)
    league = BaseLeague(cfg.policy.other.league)
    n_players = len(league.active_players_ids)
    
    with task.start(async_mode=True, ctx=BattleContext()):
        if task.router.node_id == 0:
            task.use(coordinator_mocker(cfg))
        elif task.router.node_id == 1:
            task.use(actor_mocker(cfg, league))
        else:
            cfg.policy.collect.unroll_len = 1
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            player = league.active_players[task.router.node_id % n_players]
            policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])

            task.use(LeagueLearnerCommunicator(cfg, policy.learn_mode, player))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.run(max_step=1000000)


@pytest.mark.unittest
def test_league_learner():
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(main)
