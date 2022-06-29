import time
from copy import deepcopy
from easydict import EasyDict
from ditk import logging
from ding.framework.event_enum import EventEnum
from ding.framework.middleware.functional.actor_data import ActorData, ActorDataMeta, ActorEnvTrajectories
from ding.framework.middleware.league_learner import LeagueLearnerCommunicator
from ding.league.v2.base_league import BaseLeague
from ding.data import DequeBuffer
from ding.utils import DistContext, get_rank
from ding.framework import task
from ding.framework.context import BattleContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, nstep_reward_enhancer, online_logger, ddp_termination_checker
from ding.utils import set_pkg_seed
from ding.utils.sparse_logging import log_every_sec
from dizoo.distar.config import distar_cfg
from dizoo.distar.envs.fake_data import fake_rl_data_batch_with_last
from dizoo.distar.policy.distar_policy import DIStarPolicy

env_cfg = dict(
    actor=dict(job_type='train', ),
    env=dict(
        map_name='KingsCove',
        player_ids=['agent1', 'agent2'],
        races=['zerg', 'zerg'],
        map_size_resolutions=[True, True],  # if True, ignore minimap_resolutions
        minimap_resolutions=[[160, 152], [160, 152]],
        realtime=False,
        replay_dir='.',
        random_seed='none',
        game_steps_per_episode=100000,
        update_bot_obs=False,
        save_replay_episodes=1,
        update_both_obs=False,
        version='4.10.0',
    ),
)
env_cfg = EasyDict(env_cfg)


def coordinator_mocker(cfg):
    task.on(EventEnum.LEARNER_SEND_META, lambda x: logging.info("test: {}".format(x)))
    task.on(EventEnum.LEARNER_SEND_MODEL, lambda x: logging.info("test: send model success"))

    def _coordinator_mocker(ctx):
        time.sleep(1.0)

    return _coordinator_mocker


def actor_mocker(cfg, league):

    def _actor_mocker(ctx):
        players = league.active_players
        log_every_sec(logging.INFO, 5, "Actor: actor player: {}".format(player.player_id))
        for _ in range(6):
            for player in players:
                meta = ActorDataMeta(player_total_env_step=0, actor_id=0, send_wall_time=time.time())
                data = fake_rl_data_batch_with_last()
                actor_data = ActorData(meta=meta, train_data=[ActorEnvTrajectories(env_id=0, trajectories=[data])])
                task.emit(EventEnum.ACTOR_SEND_DATA.format(player=player.player_id), actor_data)
        time.sleep(1.0)

    return _actor_mocker


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = deepcopy(distar_cfg)
    cfg.exp_name = 'distar-multi-gpu-learn-seed0'
    cfg.policy.learn.multi_gpu = True
    with DistContext():
        rank = get_rank()
        with task.start(async_mode=False, ctx=BattleContext()):

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])
            league = BaseLeague(cfg.policy.other.league)
            player = league.active_players[0]

            task.use(coordinator_mocker(cfg))
            task.use(actor_mocker(cfg, league))
            task.use(LeagueLearnerCommunicator(cfg, policy.learn_mode, player, rank))
            task.use(eps_greedy_handler(cfg))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            # if rank == 0:
            #     task.use(CkptSaver(cfg, policy, train_freq=1000))
            #     task.use(online_logger(record_train_iter=True))
            task.run()


if __name__ == "__main__":
    main()
