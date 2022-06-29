from typing import TYPE_CHECKING, Dict
from time import sleep
from threading import Lock

from ditk import logging
from ding.framework import task, EventEnum
from ding.utils import log_every_sec

if TYPE_CHECKING:
    from easydict import EasyDict
    from ding.framework import Task, Context
    from ding.league.v2 import BaseLeague
    from ding.league.player import PlayerMeta
    from ding.league.v2.base_league import Job


class LeagueCoordinator:

    def __init__(self, cfg: "EasyDict", league: "BaseLeague") -> None:
        self.league = league
        self._lock = Lock()
        self._total_send_jobs = 0
        self._eval_frequency = 10
        self._running_jobs = dict()

        task.on(EventEnum.ACTOR_GREETING, self._on_actor_greeting)
        task.on(EventEnum.LEARNER_SEND_META, self._on_learner_meta)
        task.on(EventEnum.ACTOR_FINISH_JOB, self._on_actor_job)

    def _on_actor_greeting(self, actor_id):
        logging.info("[Coordinator {}] recieve actor {} greeting".format(task.router.node_id, actor_id))
        with self._lock:
            player_num = len(self.league.active_players_ids)
            player_id = self.league.active_players_ids[self._total_send_jobs % player_num]
            job = self.league.get_job_info(player_id)
            job.job_no = self._total_send_jobs
            self._total_send_jobs += 1
        if job.job_no > 0 and job.job_no % self._eval_frequency == 0:
            job.is_eval = True
        job.actor_id = actor_id
        self._running_jobs["actor_{}".format(actor_id)] = job
        task.emit(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=actor_id), job)

    def _on_learner_meta(self, player_meta: "PlayerMeta"):
        log_every_sec(
            logging.INFO, 5,
            '[Coordinator {}] recieve learner meta from player {}'.format(task.router.node_id, player_meta.player_id)
        )
        self.league.update_active_player(player_meta)
        self.league.create_historical_player(player_meta)

    def _on_actor_job(self, job: "Job"):
        logging.info(
            "[Coordinator {}] recieve actor finished job, player {}".format(task.router.node_id, job.launch_player)
        )
        self.league.update_payoff(job)
    
    def _print_job(self, jobs: Dict[str, "Job"]):
        res = ""
        for actor_id, job in jobs.items():
            res += "{}:\n{}\n".format(actor_id, repr(job))
        return res

    def __del__(self):
        logging.info("[Coordinator {}] all tasks finished, coordinator closed".format(task.router.node_id))

    def __call__(self, ctx: "Context") -> None:
        sleep(1)
        log_every_sec(
            logging.INFO, 60, "[Coordinator {}] running jobs:\n{}".format(task.router.node_id, self._print_job(self._running_jobs))
        )
