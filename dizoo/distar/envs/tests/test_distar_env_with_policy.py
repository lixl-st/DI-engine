import os
import shutil

from distar.ctools.utils import read_config
import torch

from dizoo.distar.envs import DIStarEnv
import traceback

from dizoo.distar.envs import DIStarEnv
from dizoo.distar.policy import DIStarPolicy
import traceback

class TestDIstarEnv:
    def __init__(self):

        cfg = read_config('./test_distar_config.yaml')
        self._whole_cfg = cfg
        self._policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['collect']).collect_mode
        rl_model = torch.load('./rl_model.pth')
        
        self._policy.load_state_dict(rl_model)

    def _inference_loop(self, job={}):

        torch.set_num_threads(1)

        self._env = DIStarEnv(self._whole_cfg)

        with torch.no_grad():
            for _ in range(1):
                try:
                    observations = self._env.reset()
                    print(observations.keys())
                    self._policy.reset(observations[0])

                    while True:  # one episode loop
                        # agent step
                        actions = {}
                        for player_index, player_obs in observations.items():
                            output = self._policy.forward(player_obs)
                            actions[player_index] = output['action']
                        # print('-------------------------------------------------------')
                        # print(observations)
                        # print(actions)

                        # env step
                        timestep = self._env.step(actions)
                        if not timestep.done:
                            observations = timestep.obs
                        else:
                            break
                        
                except Exception as e:
                    print('[EPISODE LOOP ERROR]', e, flush=True)
                    print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                    self._env.close()
            self._env.close()

if __name__ == '__main__':

    ## main
    if os.path.exists(r'C:\Program Files (x86)\StarCraft II'):
        sc2path = r'C:\Program Files (x86)\StarCraft II'
    elif os.path.exists('/Applications/StarCraft II'):
        sc2path = '/Applications/StarCraft II'
    else:
        assert 'SC2PATH' in os.environ.keys(), 'please add StarCraft2 installation path to your environment variables!'
        sc2path = os.environ['SC2PATH']
        assert os.path.exists(sc2path), 'SC2PATH: {} does not exist!'.format(sc2path)
    if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2')):
        shutil.copytree(os.path.join(os.path.dirname(__file__), '../envs/maps/Ladder2019Season2'), os.path.join(sc2path, 'Maps/Ladder2019Season2'))

    ## actor_run
    actor = TestDIstarEnv()
    actor._inference_loop()