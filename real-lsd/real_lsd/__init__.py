from gym.envs.registration import register
import logging
import glog as log
# from gym_unrealcv.envs.utils.misc import load_env_setting
from real_lsd.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker

'''# ------------------------------------------------------------------------------
# Landing for MAVs
# ------------------------------------------------------------------------------
for env in ['cpptest']:
    setting_file = 'landing/{env}.json'.format(env=env)
    settings = load_env_setting(setting_file)
    for i, reset in enumerate(['random']): # , 'waypoint', 'testpoint'
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'PoseColor', 'HeightFeatures', 'StepHeightFeatures', 'StepHeightVelocityFeatures']: # observation type
                for category in settings['targets']:
                    register(
                        id='MyUnrealLand-{env}{category}-{action}{obs}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i),
                        entry_point='real_lsd.envs:UnrealCvLanding_base',
                        kwargs={'setting_file': 'landing/{env}.json'.format(env=env),
                                'category': category,
                                'reset_type': reset,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'mask',  # mask, bbox, distance, bbox_distance
                                'docker': use_docker,
                                },
                        max_episode_steps=200
                    )
                    log.info('Registered env: '+ 'MyUnrealLand-{env}{category}-{action}{obs}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i))
'''

# gorner

env = 'gorner'
setting_file = 'landing/{env}.json'.format(env=env)
settings = load_env_setting(setting_file)
resolution = settings['resolution']
action ='Discrete'
obs = settings['obs']
category = 'FloorGood'
max_steps = settings['maxsteps']
for i, reset in enumerate(['random']):

	            register(
	                id='MyUnrealLand-{env}{category}-{action}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i),
	                entry_point='real_lsd.envs:UnrealCvLanding_base',
	                kwargs={'setting_file': 'landing/{env}.json'.format(env=env),
	                        'category': category,
	                        'reset_type': reset,
	                        'action_type': action,
	                        'observation_type': obs,
	                        'reward_type': 'mask',  # mask, bbox, distance, bbox_distance
	                        'docker': use_docker,
                                'resolution':(resolution[0], resolution[1]),
	                        },
	                max_episode_steps=max_steps
	            )
	            log.info('Registered env: '+ 'MyUnrealLand-{env}-{action}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i))

