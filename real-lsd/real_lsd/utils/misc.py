import os
import numpy as np


def load_env_setting(filename):
    f = open(get_settingpath(filename))
    type = os.path.splitext(filename)[1]
    if type == '.json':
        import json
        setting = json.load(f)
    else:
        print ('unknown type')
    return setting


def get_settingpath(filename):
    import real_lsd
    gympath = os.path.dirname(real_lsd.__file__)
    return os.path.join(gympath, 'envs/settings', filename)
