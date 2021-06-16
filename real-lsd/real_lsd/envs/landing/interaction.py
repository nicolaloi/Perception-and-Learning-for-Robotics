import gym
from gym import spaces

import time
import glog as log
import numpy as np
import distutils.version
import PIL.Image
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torchsummary as summary

from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv

class Landing(UnrealCv):
    def __init__(self, env, cam_id=0, port=9000,
                 ip='127.0.0.1', targets=None, resolution=(160, 120)):
        super(Landing, self).__init__(env=env, port=port, ip=ip, cam_id=cam_id, resolution=resolution)
        self.lock = 0
        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)

        # Initialise feature extraction Network
        # Load in pretrained mobilenet V2 network and reduce to feature extractor
        mobilenet = models.mobilenet_v2(pretrained=True)
        mobilenet.eval()
        self.feature_network = nn.Sequential(*(list(mobilenet.children())[0]))
        self.feature_network.eval()

        # Preprocess data before inference --> not sure why?
        self.preprocess = transforms.Compose([
            transforms.Resize(80),
            transforms.CenterCrop(75),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.img_color = np.zeros(1)
        self.img_depth = np.zeros(1)
        self.features  = np.zeros(1)
        self.height    = np.zeros(1)
        self.step      = 0
        self.velocity  = np.zeros(3)

        self.use_gym_10_api = distutils.version.LooseVersion(gym.__version__) >= distutils.version.LooseVersion('0.10.0') # not sure what this is

    def get_observation(self, cam_id, observation_type, mode='direct'):
        if observation_type == 'Color':
            state = self.img_color == self.read_image(cam_id, 'lit', mode)
            #self.img_color = self.img_color.astype('float32')
            #state = (self.img_color[:,:,:]-128)/128
            #print(self.img_color)
            #print("_______________")
            #print(state)
            #print("/\/\/\/\//\/\/\/\/\/\/\\.\./\/\/\/\/\//\/\/\/\/\/")
        elif observation_type == 'Depth':
            self.img_depth = state = self.read_depth(cam_id)

        elif observation_type == 'Rgbd':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)

        elif observation_type == 'CG':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_gray = self.img_color.mean(2)
            self.img_gray = np.expand_dims(self.img_gray, -1)
            state = np.concatenate((self.img_color, self.img_gray), axis=2)

        elif observation_type == 'PoseColor':
            self.img_color = self.read_image(cam_id, 'lit', mode).flatten()
            self.pose =  np.asarray(self.get_pose(cam_id, type='hard'), dtype=np.float64)
            state = np.concatenate((self.pose, self.img_color), axis=0)

        elif observation_type == 'Features':
            #self.img_depth = self.read_depth(cam_id)

            #self.height = np.asarray([self.get_pose(cam_id, type='hard')[2]], dtype=np.float64)
            #log.warn("height is: {} with dimension: {}".format(self.height, self.height.shape))
            
            #self.features = self.get_features(cam_id, 'lit')
            self.features1 = self.get_features(cam_id, 'lit')
            self.features2 = self.get_features(cam_id, 'depth')
            self.features = np.concatenate((self.features1,self.features2),axis=0)
            state = self.features
            log.warn("Features are dimension: {}".format(self.features.shape))

            #state = np.concatenate((self.height, self.features), axis=0)
            assert (np.count_nonzero(state) > 1)

        elif observation_type == 'StepHeightFeatures':
            self.img_depth = self.read_depth(cam_id)

            self.height = np.asarray([self.get_pose(cam_id, type='hard')[2]], dtype=np.float64)
            log.warn("height is: {} with dimension: {}".format(self.height, self.height.shape))

            step = np.asarray([self.step], dtype=np.float64)
            log.warn("step is: {} with dimension: {}".format(step, step.shape))

            self.features = self.get_features(cam_id, 'lit')
            log.info("Features are dimension: {}".format(self.features.shape))

            state = np.concatenate((step, self.height, self.features), axis=0)
            assert (np.count_nonzero(state) > 1)

        elif observation_type == 'StepHeightVelocityFeatures':
            self.img_depth = self.read_depth(cam_id)

            self.height = np.asarray([self.get_pose(cam_id, type='hard')[2]], dtype=np.float64)
            log.warn("height is: {} with dimension: {}".format(self.height, self.height.shape))

            step = np.asarray([self.step], dtype=np.float64)
            log.warn("step is: {} with dimension: {}".format(step, step.shape))

            vel = self.get_velocity()
            log.info("Velocit is: {} {} {}".format(vel[0], vel[1], vel[2]))

            self.features = self.get_features(cam_id, 'lit')
            log.info("Features are dimension: {}".format(self.features.shape))

            state = np.concatenate((step, self.height, vel, self.features), axis=0)
            assert (np.count_nonzero(state) > 1)

        return state


    def define_observation(self, cam_id, observation_type, mode='direct'):
        state = self.get_observation(cam_id, observation_type, mode)

        if observation_type == 'Color' or observation_type == 'CG':
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=0, high=255, shape=state.shape)

        elif observation_type == 'Depth':
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=0, high=100, shape=state.shape, dtype=np.float16)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=0, high=100, shape=state.shape)

        elif observation_type == 'Rgbd':
            s_high = state
            s_high[:, :, -1] = 100.0  # max_depth
            s_high[:, :, :-1] = 255  # max_rgb
            s_low = np.zeros(state.shape)
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=s_low, high=s_high)

        elif observation_type == 'PoseColor':
            low_bound = np.full(state.shape, 0)
            low_bound[:6] = -np.inf
            high_bound = np.full(state.shape, 255)
            high_bound[:6] = np.inf
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape, dtype=np.float64)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape)

        elif observation_type == 'Features':
            low_bound = np.full(state.shape, -np.inf)
            high_bound = np.full(state.shape, np.inf)
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape, dtype=np.float64)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape)
        
        elif observation_type == 'StepHeightFeatures':
            low_bound = np.full(state.shape, -np.inf)
            high_bound = np.full(state.shape, np.inf)
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape, dtype=np.float64)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape)

        elif observation_type == 'StepHeightVelocityFeatures':
            low_bound = np.full(state.shape, -np.inf)
            high_bound = np.full(state.shape, np.inf)
            if self.use_gym_10_api:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape, dtype=np.float64)  # for gym>=0.10
            else:
                observation_space = spaces.Box(low=low_bound, high=high_bound, shape=state.shape)

        return observation_space

    def set_texture(self, target, color=(1, 1, 1), param=(0, 0, 0), picpath=None, tiling=1, e_num=0): #[r, g, b, meta, spec, rough, tiling, picpath]
        param = param / param.max()
        # color = color / color.max()
        cmd = 'vbp {target} set_mat {e_num} {r} {g} {b} {meta} {spec} {rough} {tiling} {picpath}'
        res = self.client.request(cmd.format(target=target, e_num=e_num, r=color[0], g=color[1], b=color[2],
                               meta=param[0], spec=param[1], rough=param[2], tiling=tiling,
                               picpath=picpath))

    def set_light(self, target, direction, intensity, color): # param num out of range
        cmd = 'vbp {target} set_light {row} {yaw} {pitch} {intensity} {r} {g} {b}'
        color = color/color.max()
        res = self.client.request(cmd.format(target=target, row=direction[0], yaw=direction[1],
                                             pitch=direction[2], intensity=intensity,
                                             r=color[0], g=color[1], b=color[2]))

    def set_skylight(self, target, color, intensity ): # param num out of range
        cmd = 'vbp {target} set_light {r} {g} {b} {intensity} '
        res = self.client.request(cmd.format(target=target, intensity=intensity,
                                             r=color[0], g=color[1], b=color[2]))

    def get_features(self, cam_id, viewmode):
        cmd = 'vget /camera/{cam_id}/{viewmode} png'
        res = None
        while res is None:
            res = self.client.request(cmd.format(cam_id=cam_id, viewmode=viewmode))

        image_rgb = self.decode_png(res)
        image_rgb = image_rgb[:, :, :-1]  # delete alpha channel
        image = image_rgb[:, :, ::-1]  # transpose channel order
        self.img_color = image
        img_pil =  PIL.Image.open(BytesIO(res)).convert('RGB')
        img_tensor = self.preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0)
        features = self.feature_network(img_tensor).detach().numpy().flatten()
        return features

    def set_pose(self, cam_id, pose, mode='hard'):
        log.info("this set_pose was called")
        self.set_location(cam_id, pose[:3])
        log.info("going to set this rotation: {}".format(pose[-3:]))
        self.set_rotation(cam_id, pose[-3:])

    def get_pose(self, cam_id, type='hard'):  # pose = [x, y, z, roll, yaw, pitch]
        if type == 'soft':
            pose = none
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if type == 'hard':
            log.info("Get Pose, Mode=hard, cam_id: {}".format(cam_id))
            pose = None
            self.cam[cam_id]['location'] = self.get_location(cam_id)
            self.cam[cam_id]['rotation'] = self.get_rotation(cam_id)
            pose = self.cam[cam_id]['location'] + self.cam[cam_id]['rotation']
            return pose

    def set_FOV(self, cam_id, FOV):
        cmd = 'vset /camera/{cam_id}/horizontal_fieldofview {FOV}'
        self.client.request(cmd.format(cam_id=cam_id, FOV=FOV))

    def set_step(self, step):
        self.step = step

    def get_step(self):
        return self.step

    def set_velocity(self, velocity):
        vel = np.asarray(velocity, dtype=np.float64)
        self.velocity = vel

    def get_velocity(self):
        return self.velocity

    # Take step size for x y z and use moveto function to get there
    # IN: cam_id, delta_x, delta_y, delta_z
    # OUT:move agent to correct location, returns boolean for collision
    def move_3d(self, cam_id, delta_x, delta_y, delta_z):
        #log.warn("Executing move_3d for cam_id {}".format(cam_id))
        location_now = None
        rotation_now = None
        location_exp = None

        pose = self.get_pose(cam_id)
        location_now = self.cam[cam_id]['location']
        rotation_now = self.cam[cam_id]['rotation']
        #log.warn("Current location: {}".format(location_now))
        #log.warn("Current location: {}, Current rotation: {}".format(location_now, rotation_now))

        # define new desired location
        #log.warn("Passed Deltas: {}, {}, {}".format(delta_x, delta_y, delta_z))
        new_x = location_now[0] + delta_x
        new_y = location_now[1] + delta_y
        new_z = location_now[2] + delta_z
        log.info("new_x: {}".format(new_x))
        log.info("new_y: {}".format(new_y))
        log.info("new_z: {}".format(new_z))
        location_exp = [new_x, new_y, new_z]
        #log.warn("Expecting to move to this location: {}".format(location_exp))

        while self.lock:
            log.info("waiting for lock to open.")
            continue

        if not self.lock:
            log.info("acquiring lock")
            self.lock = 1
            while self.lock:
                log.info("locked.")
                self.moveto(cam_id, location_exp)
                # log.warn("about to set location for cam_id {}".format(cam_id))
                # self.set_location(cam_id, location_exp)
                self.lock = 0
                log.info("unlocked.")
        else:
            log.info("process was locked, skipping this move")

        log.info("Get Pose being called.")

        pose = self.get_pose(cam_id)
        location_now = self.cam[cam_id]['location']
        rotation_now = self.cam[cam_id]['rotation']
        #log.warn("moved to location: {}, rotated to rotation: {}".format(location_now, rotation_now))

        error = self.get_distance(location_now, location_exp, n=3)
        #log.warn("Error: {}".format(error))

        if error < 10: # weird offset
            return False
        else:
            return True
