## Reinforcement learning for landing MAVs
***ReaL LSD***
This repository was developed as a semester project at the Autonomous Systems Lab ETH Zurich. Extending on work done in gym-unrealcv, the repository was built to investigate the feasibility of deep reinforcement learning for solving the problem of landiing MAVs in emergency situations. A custom scene and MAV model were built in Unreal Engine which served as the environment and agent respectively. The agent was trained using Proximal Policy Optimization (PPO) and truncated Generalized Advantage Estimation (GAE).

### Setup

The real_lsd package extends and inherits classes from the [Gym-UnrealCV](https://github.com/zfw1226/gym-unrealcv) package and as such depends on it. Install gym-unrealcv and setup the package using

```
git clone https://github.com/zfw1226/gym-unrealcv.git
cd gym-unrealcv
pip install -e .
```

and install `OpenCV`. If you are using `anaconda` or `miniconda`, see the section on installation in the [README](https://github.com/zfw1226/gym-unrealcv#install-gym-unrealcv) of gym-unrealcv.

Installing the real_lsd setting up the package can be done similarly

```
git clone https://github.com/nanaimi/real-lsd.git
cd real-lsd
pip install -e .
```

Once the `real-lsd` repository is installed, copy the game binary used as the environment for training and testing the agent from `real-lsd/real_lsd/envs/UnrealEnvs/` into 'gym-unrealcv/gym_unrealcv/env/UnrealEnv/' using

```
cp -R <yourpath>/real-lsd/real_lsd/envs/UnrealEnvs/<Binary> <yourpath>/gym-unrealcv/gym_unrealcv/env/UnrealEnv/
```

### Registering Environments in OpenAI's Gym
Following the instructions found on [OpenAI's Gym Registration](https://gym.openai.com/docs/#the-registry), custom environment can be registered by using `register()` at load time. This is done in the `__init__.py` file of real_lsd and can be seen [here](https://github.com/nanaimi/real-lsd/blob/d8514ac414e987b86fc3f40e4fb5b101e002b529/real_lsd/__init__.py#L19-L31).

### Testing UnrealCV

To test [UnrealCV](https://github.com/unrealcv/unrealcv) in a packaged game built on Unreal Engine, download any game binaries built with the UnrealCV Plugin. Examples can be found in the model zoo [here](http://docs.unrealcv.org/en/master/reference/model_zoo.html).

After downloading the game binary, start the game and press the key ` twice to open the unrealCV console. Test different commands for interacting with the game in the console. Available requests can be found in the [UnrealCV Wiki](http://docs.unrealcv.org/en/master/reference/commands.html).

Alternatively, you can test the UnrealCV python client by launching the `examples/test.ipynb ` notebook.

### Training and Testing PPO Agent

The implementation of the PPO agent can be found under `agents/ppo_agent`, including the python executable that trains the agent and runs it through multiple episodes for testing. Run the following to train a new agent with PPO and test it in 50 episodes and specify the path to which the data collected throughout training should be stored.

```
python agents/ppo_agent/lander.py
```

### Testing a trained agent

Test a trained agent in the custom scene built for showing a proof a concept.

TODO
- add model model parameters
- add demo notebook

### Custom environments

TODO

### Other Dependencies
***CUDA*** = 11.0
