import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from metrics import MetricLogger
from agent import Mario

from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, TransformObservation, FrameStackObservation

from smb_env_cynes import SuperMarioBrosEnv
# import gym_smb_cynes

from joypad_space import JoypadSpace

from metrics import MetricLogger

from wrappers import ResizeObservation, SkipFrame
    
def make_env():
    def _thunk():
        env = SuperMarioBrosEnv(rom_path='super-mario-bros-rectangle.nes', headless=False)
        env = JoypadSpace(
            env,
            [['right', 'B'],
             ['right', 'B', 'A']]
        )
        env = SkipFrame(env, skip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = TransformObservation(env, lambda x: x / 255., env.observation_space)
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _thunk

def main():
    num_envs = 2  # Set the number of environments you want to run in parallel
    env_fns = [make_env() for _ in range(num_envs)]
    envs = AsyncVectorEnv(env_fns)

    envs.reset()

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
    mario = Mario(state_dim=(4, 84, 84), action_dim=2, save_dir=save_dir, checkpoint=checkpoint)

    logger = MetricLogger(save_dir)

    episodes = 40000

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):

        states, infos = envs.reset()

        # Initialize cumulative reward for each environment
        cumulative_rewards = [0] * num_envs

        # Play the game!
        while True:
            
            # 4. Run agent on the states
            actions = [mario.act(states[i]) for i in range(num_envs)]

            # 5. Agents perform actions
            next_states, rewards, dones, truncs, infos = envs.step(actions)

            # 6. Remember
            for i in range(num_envs):
                mario.cache(states[i], next_states[i], actions[i], rewards[i], dones[i])

            # 7. Learn
            q, loss = mario.learn()

            # 8. Logging
            for i in range(num_envs):
                cumulative_rewards[i] += rewards[i]
                logger.log_step(rewards[i], loss, q)

            # 9. Update states
            states = next_states

            # 10. Check if end of game
            if all(dones):
                break

        # Log cumulative rewards for all environments
        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )

if __name__ == '__main__':
    main()