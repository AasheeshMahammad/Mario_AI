import gym_super_mario_bros as mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation  # reinforcement api
from stable_baselines3.common.vec_env import vec_frame_stack, dummy_vec_env # Pytorch version of Stable Baselines
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT_DIR = './train'
LOG_DIR = './logs'


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, checkFreq, savePath, verbose=1):
        super(TrainAndLoggingCallback,self).__init__(verbose)
        self.checkFreq = checkFreq
        self.savePath = savePath
    
    def _init_callback(self):
        if self.savePath is not None:
            os.makedirs(self.savePath,exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.checkFreq == 0:
            modelPath = os.path.join(self.savePath,f'best_model_{self.n_calls}')
            self.model.save(modelPath)
        return True

def game():
    env = mario.make('SuperMarioBros-v0')  # environment of game
    env = JoypadSpace(env,SIMPLE_MOVEMENT)
    '''done = True
    try:
        for step in range(10**5):
            if done:
                env.reset()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            env.render()
    except KeyboardInterrupt:
        env.close()'''
    return env

def preprocess(env):
    #plt.imshow(env.reset())
    #plt.pause(5)
    env = GrayScaleObservation(env,keep_dim=True) # reduces color space => r,g,b - > w/b
    #plt.imshow(env.reset())
    #plt.pause(5)
    #print([env],[lambda : env])
    env = dummy_vec_env.DummyVecEnv([lambda :env])
    env = vec_frame_stack.VecFrameStack(env, 4, channels_order='last') # number of stack to remember
    #print(env.reset().shape)
    return env

def learn(env):
    callBack = TrainAndLoggingCallback(checkFreq=10000,savePath=CHECKPOINT_DIR)
    model =  PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=10**-6, n_steps=512)
    model.learn(total_timesteps=2*10**4, callback=callBack)
    
    
def test(env):
    model = PPO.load('./train/best_model_20000')
    state = env.reset()
    try:
        while True:
            action, state = model.predict(state)
            state, reward, done, info = env.step(action)
            env.render()
    except KeyboardInterrupt:
        env.close()

def main():
    env = game()
    env = preprocess(env)
    learn(env)
    test(env)


if __name__ == '__main__':
    main()
