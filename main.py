import gym_super_mario_bros as mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation  # reinforcement api
from stable_baselines3.common.vec_env import vec_frame_stack, dummy_vec_env # Pytorch version of Stable Baselines
import os
from stable_baselines3.common.vec_env import subproc_vec_env
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

def preprocess(num_cpus=4, test=False):
    #plt.imshow(env.reset())
    #plt.pause(5)
    #plt.imshow(env.reset())
    #plt.pause(5)
    #print([env],[lambda : env])
    if not test:
        env = [lambda : GrayScaleObservation(game(),keep_dim=True) for _ in range(num_cpus)]
        env = subproc_vec_env.SubprocVecEnv(env)
    else:
        env = game()
        env = dummy_vec_env.DummyVecEnv(env)
    #env = vec_frame_stack.VecFrameStack(env, 4, channels_order='last') # number of stack to remember
    #print(env.reset().shape)
    return env

def learn():
    env = preprocess()
    callBack = TrainAndLoggingCallback(checkFreq=5000//4,savePath=CHECKPOINT_DIR)
    model =  PPO('CnnPolicy', env, verbose=1, learning_rate=10**-4, n_steps=512,device='cuda')
    model.learn(total_timesteps=5*10**4, callback=callBack)
    env.close()
    
    
def test():
    env = preprocess(test=True)
    model = PPO.load('./train/best_model_10000')
    state = env.reset()
    try:
        while True:
            action, state = model.predict(state)
            state, reward, done, info = env.step(action)
            env.render()
    except KeyboardInterrupt:
        env.close()

def main():
    #learn()
    test()


if __name__ == '__main__':
    main()
