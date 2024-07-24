import gymnasium as gym
from Envs.NormalTBS import NormalTBS
from Envs.NormalTBSOneMap import NormalTBSOneMap
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
import train



def Moduletrain(save_path,log_path,env,times = 2000000):
    model = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    learning_rate=1e-4,  # 学习率
    gamma=0.995,  # 折扣因子
    gae_lambda=0.95,  # GAE λ
    clip_range=0.2,  # 剪辑范围
    ent_coef=0.2,  # 熵系数
    batch_size=512,  # 批大小
    n_steps=256,  # 步数
    n_epochs=16,  # 训练次数
    policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=1),  # LSTM 设置
    verbose=1,
    )
    
    train.train(model,env,times,save_path,log_path,5)


def Normal_train():
    Normal_save_path = 'trained_modules/normalModuleOneMap/normalOneMap_best'
    Normal_log_path = 'logs/normalOneMap_Log'
    Normal_env = make_vec_env("NormalTBS_Env-v2",monitor_dir=Normal_log_path)
    Moduletrain(Normal_save_path,Normal_log_path,Normal_env,6000000)



def main():
    Normal_train()




main()