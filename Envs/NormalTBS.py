from typing import List
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from gymnasium.envs.registration import register
from setuptools import setup



class NormalTBS(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
     
    def __init__(self,render_mode = "human",width = 6,height = 6):
        super().__init__()
        self.width = width
        self.height = height
        
        self.origin_map_lv1 = np.transpose(np.array([
            [ 0, 2,-1, 0, 0, 4],
            [ 0, 0,-1, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0,-1, 0],
            [ 0, 0, 0,-1, 0, 0],
            [ 1, 0, 0,-1, 0, 3],
            ]))
        
        
        self.origin_map_lv2 = np.transpose(np.array([
            [ 3,-1, 0, 2, 0, 4],
            [ 0, 0, 0,-1, 0, 0],
            [-1,-1,-1, 0, 0, 3],
            [ 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0,-1, 0, 0],
            [ 1, 0, 0, 0, 2, 0],
            ]))
        
        self.origin_map_lv3 = np.transpose(np.array([
            [ 3, 0, 2, 0, 0, 4],
            [ 0,-1, 0,-1, 0, 0],
            [-1, 0, 0, 0, 3, 0],
            [ 0, 0, 0,-1, 0, 0],
            [ 0, 0, 0, 0,-1,-1],
            [ 1, 0, 0, 0,-1, 3],
            ]))
        
        self.state_list = [self.origin_map_lv1,self.origin_map_lv2,self.origin_map_lv3]

        self.start_pos = [0,5]
        self.agent_pos = self.start_pos.copy()

        self.max_HP = 3
        self.curr_HP = self.max_HP
        
        self.max_bullet = 2
        self.curr_bullet = self.max_bullet

        self.curr_state_index = 0
        self.curr_coin = 0
        self.curr_enemyPoint = 0

        self.curr_map = self.state_list[self.curr_state_index].copy()


        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0,7,(2 ,),np.int32),
            'Lv':spaces.Discrete(5) ,
            'bullet': spaces.Discrete(self.max_bullet+1) ,
            "hp": spaces.Discrete(self.max_HP+1),  #hp
            'enemy': spaces.Discrete(7) ,
            "coin": spaces.Discrete(5),  #hp
            })

        
        
            
           
        self.action_space = spaces.MultiDiscrete([3, 4])
        

       
    def _get_obs(self):
        return {
            "agent": np.array([self.agent_pos], dtype=np.int32),
            'Lv': self.curr_state_index+1,
            'bullet': self.curr_bullet,
            "hp": self.curr_HP,
            'enemy': self.curr_enemyPoint,
            "coin": self.curr_coin,
        }





    def _get_info(self):
        return {
                "agent": np.array([self.agent_pos], dtype=np.int32),
                'Lv': self.curr_state_index+1,
                'bullet': self.curr_bullet,
                "hp": self.curr_HP,
                'enemy': self.curr_enemyPoint,
                "coin": self.curr_coin,
                }

    def reset(self,seed=None, options=None):
    
        self.agent_pos = self.start_pos.copy()
        self.curr_HP = self.max_HP
        self.curr_bullet = self.max_bullet

        self.curr_state_index = 0
        self.curr_coin = 0
        self.curr_enemyPoint = 0

        self.curr_map = self.state_list[self.curr_state_index].copy()

        return self._get_obs() , self._get_info()
      
     
    def step(self, action):
          
        action_type = action[0]
        action_dic = action[1]

        next_x= self.agent_pos[0]
        next_y =self.agent_pos[1]
        reward = 0
        terminated = False
        truncated =False
        

        
        if(action_dic == 0):#up
            next_y-=1
        elif(action_dic == 1):#down
            next_y+=1
        elif(action_dic == 2):#right
            next_x+=1
        elif(action_dic == 3):#left
            next_x-=1
        

        if action_type == 0:#move
            
            if(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height):
                reward -=5
            else:
                # wall:-1
                if(self.curr_map[next_x,next_y] == -1):
                    reward -=5

                # way
                elif(self.curr_map[next_x,next_y] == 0):
                    reward -=0.3
                    self._update_agent_position(next_x,next_y)

                #coin
                elif(self.curr_map[next_x,next_y] == 2):  
                    reward +=10
                    self.curr_coin +=1   
                    self._update_agent_position(next_x,next_y)

                #enemy
                elif(self.curr_map[next_x,next_y] == 3):

                    reward -= 10
                    self.curr_HP -= 1

                    if self.curr_HP <= 0:
                        reward -= 100
                        terminated = True
                        
                #exit
                elif(self.curr_map[next_x,next_y] == 4):  

                    reward+=10
                    self.curr_state_index +=1

                    if  self.curr_state_index > 2:
                        reward+=20
                        terminated = True
                    else:
                        self.curr_map = self.state_list[self.curr_state_index].copy()
                        self.agent_pos = self.start_pos

                
                     
                    
        elif action_type == 1:#attack
                
                # way
                if(self.curr_map[next_x,next_y] == 0):
                    reward -=10
                #enemy
                elif(self.curr_map[next_x,next_y] == 3):
                    reward += 10
                    self.curr_enemyPoint +=1

        elif action_type == 2:#shoot
            if self.curr_bullet == 0:
                reward -=10
    
            else:
                self.curr_bullet -=1
                if_hit = False
                for i in range(3):
                    if(action_dic == 0):#up
                        next_y-=i
                    elif(action_dic == 1):#down
                        next_y+=i
                    elif(action_dic == 2):#right
                        next_x+=i
                    elif(action_dic == 3):#left
                        next_x-=i
                    if(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height):
                        break
                    #enemy
                    elif(self.curr_map[next_x,next_y] == 3):
                        reward += 10
                        self.curr_enemyPoint +=1
                        if_hit = True
                        break
                
                if not if_hit:
                    reward -=15

                
        
        observation = self._get_obs()
        info = self._get_info()

        
        
        
        return observation, reward, terminated, truncated, info
     

    
     
    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = (next_x, next_y)
    


register(
    id='NormalTBS_Env-v0',
    entry_point='Envs.NormalTBS:NormalTBS',
    max_episode_steps=3000,
)
           