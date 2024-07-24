from typing import List
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from gymnasium.envs.registration import register
from setuptools import setup



class NormalTBSOneMap(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
     
    def __init__(self,render_mode = "human",width = 13,height = 13):
        super().__init__()
        self.width = width
        self.height = height
        
        self.origin_map = np.transpose(np.array([
            [ 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [ 3,-1, 0,-1, 0, 0, 0,-1, 0,-1, 0, 0, 0],
            [ 0,-1, 0,-1, 0, 0, 0,-1, 2,-1, 0, 0, 0],
            [ 0,-1, 0,-1, 3, 0, 0,-1,-1,-1, 0, 0, 0],
            [ 0, 0, 0,-1, 0, 0, 0,-1, 4,-1, 0, 0, 0],
            [-1,-1,-1,-1, 0, 0, 0,-1,-1,-1, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 2, 0, 0, 0,-1,-1,-1,-1,-1, 0],
            [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0],
            [ 0, 0,-1,-1,-1, 0, 0,-1, 0,-1, 0,-1, 0],
            [ 0, 0,-1, 0,-1, 0, 0,-1, 0,-1, 0, 0, 0],
            [ 1, 0,-1, 0, 3, 0, 0, 2, 0,-1, 0, 2, 0],
            ],np.int32))
        
        
       

        self.start_pos = [0,5]
        self.agent_pos = self.start_pos.copy()

        self.max_HP = 3
        self.curr_HP = self.max_HP
        
        self.max_bullet = 2
        self.curr_bullet = self.max_bullet

        self.curr_state_index = 0
        self.curr_coin = 0
        self.curr_enemyPoint = 0

        self.curr_map = self.origin_map.copy()


        self.observation_space = spaces.Dict({
            "map":spaces.Box(-5,10,(width,height),np.int32),
            "agent": spaces.Box(0,7,(2 ,),np.int32),
            'bullet': spaces.Discrete(self.max_bullet+1) ,
            "hp": spaces.Discrete(self.max_HP+1),  #hp
            'enemy': spaces.Discrete(7) ,
            "coin": spaces.Discrete(5),  #hp
            })

        
        
            
           
        self.action_space = spaces.MultiDiscrete([3, 4])
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 50  # Size of each cell in pixels
       
    def _get_obs(self):
        return {
            "map": self.curr_map,
            "agent": np.array([self.agent_pos], dtype=np.int32),
            'bullet': self.curr_bullet,
            "hp": self.curr_HP,
            'enemy': self.curr_enemyPoint,
            "coin": self.curr_coin,
        }





    def _get_info(self):
        return {
                "agent": np.array([self.agent_pos], dtype=np.int32),
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

        self.curr_map = self.origin_map.copy()

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
                reward -=1
            else:
                # wall:-1
                if(self.curr_map[next_x,next_y] == -1):
                    reward -=1

                # way
                elif(self.curr_map[next_x,next_y] == 0):
                    reward -=0.2
                    self._update_agent_position(next_x,next_y)

                #coin
                elif(self.curr_map[next_x,next_y] == 2):  
                    reward +=20
                    self.curr_coin +=1   
                    self._update_agent_position(next_x,next_y)

                #enemy
                elif(self.curr_map[next_x,next_y] == 3):

                    reward -= 20
                    self.curr_HP -= 1

                    if self.curr_HP <= 0:
                        reward -= 100
                        terminated = True
                        
                #exit
                elif(self.curr_map[next_x,next_y] == 4):  

                    
                    reward+=20
                    terminated = True
                    self._update_agent_position(next_x,next_y)

                
                     
                    
        elif action_type == 1:#attack
                
                # way
                if(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height or self.curr_map[next_x,next_y] == 0):
                    reward -=50
                #enemy
                elif(self.curr_map[next_x,next_y] == 3):
                    reward += 20  
                    self.curr_enemyPoint +=1
                    self.curr_map[next_x,next_y] =0

        elif action_type == 2:#shoot
            if self.curr_bullet == 0:
                reward -=50
    
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
                        reward += 20 
                        self.curr_enemyPoint +=1
                        if_hit = True
                        self.curr_map[next_x,next_y] =0
                        break
                
                if not if_hit:
                    reward -=50

                
        
        observation = self._get_obs()
        info = self._get_info()
        
        
        
        print(action,info)
        return observation, reward, terminated, truncated, info
     

    
     
    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = (next_x, next_y)
    

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.window.fill((255, 255, 255))  # Fill the screen with white

            # Draw the grid
            for x in range(self.width):
                for y in range(self.height):
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if self.curr_map[x, y] == -1:  # Wall
                        pygame.draw.rect(self.window, (128, 128, 128), rect)
                    elif self.curr_map[x, y] == 0:  # Way
                        pygame.draw.rect(self.window, (255, 255, 255), rect)
                    elif self.curr_map[x, y] == 1:  # Agent
                        pygame.draw.rect(self.window, (173, 216, 230), rect)
                    elif self.curr_map[x, y] == 2:  # Coin
                        pygame.draw.rect(self.window, (255, 255, 0), rect)
                    elif self.curr_map[x, y] == 3:  # Enemy
                        pygame.draw.rect(self.window, (255, 0, 0), rect)
                    elif self.curr_map[x, y] == 4:  # Exit
                        pygame.draw.rect(self.window, (0, 255, 0), rect)

            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

register(
    id='NormalTBS_Env-v2',
    entry_point='Envs.NormalTBSOneMap:NormalTBSOneMap',
    max_episode_steps=3000,
)
           