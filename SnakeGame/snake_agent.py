import torch
import random
import numpy as np
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from snake_train import Linear_QNet, QTrainer
from utils import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARN_RATE = 0.001

class SnakeAgent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LEARN_RATE, self.gamma)
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_state(self, game:SnakeGameAI):
        head = game.snake[0]
        
        head_point_left = Point(head.x-BLOCK_SIZE, head.y) #subtract block size to go left one block
        head_point_right = Point(head.x+BLOCK_SIZE, head.y) #add to go right
        head_point_up = Point(head.x, head.y-BLOCK_SIZE) #subtract to go up
        head_point_down = Point(head.x, head.y+BLOCK_SIZE) #add to go down
        
        food_point = game.food
        
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN
        
        danger_left = 0
        danger_right = 0
        danger_straight = 0
        
        if direction_left and game.is_collision(head_point_left) or \
           direction_right and game.is_collision(head_point_right) or \
           direction_up and game.is_collision(head_point_up) or \
           direction_down and game.is_collision(head_point_down):
               danger_straight = 1
        if direction_left and game.is_collision(head_point_up) or \
           direction_right and game.is_collision(head_point_down) or \
           direction_up and game.is_collision(head_point_right) or \
           direction_down and game.is_collision(head_point_left):
               danger_right = 1
        if direction_left and game.is_collision(head_point_down) or \
           direction_right and game.is_collision(head_point_up) or \
           direction_up and game.is_collision(head_point_left) or \
           direction_down and game.is_collision(head_point_right):
               danger_left = 1
        
        food_left = food_point.x < head.x
        food_right = food_point.x > head.x
        food_up = food_point.y < head.y
        food_down = food_point.y > head.y
          
        return np.array([danger_straight,
                          danger_right,
                          danger_left,
                          direction_left,
                          direction_right,
                          direction_up,
                          direction_down,
                          food_left,
                          food_right,
                          food_up,
                          food_down], dtype=int)
        
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #pop left is max memory reached
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory
            
        state, action, reward, next_state, done = zip(*mini_batch)
        
        self.trainer.train_step(state,action,reward,next_state,done)  
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self,state):
        #tradeoff exploration / exploitation
        
        self.epsilon = 80 - self.num_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float32).to(self.device)
            predictions = self.model(state0)
            move = torch.argmax(predictions).item()
            final_move[move] = 1
            
        return final_move
            
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = SnakeAgent()
    game = SnakeGameAI()
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            #train long memory
            
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print(f"Game: {agent.num_games} Score: {score} Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()