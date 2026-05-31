import numpy as np
import torch
import torch.nn as nn
import math
import time
import os
import copy
import random
from DQN import DQN

# =========================================================================
# LIGHTWEIGHT SIMULATOR - GIẢ LẬP NHANH ĐỂ MCTS DUYỆT TƯƠNG LAI
# =========================================================================
class MiniEngine:
    @staticmethod
    def step(obs: dict, my_action: int, agent_id: int) -> dict:
        next_obs = {
            "map": np.copy(obs["map"]),
            "players": np.copy(obs["players"]),
            "bombs": np.copy(obs["bombs"])
        }
        my_player = next_obs['players'][agent_id]
        if my_player[2] == 0: return next_obs
            
        r, c = my_player[0], my_player[1]
        if my_action == 1: c_new = max(0, c - 1); r_new = r
        elif my_action == 2: c_new = min(12, c + 1); r_new = r
        elif my_action == 3: r_new = max(0, r - 1); c_new = c
        elif my_action == 4: r_new = min(12, r + 1); c_new = c
        else: r_new, c_new = r, c
        
        cell_type = next_obs['map'][r_new][c_new]
        bomb_blocking = any(b[0] == r_new and b[1] == c_new for b in next_obs['bombs']) if (r_new != r or c_new != c) else False
        
        if cell_type not in [1, 2] and not bomb_blocking:
            my_player[0], my_player[1] = r_new, c_new
            
        if my_action == 5 and my_player[3] > 0:
            bomb_exists = any(b[0] == r and b[1] == c for b in next_obs['bombs'])
            if not bomb_exists:
                new_bomb = np.array([r, c, 7, agent_id], dtype=np.int8)
                if next_obs['bombs'].size == 0:
                    next_obs['bombs'] = np.array([new_bomb])
                else:
                    next_obs['bombs'] = np.vstack([next_obs['bombs'], new_bomb])
                my_player[3] -= 1

        if next_obs['bombs'].size > 0:
            next_obs['bombs'][:, 2] -= 1
            chain_triggered = True
            while chain_triggered:
                chain_triggered = False
                for i, b in enumerate(next_obs['bombs']):
                    if b[2] <= 0 and b[2] > -99:
                        br, bc, _, b_owner = b
                        radius = 1 + next_obs['players'][b_owner][4]
                        if (my_player[0] == br and abs(my_player[1] - bc) <= radius) or \
                           (my_player[1] == bc and abs(my_player[0] - br) <= radius):
                            my_player[2] = 0
                        for j, other_b in enumerate(next_obs['bombs']):
                            if i != j and other_b[2] > 0:
                                if (other_b[0] == br and abs(other_b[1] - bc) <= radius) or \
                                   (other_b[1] == bc and abs(other_b[0] - br) <= radius):
                                    next_obs['bombs'][j][2] = 0
                                    chain_triggered = True
                        next_obs['bombs'][i][2] = -99
            next_obs['bombs'] = np.array([b for b in next_obs['bombs'] if b[2] > 0])
            if next_obs['bombs'].size == 0:
                next_obs['bombs'] = np.empty((0, 4), dtype=np.int8)
        return next_obs

class MCTSNode:
    def __init__(self, obs: dict, parent=None, action_leading_here=None):
        self.obs = obs
        self.parent = parent
        self.action_leading_here = action_leading_here
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == 6

# =========================================================================
# CLASS AGENT BẮT BUỘC THEO INTERFACE CỦA BAN TỔ CHỨC
# =========================================================================
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 8
        self.num_actions = 6
        
        # Thiết lập mốc thời gian an toàn
        self.time_limit = 0.085       
        self.max_mcts_depth = 3       

        # Khởi tạo mạng DQN để làm hàm đánh giá trực giác cho MCTS
        self.rl_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.rl_net.eval()
        
        # Nạp trọng số thu được từ file train.py (phải đặt cùng thư mục file zip)
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        if os.path.exists(model_path):
            try:
                self.rl_net.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception:
                pass

    def _preprocess_obs(self, obs: dict) -> torch.Tensor:
        state_grid = np.zeros((self.input_channels, 13, 13), dtype=np.float32)
        game_map = obs['map']
        for i in range(5):
            state_grid[i] = (game_map == i).astype(np.float32)
        for p_id, player in enumerate(obs['players']):
            if player[2] == 1:
                state_grid[5 if p_id == self.agent_id else 6][player[0]][player[1]] = 1.0
        if obs['bombs'].size > 0:
            for bomb in obs['bombs']:
                state_grid[7][int(bomb[0])][int(bomb[1])] = (8.0 - float(bomb[2])) / 7.0
        return torch.FloatTensor(state_grid).unsqueeze(0).to(self.device)

    def _evaluate_via_rl(self, obs: dict) -> float:
        if obs['players'][self.agent_id][2] == 0: 
            return -20.0
        tensor_state = self._preprocess_obs(obs)
        with torch.no_grad():
            q_values = self.rl_net(tensor_state)
        return torch.max(q_values).item()

    def _uct_select(self, node: MCTSNode, c=1.414) -> int:
        best_score = -float('inf')
        best_action = 0
        for action, child in node.children.items():
            exploitation = child.total_value / child.visit_count
            exploration = c * math.sqrt(math.log(node.visit_count) / child.visit_count)
            uct_value = exploitation + exploration
            if uct_value > best_score:
                best_score = uct_value
                best_action = action
        return best_action

    def act(self, obs: dict) -> int:
        start_time = time.time()
        root_node = MCTSNode(obs=obs)
        
        try:
            while (time.time() - start_time) < self.time_limit:
                node = root_node
                depth = 0
                
                # 1. Selection
                while node.is_fully_expanded() and depth < self.max_mcts_depth:
                    action = self._uct_select(node)
                    node = node.children[action]
                    depth += 1
                
                # 2. Expansion
                if depth < self.max_mcts_depth and not node.is_fully_expanded():
                    unexpanded_actions = [a for a in range(6) if a not in node.children]
                    chosen_action = random.choice(unexpanded_actions)
                    simulated_next_obs = MiniEngine.step(node.obs, chosen_action, self.agent_id)
                    child_node = MCTSNode(obs=simulated_next_obs, parent=node, action_leading_here=chosen_action)
                    node.children[chosen_action] = child_node
                    node = child_node
                
                # 3. Simulation via RL Network
                value = self._evaluate_via_rl(node.obs)
                
                # 4. Backpropagation
                curr_node = node
                while curr_node is not None:
                    curr_node.visit_count += 1
                    curr_node.total_value += value
                    curr_node = curr_node.parent
            
            if root_node.children:
                best_action = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]
                return int(best_action)
            return 0
            
        except Exception:
            return 0