from btc.engine.game import BomberEnv
from btc.agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# =========================================================================
# BỘ NHỚ ĐỆM TRẢI NGHIỆM (REPLAY BUFFER) - Nâng cấp để lưu thêm dữ liệu Aux
# =========================================================================
class DQNModel(nn.Module):
    """
    Two-branch DQN:
      - Conv2D branch for spatial map/object channels
      - MLP branch for auxiliary scalar features
    """
    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, map_x, aux_x):
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        feat = torch.cat([map_feat, aux_feat], dim=1)
        return self.head(feat)
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, map_state, aux_state, action, reward, next_map_state, next_aux_state, done):
        # Lưu trữ trọn vẹn cấu trúc dữ liệu của mạng 2 nhánh
        self.buffer.append((map_state, aux_state, action, reward, next_map_state, next_aux_state, done))

    def sample(self, batch_size):
        # Bóc tách cấu trúc dữ liệu đa nhánh từ hàng đợi ngẫu nhiên
        map_s, aux_s, act, rew, next_map_s, next_aux_s, dn = zip(*random.sample(self.buffer, batch_size))
        
        return (np.array(map_s), np.array(aux_s), np.array(act), np.array(rew, dtype=np.float32), 
                np.array(next_map_s), np.array(next_aux_s), np.array(dn, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
    
# =========================================================================
# LỚP HUÂN LUYỆN LOCAL SỬ DỤNG MẠNG HAI NHÁNH (TWO-BRANCH)
# =========================================================================
class LocalTrainer:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = 6
        
        # --- CẤU HÌNH SIÊU THAM SỐ ĐẦU VÀO MẠNG 2 NHÁNH ---
        # Nhánh 1 (Map): 7 channels không gian (0->4 địa hình + 5 vị trí mình + 6 vị trí địch)
        self.map_shape = (7, 13, 13) 
        # Nhánh 2 (Aux): 3 chỉ số phẳng (bombs_left, bomb_radius_bonus, normalized_step)
        self.aux_dim = 3             
        
        # Khởi tạo Double DQN theo class mạng mới của bạn
        self.policy_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        self.target_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer()
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997
        self.batch_size = 128

    def preprocess_obs(self, obs: dict, current_step: int) -> tuple:
        """
        SỬA ĐỔI QUAN TRỌNG: Tách obs thành 2 phần:
          - map_grid (7, 13, 13): Cho nhánh CNN
          - aux_vector (3,): Cho nhánh MLP
        """
        # 1. Xử lý nhánh Địa hình Không gian (7 Kênh)
        map_grid = np.zeros((7, 13, 13), dtype=np.float32)
        game_map = obs['map']
        for i in range(5):
            map_grid[i] = (game_map == i).astype(np.float32)
            
        for p_id, player in enumerate(obs['players']):
            if player[2] == 1: # Nếu alive == 1
                map_grid[5 if p_id == self.agent_id else 6][player[0]][player[1]] = 1.0
                
        # 2. Xử lý nhánh Thông số phẳng (Auxiliary Scalars)
        my_player = obs['players'][self.agent_id]
        bombs_left = float(my_player[3])
        radius_bonus = float(my_player[4])
        normalized_step = float(current_step) / 500.0 # Chuẩn hóa step về khoảng [0, 1]
        
        aux_vector = np.array([bombs_left, radius_bonus, normalized_step], dtype=np.float32)
        
        return map_grid, aux_vector

    def choose_action(self, map_grid, aux_vector):
        """Chọn hành động tích hợp dữ liệu 2 nhánh"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Chuyển đổi cả 2 mảng thành Tensor PyTorch và đẩy lên thiết bị tính toán
        tensor_map = torch.FloatTensor(map_grid).unsqueeze(0).to(self.device)
        tensor_aux = torch.FloatTensor(aux_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Nạp song song cả 2 nhánh dữ liệu vào hàm forward(map_x, aux_x) của bạn
            q_values = self.policy_net(tensor_map, tensor_aux)
            
        return torch.argmax(q_values, dim=1).item()

    def shape_reward(self, obs_old, action, obs_new):
        reward = 0.0
        my_old = obs_old['players'][self.agent_id]
        my_new = obs_new['players'][self.agent_id]
        
        if my_new[2] == 0: return -30.0
        reward += 0.005
        
        if my_new[4] > my_old[4]: reward += 2.0
        if my_new[3] > my_old[3]: reward += 2.0
        return reward

    def train_step(self):
        """Bước huấn luyện tối ưu hóa đồ thị đạo hàm đa nhánh"""
        if len(self.memory) < self.batch_size:
            return
        
        # Lấy mẫu dữ liệu đa nhánh từ Replay Buffer
        map_s, aux_s, actions, rewards, next_map_s, next_aux_s, dones = self.memory.sample(self.batch_size)
        
        # Chuyển đổi toàn bộ cấu trúc sang Tensor PyTorch
        map_s_t = torch.FloatTensor(map_s).to(self.device)
        aux_s_t = torch.FloatTensor(aux_s).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_map_s_t = torch.FloatTensor(next_map_s).to(self.device)
        next_aux_s_t = torch.FloatTensor(next_aux_s).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Tính Q(s, a) hiện tại bằng cách nạp cả map và aux vào mạng chính
        current_q = self.policy_net(map_s_t, aux_s_t).gather(1, actions_t).squeeze(1)
        
        # Tính Q-target bằng cách nạp map và aux kế tiếp vào mạng target
        with torch.no_grad():
            next_actions = self.policy_net(
                next_map_s_t,
                next_aux_s_t
            ).argmax(1, keepdim=True)

            next_q = self.target_net(
                next_map_s_t,
                next_aux_s_t
            ).gather(1, next_actions).squeeze(1)

            target_q = rewards_t + (
                self.gamma * next_q * (1 - dones_t)
            )

        # Lan truyền ngược cập nhật toàn bộ tham số của cả 2 encoder cùng lúc
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="model.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print(f"--> Đã lưu trọng số mạng 2 nhánh vào file: {filename}")


# =========================================================================
# VÒNG LẶP HUẤN LUYỆN TÍCH HỢP ENGINE VÀ TRACKER CHỈ SỐ
# =========================================================================
if __name__ == "__main__":
    num_episodes = 2000
    max_steps = 500
    seed = 42 
    
    env = BomberEnv(max_steps=max_steps, seed=seed)
    my_trainer = LocalTrainer(agent_id=0)
    
    btc_bot_1 = SmarterRuleAgent(1)
    btc_bot_2 = BoxFarmerAgent(2)
    btc_bot_3 = SimpleRuleAgent(3)
    
    win_history = deque(maxlen=100)       
    reward_history = deque(maxlen=100)    
    survival_history = deque(maxlen=100)  
    best_winrate = -1

    print("=== Khởi động hệ thống huấn luyện thực tế (Mạng 2 nhánh) ===")
    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for episode in range(1, num_episodes + 1):
            episode_seed = None if seed is None else seed + episode
            obs = env.reset(seed=episode_seed)
            done = False
            step = 0
            episode_reward = 0.0

            while not done and step < max_steps:
                # --- BƯỚC 1: Tiền xử lý obs ra cặp dữ liệu (Map, Aux) ---
                map_grid, aux_vector = my_trainer.preprocess_obs(obs, current_step=step)
                
                # Chọn hành động từ luồng dữ liệu kép
                action_0 = my_trainer.choose_action(map_grid, aux_vector)
                
                action_1 = btc_bot_1.act(obs)
                action_2 = btc_bot_2.act(obs)
                action_3 = btc_bot_3.act(obs)
                
                actions = [action_0, action_1, action_2, action_3]
                
                # --- BƯỚC 2: Đẩy mảng hành động vào Engine thật ---
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated
                
                # --- BƯỚC 3: Tính toán obs mới và lưu cặp trạng thái kép vào Buffer ---
                if obs["players"][0][2] == 1:
                    reward = my_trainer.shape_reward(obs, action_0, next_obs)
                    if done:
                        alive_final = [bool(p[2]) for p in next_obs["players"]]
                        survivors = [i for i in range(4) if alive_final[i]]

                        # Win
                        if len(survivors) == 1 and survivors[0] == 0:
                            reward += 100.0

                        # Lose
                        elif len(survivors) == 1:
                            reward -= 20.0

                        # Draw
                        else:
                            reward += 10.0
                    episode_reward += reward
                    
                    # Tạo cặp dữ liệu tương lai cho bước học kế tiếp
                    next_map_grid, next_aux_vector = my_trainer.preprocess_obs(next_obs, current_step=step+1)
                    
                    # Đẩy toàn bộ 7 tham số MDP của cấu trúc đa nhánh vào bộ nhớ
                    my_trainer.memory.push(map_grid, aux_vector, action_0, reward, 
                                        next_map_grid, next_aux_vector, done)
                
                obs = next_obs
                step += 1
                
                # --- BƯỚC 4: Huấn luyện mạng Neural qua từng step ---
                my_trainer.train_step()
                
            # --- BƯỚC 5: Ghi nhận thông tin và in báo cáo phong độ ---
            alive_final = [bool(p[2]) for p in obs["players"]]
            survivors = [i for i in range(4) if alive_final[i]]
            is_win = 1 if (len(survivors) == 1 and survivors[0] == 0) else 0
            win_history.append(is_win)
            reward_history.append(episode_reward)
            survival_history.append(step) 
                
            # Cứ sau 10 trận đấu: Tiến hành đồng bộ hóa trọng số từ mạng chính sang mạng mục tiêu (Target Net)
            if episode % 10 == 0:
                my_trainer.target_net.load_state_dict(
                    my_trainer.policy_net.state_dict()
                )
                
            # Chỉ bắt đầu đánh giá sau khi đủ 100 trận
            if len(win_history) >= 100:
                current_winrate = np.mean(list(win_history)[-100:])

                if current_winrate > best_winrate:
                    best_winrate = current_winrate
                    my_trainer.save_model("best_model.pth")

            # Checkpoint định kỳ
            if episode % 100 == 0:
                avg_win_rate = np.mean(win_history) * 100       # Tỷ lệ thắng trung bình (%)
                avg_reward = np.mean(reward_history)            # Điểm thưởng trung bình nhận được
                avg_survival = np.mean(survival_history)        # Số bước sống sót trung bình
                
                print(f"\n================ BÁO CÁO PHONG ĐỘ (TRẬN {episode}) ================")
                print(f"* Tỷ lệ khám phá (Epsilon)   : {my_trainer.epsilon:.4f}")
                print(f"* Kích thước bộ nhớ (Memory) : {len(my_trainer.memory)}/50000")
                print(f"* TỶ LỆ THẮNG (100 trận gần nhất) : {avg_win_rate:.2f}%")
                print(f"* SỐ STEP SỐNG SÓT TRUNG BÌNH: {avg_survival:.1f}/{max_steps}")
                print(f"* ĐIỂM THƯỞNG TRUNG BÌNH     : {avg_reward:.2f}")
                print("================================================================\n")
                my_trainer.save_model("last_model.pth")
            pbar.update(1)
            # pbar.set_postfix(step=f"{step:.2f}", episode_reward=f"{episode_reward:.3f}", win=f"{"YES" if is_win else "NO"}", memory=f"{len(my_trainer.memory)}/50000", epsilon=f"{my_trainer.epsilon:.3f}")
            pbar.set_postfix(st=f"{step}", re=f"{episode_reward:.2f}", win=f"{"Y" if is_win else "N"}")