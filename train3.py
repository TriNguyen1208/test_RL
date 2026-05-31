import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from btc.agent import (
    BoxFarmerAgent,
    GeniusRuleAgent,
    RandomAgent,
    SimpleRuleAgent,
    SmarterRuleAgent,
    TacticalRuleAgent,
)
from btc.engine.game import BomberEnv


# =========================================================================
# MẠNG NEURAL HAI NHÁNH (TWO-BRANCH DQN) - ĐÃ TỐI ƯU KÍCH THƯỚC CNN
# =========================================================================
class DQNModel(nn.Module):
    """Two-branch DQN:

    - Nhánh Conv2D được thêm MaxPool2d để giảm chiều, tránh làm loãng gradient.
    - Nhánh MLP xử lý các thông số phẳng.
    """

    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape

        # THAY ĐỔI: Thêm các tầng MaxPool2d để nén chiều không gian từ 13x13 -> 6x6 -> 3x3
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Khung nhìn còn 6x6
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Khung nhìn còn 3x3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Tính toán chính xác kích thước đầu ra sau flatten của CNN
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)
            # Lúc này conv_out_dim sẽ là 64 * 3 * 3 = 576 (thay vì 10816 như cũ)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Đầu vào của head bây giờ là: 576 (CNN) + 32 (Aux) = 608 chiều (Tỷ lệ cực kỳ cân bằng)
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


# =========================================================================
# BỘ NHỚ ĐỆM TRẢI NGHIỆM (REPLAY BUFFER)
# =========================================================================
class ReplayBuffer:

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        map_state,
        aux_state,
        action,
        reward,
        next_map_state,
        next_aux_state,
        done,
    ):
        self.buffer.append(
            (
                map_state,
                aux_state,
                action,
                reward,
                next_map_state,
                next_aux_state,
                done,
            )
        )

    def sample(self, batch_size):
        map_s, aux_s, act, rew, next_map_s, next_aux_s, dn = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(map_s),
            np.array(aux_s),
            np.array(act),
            np.array(rew, dtype=np.float32),
            np.array(next_map_s),
            np.array(next_aux_s),
            np.array(dn, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================================
# LỚP HUÂN LUYỆN LOCAL SỬ DỤNG MẠNG HAI NHÁNH
# =========================================================================
class LocalTrainer:

    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_actions = 6

        self.map_shape = (7, 13, 13)
        self.aux_dim = 3

        self.policy_net = DQNModel(
            self.map_shape, self.aux_dim, self.num_actions
        ).to(self.device)
        self.target_net = DQNModel(
            self.map_shape, self.aux_dim, self.num_actions
        ).to(self.device)

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
        map_grid = np.zeros((7, 13, 13), dtype=np.float32)
        game_map = obs["map"]
        for i in range(5):
            map_grid[i] = (game_map == i).astype(np.float32)

        for p_id, player in enumerate(obs["players"]):
            if player[2] == 1:
                map_grid[5 if p_id == self.agent_id else 6][player[0]][
                    player[1]
                ] = 1.0

        my_player = obs["players"][self.agent_id]
        bombs_left = float(my_player[3])
        radius_bonus = float(my_player[4])
        normalized_step = float(current_step) / 500.0

        aux_vector = np.array(
            [bombs_left, radius_bonus, normalized_step], dtype=np.float32
        )

        return map_grid, aux_vector

    def choose_action(self, map_grid, aux_vector):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        tensor_map = torch.FloatTensor(map_grid).unsqueeze(0).to(self.device)
        tensor_aux = torch.FloatTensor(aux_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(tensor_map, tensor_aux)

        return torch.argmax(q_values, dim=1).item()

    def shape_reward(self, obs_old, action, obs_new):
        reward = 0.0

        my_old = obs_old["players"][self.agent_id]
        my_new = obs_new["players"][self.agent_id]
        
        my_old_pos = (my_old[0], my_old[1])
        my_new_pos = (my_new[0], my_new[1])

        # ĐỊNH NGHĨA VỊ TRÍ XUẤT PHÁT CỐ ĐỊNH THEO ID (Cập nhật từ thông tin của bạn)
        spawn_positions = {
            0: (1, 1),    # Trên - Trái
            1: (11, 11),  # Dưới - Phải
            2: (1, 11),   # Trên - Phải
            3: (11, 1)    # Dưới - Trái
        }
        my_spawn = spawn_positions[self.agent_id]

        # 1. Trạng thái cơ bản
        if my_old[2] == 1 and my_new[2] == 0:
            return -1.0  # Chết
        reward += 0.005  # Thưởng sống sót

        # 2. Phạt đứng im hoặc đâm đầu vào tường
        if my_old_pos == my_new_pos:
            if action != 5:
                if action in [1, 2, 3, 4]:
                    reward -= 0.02  # Đâm tường
                else:
                    reward -= 0.01  # Đứng im

        # 3. PHẠT CHIẾN THUẬT: Né góc chết (Tránh lùi sâu vào góc xuất phát vô nghĩa)
        # Tính khoảng cách Manhattan tới góc xuất phát
        old_dist_to_spawn = abs(my_old_pos[0] - my_spawn[0]) + abs(my_old_pos[1] - my_spawn[1])
        new_dist_to_spawn = abs(my_new_pos[0] - my_spawn[0]) + abs(my_new_pos[1] - my_spawn[1])
        
        # Nếu bot đang ở gần góc (khoảng cách <= 2) mà lại chủ động đi lùi sát vào góc hơn nữa
        if new_dist_to_spawn < old_dist_to_spawn and new_dist_to_spawn <= 1:
            reward -= 0.03 # Phạt lùi vào góc chết tự bẫy

        # 4. Thưởng ăn item
        if my_new[4] > my_old[4]: reward += 0.1
        if my_new[3] > my_old[3]: reward += 0.1

        # 5. THƯỞNG CHIẾN THUẬT: Khuyến khích phá gạch mở đường
        old_boxes = np.sum(obs_old["map"] == 2)
        new_boxes = np.sum(obs_new["map"] == 2)
        destroyed_boxes = old_boxes - new_boxes
        
        if destroyed_boxes > 0:
            # Nếu bản đồ còn rất nhiều gạch (chứng tỏ đây là những ô gạch đầu trận)
            # Thưởng cực đậm để Bot học được cách "mở lối đi riêng" thoát khỏi vùng 2x2
            if old_boxes > 50: 
                reward += destroyed_boxes * 0.3  # Thưởng lớn cho ô gạch đầu trận
            else:
                reward += destroyed_boxes * 0.1  # Thưởng bình thường cho giai đoạn sau

        # 6. Đặt bom có mục tiêu (Giữ nguyên logic quét chữ thập)
        if action == 5 and my_old[3] > 0:
            has_target = False
            my_r, my_c = my_old_pos
            current_radius = int(my_old[4]) + 1
            
            for d_r, d_c in [(-1,0), (1,0), (0,-1), (0,1)]:
                for step_dist in range(1, current_radius + 1):
                    nr, nc = my_r + d_r * step_dist, my_c + d_c * step_dist
                    if 0 <= nr < 13 and 0 <= nc < 13:
                        if obs_old["map"][nr][nc] == 1: break
                        if obs_old["map"][nr][nc] == 2:
                            has_target = True
                            break
            if has_target:
                reward += 0.05
            else:
                reward -= 0.05  # Phạt đặt bom lãng phí ở góc trống an toàn

        # 7. Thưởng giết địch
        old_enemy_alive = sum(p[2] for i, p in enumerate(obs_old["players"]) if i != self.agent_id)
        new_enemy_alive = sum(p[2] for i, p in enumerate(obs_new["players"]) if i != self.agent_id)
        killed = old_enemy_alive - new_enemy_alive
        if killed > 0:
            reward += killed * 1.0

        return reward

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        map_s, aux_s, actions, rewards, next_map_s, next_aux_s, dones = (
            self.memory.sample(self.batch_size)
        )

        map_s_t = torch.FloatTensor(map_s).to(self.device)
        aux_s_t = torch.FloatTensor(aux_s).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_map_s_t = torch.FloatTensor(next_map_s).to(self.device)
        next_aux_s_t = torch.FloatTensor(next_aux_s).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = (
            self.policy_net(map_s_t, aux_s_t).gather(1, actions_t).squeeze(1)
        )

        with torch.no_grad():
            # THAY ĐỔI CHÍNH: Thêm `.detach()` để cắt đứt hoàn toàn luồng tính toán gradient ngầm của policy_net
            next_actions = (
                self.policy_net(next_map_s_t, next_aux_s_t)
                .argmax(1, keepdim=True)
                .detach()
            )

            next_q = (
                self.target_net(next_map_s_t, next_aux_s_t)
                .gather(1, next_actions)
                .squeeze(1)
            )
            target_q = rewards_t + (self.gamma * next_q * (1 - dones_t))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Dòng giữ nguyên, nhưng bây giờ số total_norm in ra sẽ "đẹp" và phản ánh đúng thực tế hơn
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0
        )
        # Thỉnh thoảng in hoặc bạn có thể bỏ print này nếu rối màn hình tqdm
        # print(f"Gradient Norm: {total_norm.item()}")

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
                map_grid, aux_vector = my_trainer.preprocess_obs(
                    obs, current_step=step
                )
                action_0 = my_trainer.choose_action(map_grid, aux_vector)

                action_1 = btc_bot_1.act(obs)
                action_2 = btc_bot_2.act(obs)
                action_3 = btc_bot_3.act(obs)

                actions = [action_0, action_1, action_2, action_3]

                next_obs, terminated, truncated = env.step(actions)
                
                # Cập nhật trạng thái done THẬT của môi trường
                game_over = terminated or truncated

                # Tính reward step bình thường (không tính phần thưởng thắng/thua ở đây nữa)
                reward = my_trainer.shape_reward(obs, action_0, next_obs)

                next_map_grid, next_aux_vector = my_trainer.preprocess_obs(
                    next_obs, current_step=step + 1
                )

                # KIỂM TRA NẾU TRẬN ĐẤU THỰC SỰ KẾT THÚC (Do có người thắng hoặc hết step)
                if game_over or (step + 1 >= max_steps):
                    alive_final = [bool(p[2]) for p in next_obs["players"]]
                    survivors = [i for i in range(4) if alive_final[i]]

                    # Thắng trận tuyệt đối (Chỉ còn mình mình sống)
                    if len(survivors) == 1 and survivors[0] == 0:
                        reward += 5.0
                    # Hòa trận do hết giờ (Có nhiều hơn 1 người sống bao gồm mình)
                    elif len(survivors) > 1 and 0 in survivors:
                        reward -= 2.0  # Phạt hòa trận khi hết giờ
                    # Thua trận (Mình đã chết)
                    else:
                        reward -= 2.0

                episode_reward += reward

                # Đẩy vào memory với trạng thái done chính xác
                my_trainer.memory.push(
                    map_grid,
                    aux_vector,
                    action_0,
                    reward,
                    next_map_grid,
                    next_aux_vector,
                    game_over,  # Đảm bảo lưu đúng trạng thái kết thúc
                )
                
                obs = next_obs
                step += 1

                # Đồng thời cập nhật biến done để thoát while nếu game_over=True
                done = game_over

                # Huấn luyện mạng Neural qua từng step
                my_trainer.train_step()

            # --- SAU KHI THOÁT WHILE: Ghi nhận kết quả chính xác để làm báo cáo ---
            alive_final = [bool(p[2]) for p in obs["players"]]
            survivors = [i for i in range(4) if alive_final[i]]
            if len(survivors) == 1 and survivors[0] == 0:
                is_win = "Y"
            elif len(survivors) > 1 and 0 in survivors:
                is_win = "D"  # Đánh dấu chính xác trận Hòa (Draw)
            else:
                is_win = "N"

            win_history.append(is_win)
            reward_history.append(episode_reward)
            survival_history.append(step)

            # Cập nhật Target Network định kỳ mỗi 10 episodes
            if episode % 10 == 0:
                my_trainer.target_net.load_state_dict(
                    my_trainer.policy_net.state_dict()
                )

            # Lưu model tốt nhất dựa trên tỉ lệ thắng
            if len(win_history) >= 100:
                current_winrate = win_history.count("Y") / len(win_history)
                if current_winrate > best_winrate:
                    best_winrate = current_winrate
                    my_trainer.save_model("best_model.pth")

            # In báo cáo định kỳ mỗi 100 episodes
            if episode % 100 == 0:
                avg_win_rate = (win_history.count("Y") / len(win_history)) * 100
                avg_draw_rate = (win_history.count("D") / len(win_history)) * 100
                avg_reward = np.mean(reward_history)
                avg_survival = np.mean(survival_history)

                print(f"\n================ BÁO CÁO PHONG ĐỘ (TRẬN {episode}) ================")
                print(f"* Tỷ lệ khám phá (Epsilon)   : {my_trainer.epsilon:.4f}")
                print(f"* Kích thước bộ nhớ (Memory) : {len(my_trainer.memory)}/50000")
                print(f"* TỶ LỆ THẮNG - HÒA (100 trận): Tốt: {avg_win_rate:.2f}% | Hòa: {avg_draw_rate:.2f}%")
                print(f"* SỐ STEP SỐNG SÓT TRUNG BÌNH: {avg_survival:.1f}/{max_steps}")
                print(f"* ĐIỂM THƯỞNG TRUNG BÌNH     : {avg_reward:.2f}")
                print("================================================================\n")
                my_trainer.save_model("last_model.pth")

            pbar.update(1)
            pbar.set_postfix(
                st=f"{step}", re=f"{episode_reward:.2f}", win=f"{is_win}"
            )