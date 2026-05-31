from DQN import DQN
from btc.engine.game import BomberEnv
from btc.agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity=50000):
        #Khởi tạo bộ nhớ 2 đầu với 50000 phần tử
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        #Đẩy dữ liệu MDP vào bộ nhớ thành 1 tuple và lưu vào buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Lấy ngẫu nhiên một lượng 'batch_size' các bộ trải nghiệm từ bộ nhớ.
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # Chuyển đổi các danh sách này thành mảng NumPy với kiểu dữ liệu
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
    
class LocalTrainer:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id # Định danh ID của Agent đang được huấn luyện
        # Tự động chọn GPU (cuda) để tăng tốc tính toán nếu có, ngược lại dùng CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = 6 # Game có 6 hành động từ 0 đến 5
        
        # Cấu hình Double DQN
        # policy_net: Mạng nơ-ron chính, liên tục cập nhật trọng số
        self.policy_net = DQN(8, self.num_actions).to(self.device)
        # target_net: Mạng mục tiêu, giữ cố định trọng số trong một khoảng thời gian để làm mốc so sánh ổn định
        self.target_net = DQN(8, self.num_actions).to(self.device)
        # Đồng bộ hóa ban đầu: Sao chép toàn bộ trọng số từ mạng chính sang mạng mục tiêu
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Đặt target_net ở chế độ đánh giá (không tính Gradient)
        self.target_net.eval()
        # Sử dụng bộ tối ưu hóa Adam để cập nhật trọng số mạng với tốc độ học (learning rate) là 0.0001
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        #Tạo giả lập memory
        self.memory = ReplayBuffer()
        
        # Siêu tham số RL
        self.gamma = 0.95 # Hệ số chiết khấu (Discount Factor), quy định tầm nhìn xa của Agent
        self.epsilon = 1.0 # Tỷ lệ khám phá ban đầu (100% chọn hành động ngẫu nhiên để thử nghiệm)
        self.epsilon_min = 0.05 # Ngưỡng epsilon tối thiểu (Agent luôn giữ 5% ngẫu nhiên để không bị bảo thủ)
        self.epsilon_decay = 0.995 # Tốc độ giảm epsilon chậm để học kỹ hơn
        self.batch_size = 64 # Kích thước mỗi cụm dữ liệu bốc ra từ bộ nhớ để học

    def preprocess_obs(self, obs: dict) -> np.ndarray:
        """Chuyển đổi obs thành grid 8 channels cho vào mạng RL"""

        state_grid = np.zeros((8, 13, 13), dtype=np.float32)
        # Lớp 0 đến 4: Duyệt qua các giá trị map tĩnh từ BTC (Cỏ, Tường, Hộp, Item 3, Item 4)
        # Ô nào khớp với chỉ số i sẽ được gán giá trị 1.0, còn lại là 0.0
        for i in range(5):
            state_grid[i] = (obs['map'] == i).astype(np.float32) #Trả về shape: 8 x 13 x 13: giá trị 1.0 hoặc 0.0
        # Lớp 5 & 6: Duyệt qua mảng thông tin của 4 người chơi
        for p_id, player in enumerate(obs['players']):
            if player[2] == 1: # alive == 1
                # Nếu p_id trùng với ID của mình -> Ghi vào Lớp 5. Ngược lại (đối thủ) -> Ghi vào Lớp 6
                state_grid[5 if p_id == self.agent_id else 6][player[0]][player[1]] = 1.0
        # Lớp 7: Xử lý hiển thị bom nguy hiểm
        if obs['bombs'].size > 0:
            for bomb in obs['bombs']:
                # Lấy tọa độ [row, col] và chuẩn hóa thời gian nổ (timer) về khoảng cách số từ 0.14 đến 1.0
                state_grid[7][int(bomb[0])][int(bomb[1])] = (8.0 - float(bomb[2])) / 7.0
        return state_grid

    def choose_action(self, state_grid):
        # Tung xúc xắc ngẫu nhiên. Nếu nhỏ hơn eps -> Cho Agent chọn đại một nút để khám phá thế giới
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Nếu lớn hơn eps -> Bắt đầu tính toán bằng trí tuệ của mạng Neural
        # Chuyển mảng NumPy thành Tensor PyTorch và thêm chiều thành (1, 8, 13, 13)
        tensor_state = torch.FloatTensor(state_grid).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(tensor_state) # Truyền trạng thái vào mạng để nhận về 6 Q-values (inferences)
        return torch.argmax(q_values, dim=1).item() # Trả về giá trị tốt nhất khi chọn action đó từ 0 - 5

    def shape_reward(self, obs_old, action, obs_new):
        """Hàm thưởng phạt tự chế (Reward Shaping) để định hướng tư duy"""
        reward = 0.0
        my_old = obs_old['players'][self.agent_id] # Lấy thông tin của mình ở step cũ
        my_new = obs_new['players'][self.agent_id] # Lấy thông tin của mình ở step mới
        
        if my_new[2] == 0: return -20.0 # Chết phạt cực nặng
        reward += 0.01 # Thưởng sống dai
        
        if my_new[4] > my_old[4]: reward += 2.0 # Ăn Item Radius
        if my_new[3] > my_old[3]: reward += 2.0 # Ăn Item Capacity
        if action == 5 and my_old[3] > 0: reward += 0.2 # Đặt bom hợp lệ
        return reward

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Bốc ngẫu nhiên một batch 64 trải nghiệm từ bộ nhớ ra để xử lý
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Chuyển đổi đồng loạt các mảng dữ liệu này thành Tensors PyTorch và đẩy lên CPU/GPU thích hợp
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Tính toán Q(s, a) hiện tại: Cho 'states_t' đi qua mạng chính, 
        # dùng hàm .gather() để chỉ lọc ra giá trị Q của đúng hành động 'actions_t' đã chọn thực tế
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        
        with torch.no_grad():
            # Cho trạng thái kế tiếp đi qua mạng mục tiêu (target_net) và lấy ra giá trị Q lớn nhất có thể đạt được
            next_max_q = self.target_net(next_states_t).max(1)[0]
            # Nếu game kết thúc (done = 1), giá trị tương lai bằng 0, mục tiêu chỉ bằng đúng phần thưởng hiện tại
            target_q = rewards_t + (self.gamma * next_max_q * (1 - dones_t))

        loss = nn.MSELoss()(current_q, target_q)
        # Tiến hành tối ưu hóa trọng số mạng Neural bằng lan truyền ngược (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Giảm dần chỉ số epsilon theo hệ số decay để Agent bớt đi lung tung
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="model.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print(f"--> Đã lưu trọng số mới nhất vào file: {filename}")


# =========================================================================
# KHỞI CHẠY VÒNG LẶP HUẤN LUYỆN KẾT NỐI ENGINE THẬT CỦA BTC
# =========================================================================
if __name__ == "__main__":
    num_episodes = 2000
    max_steps = 500
    seed = 42 # Cố định seed ban đầu để đảm bảo tính tái lập kết quả
    
    # 1. Khởi tạo môi trường thật từ mã nguồn gốc của BTC
    env = BomberEnv(max_steps=max_steps, seed=seed)
    
    # 2. Khởi tạo bộ não cần train (Agent của bạn giữ góc ID 0)
    my_trainer = LocalTrainer(agent_id=0)
    
    # 3. Khởi tạo 3 Bot Baseline "cựu binh" của BTC làm quân xanh tập trận
    btc_bot_1 = TacticalRuleAgent(1) # Bot chiến thuật cực gắt
    btc_bot_2 = BoxFarmerAgent(2)    # Bot chuyên cày cuốc phá hộp
    btc_bot_3 = SimpleRuleAgent(3)   # Bot logic cơ bản tạo nhiễu
    
    win_history = deque(maxlen=100)       # Lưu kết quả 100 trận gần nhất (1 = Thắng, 0 = Hòa/Thua)
    reward_history = deque(maxlen=100)    # Lưu tổng điểm thưởng tự chế nhận được trong 100 trận gần nhất
    survival_history = deque(maxlen=100)  # Lưu số step sống sót trước khi chết trong 100 trận gần nhất
    print("=== Khởi động hệ thống huấn luyện thực tế với Bot BTC ===")
    
    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for episode in range(num_episodes):
            # Tính toán seed biến đổi theo từng trận giống hệt cách BTC làm
            episode_seed = None if seed is None else seed + episode
            obs = env.reset(seed=episode_seed)
            done = False
            step = 0
            episode_reward = 0.0

            while not done and step < max_steps:
                # --- BƯỚC 1: Thu thập hành động từ cả 4 Agent ---
                
                # Agent của bạn (ID 0) xử lý qua lưới 8 channels và mạng Neural
                state_grid = my_trainer.preprocess_obs(obs)
                action_0 = my_trainer.choose_action(state_grid)
                
                # 3 Bot của BTC tự tính toán dựa trên thuật toán heuristic cứng của họ
                action_1 = btc_bot_1.act(obs)
                action_2 = btc_bot_2.act(obs)
                action_3 = btc_bot_3.act(obs)
                
                actions = [action_0, action_1, action_2, action_3]
                
                # --- BƯỚC 2: Đẩy mảng hành động vào Engine thật ---
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated
                step += 1
                
                # --- BƯỚC 3: Lưu trải nghiệm nếu Agent của bạn còn sống ở bước này ---
                if obs["players"][0][2] == 1:
                    reward = my_trainer.shape_reward(obs, action_0, next_obs)
                    episode_reward += reward
                    next_state_grid = my_trainer.preprocess_obs(next_obs)
                    
                    # Lưu trữ vào nhật ký Replay Buffer để học
                    my_trainer.memory.push(state_grid, action_0, reward, next_state_grid, done)
                
                obs = next_obs
                
                # --- BƯỚC 4: Thực thi Lan truyền ngược cập nhật trọng số ---
                my_trainer.train_step()
                
            # --- BƯỚC 5: Ghi nhận thông tin sau mỗi trận đấu ---
            alive_final = [bool(p[2]) for p in obs["players"]]
            survivors = [i for i in range(4) if alive_final[i]]
            
            # Ghi nhận kết quả Thắng/Thua theo luật BTC (Chỉ tính Win nếu là người DUY NHẤT còn sống)
            is_win = 1 if (len(survivors) == 1 and survivors[0] == 0) else 0
            
            # Đẩy dữ liệu trận này vào hàng đợi lịch sử 100 trận gần nhất
            win_history.append(is_win)
            reward_history.append(episode_reward)
            survival_history.append(step) # Nếu done trước 500 bước tức là đã chết ở step này, nếu sống hết ván sẽ là 500
                
            # Cứ sau 10 trận đấu: Tiến hành đồng bộ hóa trọng số từ mạng chính sang mạng mục tiêu (Target Net)
            if episode % 10 == 0:
                my_trainer.target_net.load_state_dict(my_trainer.policy_net.state_dict())
                
            # Cứ sau 100 trận đấu: In một bảng BÁO CÁO PHONG ĐỘ chi tiết ra màn hình để bạn đánh giá
            if episode % 100 == 0:
                # avg_win_rate = np.mean(win_history) * 100       # Tỷ lệ thắng trung bình (%)
                # avg_reward = np.mean(reward_history)            # Điểm thưởng trung bình nhận được
                # avg_survival = np.mean(survival_history)        # Số bước sống sót trung bình
                
                # print(f"\n================ BÁO CÁO PHONG ĐỘ (TRẬN {episode}) ================")
                # print(f"* Tỷ lệ khám phá (Epsilon)   : {my_trainer.epsilon:.4f}")
                # print(f"* Kích thước bộ nhớ (Memory) : {len(my_trainer.memory)}/50000")
                # print(f"* TỶ LỆ THẮNG (100 trận gần) : {avg_win_rate:.2f}%")
                # print(f"* SỐ STEP SỐNG SÓT TRUNG BÌNH: {avg_survival:.1f}/{max_steps}")
                # print(f"* ĐIỂM THƯỞNG TRUNG BÌNH     : {avg_reward:.2f}")
                # print("================================================================\n")
                
                # # Xuất file trọng số an toàn xuống ổ đĩa, sẵn sàng để lấy file này đem đi nộp bài
                my_trainer.save_model("model.pth")
            pbar.update(1)
            # pbar.set_postfix(step=f"{step:.2f}", episode_reward=f"{episode_reward:.3f}", win=f"{"YES" if is_win else "NO"}", memory=f"{len(my_trainer.memory)}/50000", epsilon=f"{my_trainer.epsilon:.3f}")
            pbar.set_postfix(st=f"{step}", re=f"{episode_reward:.3f}", win=f"{"Y" if is_win else "N"}")