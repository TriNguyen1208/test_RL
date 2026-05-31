import torch
import torch.nn as nn

# ịnput_channels = 8:
#     + 0: cỏ, 1: tường, 2: hộp, 3: vật phẩm tăng bán kính, 4: vật phẩm tăng số lượng
#     + 5: vị trí của agent
#     + 6: vị trí của đối thủ
#     + 7: Danh sách chứa các bom hiện có (timer đếm ngược đến 7)

#DQN nghĩa là dựa vào mạng neuron, dựa vào 8 giá trị đầu vào và trả về được
class DQN(nn.Module):
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
