import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class RepresentationNetwork(nn.Module):
    def __init__(self, board_size: int = 5, num_channels: int = 3, latent_channels: int = 32, num_res_blocks: int = 2):
        super().__init__()
        self.board_size = board_size
        self.conv_init = nn.Conv2d(num_channels, latent_channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(latent_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.ModuleList([ResidualBlock(latent_channels) for _ in range(num_res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_init(x)
        out = self.bn_init(out)
        out = self.relu(out)
        for block in self.res_blocks:
            out = block(out)
        # Bounded normalization to [0, 1] for latent stability
        return torch.sigmoid(out)

class DynamicsNetwork(nn.Module):
    def __init__(self, board_size: int = 5, latent_channels: int = 32, action_space_size: int = 25, num_res_blocks: int = 2):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size or (board_size * board_size)

        # Main transition dynamics
        self.conv_init = nn.Conv2d(latent_channels + 1, latent_channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(latent_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.ModuleList([ResidualBlock(latent_channels) for _ in range(num_res_blocks)])

        # Reward head
        self.reward_conv = nn.Conv2d(latent_channels, 1, kernel_size=1, bias=False)
        self.reward_bn = nn.BatchNorm2d(1)
        self.reward_fc1 = nn.Linear(board_size * board_size, 16)
        self.reward_fc2 = nn.Linear(16, 1)

    def encode_action_plane(self, action: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        action_plane = torch.zeros((batch_size, 1, self.board_size, self.board_size), device=device, dtype=torch.float32)
        for b in range(batch_size):
            act = action[b].item()
            if 0 <= act < self.board_size * self.board_size:
                r = act // self.board_size
                c = act % self.board_size
                action_plane[b, 0, r, c] = 1.0
        return action_plane

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor):
        batch_size = latent_state.shape[0]
        device = latent_state.device

        # One-hot action spatial plane
        action_plane = self.encode_action_plane(action, batch_size, device)

        # Concatenate action plane with latent state along channel dimension
        x = torch.cat([latent_state, action_plane], dim=1)

        # Process through convolutional body
        out = self.conv_init(x)
        out = self.bn_init(out)
        out = self.relu(out)
        for block in self.res_blocks:
            out = block(out)
        next_latent_state = torch.sigmoid(out)

        # Compute predicted scalar reward
        rw = self.reward_conv(out)
        rw = self.reward_bn(rw)
        rw = self.relu(rw)
        rw = torch.flatten(rw, start_dim=1)
        rw = self.relu(self.reward_fc1(rw))
        reward = self.reward_fc2(rw)

        return next_latent_state, reward

class PredictionNetwork(nn.Module):
    def __init__(self, board_size: int = 5, latent_channels: int = 32, action_space_size: int = 25, use_fcn: bool = False):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size or (board_size * board_size)
        self.use_fcn = use_fcn

        if self.use_fcn:
            self.policy_conv = nn.Conv2d(latent_channels, 1, kernel_size=1)
            self.value_conv = nn.Conv2d(latent_channels, 1, kernel_size=1)
            self.value_relu = nn.ReLU(inplace=True)
            self.value_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            # Policy Head
            self.policy_conv = nn.Conv2d(latent_channels, 2, kernel_size=1, bias=False)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_relu = nn.ReLU(inplace=True)
            self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_space_size)
    
            # Value Head
            self.value_conv = nn.Conv2d(latent_channels, 1, kernel_size=1, bias=False)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_relu = nn.ReLU(inplace=True)
            self.value_fc1 = nn.Linear(board_size * board_size, 32)
            self.value_fc2 = nn.Linear(32, 1)

    def forward(self, latent_state: torch.Tensor):
        if self.use_fcn:
            # Policy logits
            p = self.policy_conv(latent_state)
            policy_logits = torch.flatten(p, start_dim=1)
            
            # Value [-1, 1]
            v = self.value_conv(latent_state)
            v = self.value_relu(v)
            v = self.value_pool(v)
            v = torch.flatten(v, start_dim=1)
            value = torch.tanh(v)
        else:
            # Policy logits
            p = self.policy_conv(latent_state)
            p = self.policy_bn(p)
            p = self.policy_relu(p)
            p = torch.flatten(p, start_dim=1)
            policy_logits = self.policy_fc(p)
    
            # Value [-1, 1]
            v = self.value_conv(latent_state)
            v = self.value_bn(v)
            v = self.value_relu(v)
            v = torch.flatten(v, start_dim=1)
            v = self.policy_relu(self.value_fc1(v))
            value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


class MuZeroModels(nn.Module):
    def __init__(self, board_size: int = 5, action_space_size: int = 25, latent_channels: int = 32, num_res_blocks: int = 2, input_channels: int = 3, use_fcn: bool = False):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size or (board_size * board_size)
        self.use_fcn = use_fcn

        self.representation = RepresentationNetwork(
            board_size=board_size,
            num_channels=input_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks
        )
        self.dynamics = DynamicsNetwork(
            board_size=board_size,
            latent_channels=latent_channels,
            action_space_size=self.action_space_size,
            num_res_blocks=num_res_blocks
        )
        self.prediction = PredictionNetwork(
            board_size=board_size,
            latent_channels=latent_channels,
            action_space_size=self.action_space_size,
            use_fcn=use_fcn
        )

    def initial_inference(self, observation: torch.Tensor):
        latent_state = self.representation(observation)
        policy_logits, value = self.prediction(latent_state)
        reward = torch.zeros((observation.shape[0], 1), device=observation.device, dtype=observation.dtype)
        return value, reward, policy_logits, latent_state

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor):
        next_latent_state, reward = self.dynamics(latent_state, action)
        policy_logits, value = self.prediction(next_latent_state)
        return value, reward, policy_logits, next_latent_state
