import numpy as np
import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, main_dim, num_heads, k_dim=None, v_dim=None):
        super().__init__()
        self.main_dim = main_dim
        self.num_heads = num_heads
        self.k_dim = k_dim if k_dim != None else main_dim
        self.v_dim = v_dim if v_dim != None else main_dim
        self.query = nn.Linear(self.main_dim, self.k_dim)
        self.key = nn.Linear(self.main_dim, self.k_dim)
        self.value = nn.Linear(self.main_dim, self.v_dim)
        self.fc = nn.Linear(self.v_dim, self.main_dim)

    def forward(self, input1, input2, mask=None):
        batch_size1, num_tokens1, dim1 = input1.size()
        batch_size2, num_tokens2, dim2 = input2.size()
        query = self.query(input2)
        key = self.key(input1)
        value = self.value(input1)
        query = query.view(batch_size2, num_tokens2, self.num_heads, self.k_dim//self.num_heads).permute(0,2,1,3)
        key = key.view(batch_size1, num_tokens1, self.num_heads, self.k_dim//self.num_heads).permute(0,2,1,3)
        value = value.view(batch_size1, num_tokens1, self.num_heads, self.v_dim//self.num_heads).permute(0,2,1,3)
        x = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / math.sqrt(self.k_dim//self.num_heads)
        if mask != None:
            x = x + mask[:num_tokens2]
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, value)
        x = x.permute(0,2,1,3).reshape(batch_size2, num_tokens2, -1)
        return self.fc(x)


class EncoderLayer(nn.Module):
    def __init__(self, main_dim, ff_dim, num_heads, dropout_p):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(main_dim, num_heads)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(main_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(ff_dim, main_dim)
        )
        self.norm = nn.LayerNorm(main_dim)

    def forward(self, input, mask):
        x = self.norm(input + self.multi_head_attention(input, input, mask))
        x = self.norm(x + self.positionwise_feedforward(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, main_dim=256, ff_dim=512, patch_size=16, num_image_channels=3, num_heads=8, num_layers=5, num_classes=10, dropout_p=0.2, max_patches=4, device='cuda') -> None:
        super().__init__()
        self.main_dim = main_dim
        self.ff_dim = ff_dim
        self.patch_size = patch_size
        self.num_image_channels = num_image_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.max_patches = max_patches
        self.device = device
        self.generate_positional_encodings()
        self.patch_linear = nn.Linear(num_image_channels*patch_size**2, main_dim)
        self.cls_embedding = nn.Parameter(torch.randn(main_dim)).to(device)
        self.encoder = nn.Sequential(
            *[EncoderLayer(main_dim, ff_dim, num_heads, dropout_p) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(main_dim, num_classes)

    def generate_positional_encodings(self):
        self.positional_encoding = torch.from_numpy(np.zeros(shape=(1+self.max_patches, self.main_dim))).type(torch.float32).to(self.device)
        for i in range(self.max_patches):
            for j in range(self.main_dim):
                if j%2 == 0:
                    self.positional_encoding[i,j] = math.sin(i/100**(2*(j//2)/self.patch_size**2))
                else:
                    self.positional_encoding[i,j] = math.cos(i/100**(2*(j//2)/self.patch_size**2))

    def forward(self, input):
        batch_size, _, _, _ = input.size()
        patched = input.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patched = patched.permute(0,2,3,1,4,5).reshape(batch_size, -1, self.num_image_channels*self.patch_size**2).type(torch.float)
        patched = self.patch_linear(patched)
        x = torch.concatenate([self.cls_embedding.expand(batch_size, 1, -1), patched], 1)
        x = x + self.positional_encoding[:x.size(1)]
        for module in self.encoder._modules.values():
            x = module(x, mask=None)
        return self.fc(x[:,0])
