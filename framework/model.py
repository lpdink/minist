from torch import nn
# 一个模型类必须继承nn.Module，并实现forward方法。
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # nn.Flatten的start_dim从1开始，dim0保留给了batch_size
        self.flatten = nn.Flatten()
        self.liner_relu_stack = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.liner_relu_stack(x)
        return x