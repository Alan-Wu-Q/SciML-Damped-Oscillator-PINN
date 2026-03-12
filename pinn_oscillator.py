import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义神经网络结构
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation()]) for _ in range(N_LAYERS-1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# 2. 物理参数定义 (阻尼振子: m*u'' + mu*u' + k*u = 0)
m, mu, k = 1.0, 0.5, 4.0

def exact_solution(t): # 这里的解析解用于对比基准
    # 这是一个简化版的解析解，用于验证
    w0 = np.sqrt(k/m)
    zeta = mu / (2*np.sqrt(m*k))
    w = w0 * np.sqrt(1 - zeta**2)
    return np.exp(-zeta*w0*t) * np.cos(w*t)

# 3. 训练 PINN
def train():
    t_physics = torch.linspace(0, 10, 30).view(-1,1).requires_grad_(True) # 物理约束点
    t_data = torch.tensor([0.0]).view(-1,1).requires_grad_(True) # 初始条件点
    u_data = torch.tensor([1.0]).view(-1,1) # t=0 时 u=1
    
    model = FCN(1, 1, 32, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for i in range(2000):
        optimizer.zero_grad()
        
        # Data Loss (初始条件)
        u_pred = model(t_data)
        loss_data = torch.mean((u_pred - u_data)**2)
        
        # Physics Loss
        u = model(t_physics)
        u_t = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_physics, torch.ones_like(u_t), create_graph=True)[0]
        loss_physics = torch.mean((m*u_tt + mu*u_t + k*u)**2)
        
        loss = loss_data + loss_physics # 核心公式
        loss.backward()
        optimizer.step()
        
        if i % 500 == 0: print(f"Iteration {i}, Loss: {loss.item()}")
    
    return model

# 4. 可视化结果
model = train()
t_test = torch.linspace(0, 10, 100).view(-1,1)
u_pred = model(t_test).detach().numpy()
u_exact = exact_solution(t_test.numpy())

plt.figure(figsize=(8,4))
plt.plot(t_test, u_exact, label="Exact Solution", color="black", linestyle="--")
plt.plot(t_test, u_pred, label="PINN Prediction", color="red")
plt.legend()
plt.title("Damped Harmonic Oscillator: PINN vs Exact")
plt.savefig("result_plot.png") # 保存这张图，后面要用
plt.show()
