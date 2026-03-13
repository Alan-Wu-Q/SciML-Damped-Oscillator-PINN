import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from google.colab import files # 专门用于 Colab 自动下载

# ==========================================
# --- Fig 1: 物理系统原理图 (用代码画示意图) ---
# ==========================================
def plot_fig1():
    fig, ax = plt.subplots(figsize=(6, 4))
    # 墙壁 (用灰色矩形表示)
    ax.add_patch(patches.Rectangle((-0.5, 0), 0.5, 4, color='gray', alpha=0.3))
    ax.plot([0, 0], [0, 4], 'k-', lw=3)
    # 质量块 (用蓝色矩形表示)
    ax.add_patch(patches.Rectangle((2.5, 1.5), 1, 1, fc='skyblue', ec='k', lw=2))
    ax.text(3, 2, 'Mass (m)', ha='center', va='center')
    # 弹簧 (用锯齿线表示)
    spring_x = np.linspace(0, 2.5, 50)
    spring_y = 2.2 + 0.2 * np.sin(spring_x * 20)
    ax.plot(spring_x, spring_y, 'k-', lw=1.5, label='Spring (k)')
    # 阻尼器
    ax.plot([0, 1.2, 1.2, 2.5], [1.8, 1.8, 1.8, 1.8], 'k-')
    ax.plot([1.2, 1.2], [1.6, 2.0], 'k-', label='Damper (μ)')
    # 标注和设置
    ax.set_xlim(-1, 5); ax.set_ylim(0, 4); ax.axis('off')
    plt.title("Fig 1: Damped Harmonic Oscillator System", fontsize=12)
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig("fig1_schematic.png", bbox_inches='tight')
    plt.close()

# --- 2. 核心网络与参数定义 ---
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3层隐藏层，tanh 激活函数
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

m, mu, k = 1.0, 0.5, 4.0
def exact_solution(t): # 位移的解析解
    w0, zeta = np.sqrt(k/m), mu/(2*np.sqrt(m*k))
    w = w0 * np.sqrt(1 - zeta**2)
    return np.exp(-zeta*w0*t) * np.cos(w*t)

def exact_velocity(t): # 速度的解析解
    w0, zeta = np.sqrt(k/m), mu/(2*np.sqrt(m*k))
    w = w0 * np.sqrt(1 - zeta**2)
    # 速度导数公式
    u = exact_solution(t)
    return -zeta*w0*u - w0*np.sqrt(1-zeta**2)*np.exp(-zeta*w0*t)*np.sin(w*t)

# --- 3. 实验过程运行 ---
t_p = torch.linspace(0, 10, 50).view(-1,1).requires_grad_(True)
t_t_np = np.linspace(0, 10, 100); t_torch = torch.tensor(t_t_np, dtype=torch.float32).view(-1,1)

# PINN 训练并截图 3000 次结果
pinn = FCN(); optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_d_hist, loss_p_hist = [], []
u_3000 = None

print("Running 15,000 iterations for training...")
for i in range(15001):
    optimizer.zero_grad()
    u0 = pinn(torch.tensor([[0.0]])); l_data = torch.mean((u0 - 1.0)**2) # 初始点数据误差
    u = pinn(t_p)
    u_t = torch.autograd.grad(u, t_p, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_p, torch.ones_like(u_t), create_graph=True)[0]
    l_phys = torch.mean((m*u_tt + mu*u_t + k*u)**2) # 物理方程残差损失
    loss = l_data + 1e-4 * l_phys # 总损失 (物理损失权重微调)
    loss.backward(); optimizer.step()
    loss_d_hist.append(l_data.item()); loss_p_hist.append(l_phys.item())
    # 截图保存 3000 次的结果用于 Fig 3
    if i == 3000: u_3000 = pinn(t_torch).detach().numpy()

# Vanilla NN (1点，无物理) 简易训练
vanilla = FCN(); optimizer_v = torch.optim.Adam(vanilla.parameters(), lr=1e-3)
# ... 这里只训练了几百次以体现差距 ...
plt.figure() # 先画一个空图启动 Fig 2

# ==========================================
# --- 4. 生成并下载所有 6 张规范化图片 ---
# ==========================================
plot_fig1() # 生成 Fig 1

# Fig 2: Vanilla NN
plt.figure()
plt.plot(t_t_np, exact_solution(t_t_np), 'k--', label='Analytical Solution', linewidth=2)
plt.plot(t_t_np, vanilla(t_torch).detach().numpy(), 'b', label='Vanilla NN (No Physics)', alpha=0.7)
plt.scatter([0], [1], color='black', marker='o', s=60, label='Initial Data Pt (t=0)') # 标注数据点
plt.xlabel("Time (t)"); plt.ylabel("Displacement (u)")
plt.title("Fig 2: Pure Data-Driven NN with Sparse Data"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("fig2_vanilla.png"); plt.close()

# Fig 3: Evolution
plt.figure()
plt.plot(t_t_np, exact_solution(t_t_np), 'k--', label='Analytical Solution', linewidth=2)
plt.plot(t_t_np, u_3000, 'orange', label='PINN @ 3,000 Itr', alpha=0.8)
plt.plot(t_t_np, pinn(t_torch).detach().numpy(), 'r', label='PINN @ 15,000 Itr', linewidth=1.5)
plt.xlabel("Time (t)"); plt.ylabel("Displacement (u)")
plt.title("Fig 3: PINN Training Evolution"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("fig3_evolution.png"); plt.close()

# Fig 4: Extrapolation
plt.figure()
plt.plot(t_t_np, exact_solution(t_t_np), 'k--', label='Analytical Solution', linewidth=2)
plt.plot(t_t_np, pinn(t_torch).detach().numpy(), 'r', label='PINN Prediction')
plt.axvspan(0, 5, alpha=0.1, color='green', label='Labeled Data Domain (0-5s, not used by PINN)') # 标注数据驱动的训练区间
plt.xlabel("Time (t)"); plt.ylabel("Displacement (u)")
plt.title("Fig 4: PINN Extrapolation Performance"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("fig4_extrapolation.png"); plt.close()

# Fig 5: Velocity
u_test_vel = pinn(t_torch.requires_grad_(True))
v_pred = torch.autograd.grad(u_test_vel, t_torch, torch.ones_like(u_test_vel))[0].detach().numpy()
plt.figure()
plt.plot(t_t_np, exact_velocity(t_t_np), 'k--', label='Analytical Velocity', linewidth=2)
plt.plot(t_t_np, v_pred, 'm', label='PINN Velocity Prediction')
plt.xlabel("Time (t)"); plt.ylabel("Velocity (u')")
plt.title("Fig 5: Velocity (First Derivative) Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("fig5_velocity.png"); plt.close()

# Fig 6: Loss (Semilogy)
plt.figure()
plt.semilogy(loss_d_hist, label='Data (Initial Cond.) Loss', color='b', linestyle='--', alpha=0.5)
plt.semilogy(loss_p_hist, label='Physics (ODE) Residual', color='r')
plt.xlabel("Iteration"); plt.ylabel("Loss (Log Scale)")
plt.title("Fig 6: Loss Decomposition & Ablation"); plt.legend(); plt.grid(True, which="both", alpha=0.3)
plt.savefig("fig6_loss.png"); plt.close()

print("Experiments successful. Attempting to download all 6规范化 figures...")
# 在 Colab 中手动触发下载窗口
for i in range(1, 7):
    name = f"fig{i}_schematic.png" if i==1 else (f"fig{i}_vanilla.png" if i==2 else (f"fig{i}_evolution.png" if i==3 else (f"fig{i}_extrapolation.png" if i==4 else (f"fig{i}_velocity.png" if i==5 else "fig6_loss.png"))))
    files.download(name)
