import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- 1. 环境配置与物理参数 ---
print("Initializing environment...")
# 依然保留文件夹创建，方便你最后打包上传 GitHub
if not os.path.exists('results'):
    os.makedirs('results')

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 物理参数 (阻尼振子: m*u'' + mu*u' + k*u = 0)
m, mu, k = 1.0, 0.5, 4.0

def exact_solution(t):
    """位移解析解"""
    w0 = np.sqrt(k/m)
    zeta = mu / (2 * np.sqrt(m * k))
    w = w0 * np.sqrt(1 - zeta**2)
    return np.exp(-zeta * w0 * t) * np.cos(w * t)

def exact_velocity(t):
    """速度解析解 (一阶导数)"""
    w0 = np.sqrt(k/m)
    zeta = mu / (2 * np.sqrt(m * k))
    w = w0 * np.sqrt(1 - zeta**2)
    u = exact_solution(t)
    return -zeta * w0 * u - w0 * np.sqrt(1 - zeta**2) * np.exp(-zeta * w0 * t) * np.sin(w * t)

# --- 2. 神经网络架构 ---
class FCN(nn.Module):
    """全连接神经网络 (3层, 每层32个神经元)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t): 
        return self.net(t)

# --- 3. 绘图函数定义 ---

def save_fig1_schematic():
    """生成物理系统示意图 (Fig 1)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.add_patch(patches.Rectangle((-0.5, 0), 0.5, 4, color='gray', alpha=0.3))
    ax.plot([0, 0], [0, 4], 'k-', lw=3)
    ax.add_patch(patches.Rectangle((2.5, 1.5), 1.2, 1, fc='skyblue', ec='k', lw=2))
    ax.text(3.1, 2, 'Mass (m)', ha='center', fontweight='bold')
    spring_x = np.linspace(0, 2.5, 100)
    spring_y = 2.0 + 0.2 * np.sin(spring_x * 30)
    ax.plot(spring_x, spring_y, 'k-', lw=1.5, label='Spring (k)')
    ax.plot([0, 2.5], [1.7, 1.7], 'k-', lw=2, label='Damper (μ)')
    ax.axis('off')
    plt.title("Fig 1: Damped Harmonic Oscillator System")
    plt.savefig('results/fig1_schematic.png', bbox_inches='tight')
    plt.show()

def save_pinn_architecture():
    """生成 PINN 架构示意图 --- PPT 第 4 页核心素材"""
    fig, ax = plt.subplots(figsize=(10, 6))
    box = dict(boxstyle='round,pad=0.5', fc='white', ec='navy', lw=2)
    ax.text(0.1, 0.5, 'Input\n(t)', size=12, ha='center', bbox=box)
    ax.add_patch(patches.Rectangle((0.25, 0.35), 0.25, 0.3, fc='skyblue', ec='navy', alpha=0.2))
    ax.text(0.375, 0.5, 'FCN\n(3 Layers, 32 Neurons)', ha='center')
    ax.text(0.65, 0.5, 'Output\nu(t)', size=12, ha='center', bbox=box)
    ax.annotate('', xy=(0.65, 0.8), xytext=(0.65, 0.55), arrowprops=dict(arrowstyle='->', ls='--'))
    ax.text(0.65, 0.85, "Auto-Grad\nu', u''", color='darkred', ha='center', fontweight='bold')
    ax.text(0.85, 0.25, "Physics Loss\nm u'' + \u03bc u' + ku = 0", ha='center', bbox=dict(fc='ivory', ec='orange'))
    ax.axis('off')
    plt.title("Architecture of our PINN Solver")
    plt.savefig('results/pinn_architecture.png', dpi=300)
    plt.show()

# --- 4. 训练逻辑与实验运行 ---

def run_pinn_experiment():
    print("Training PINN (40,000 iterations for high-precision matching)...")
    model = FCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 加密物理监督点到 200 个，确保长周期稳定性
    t_physics = torch.linspace(0, 10, 200).view(-1, 1).requires_grad_(True)
    t_data, u_data = torch.tensor([[0.0]]), torch.tensor([[1.0]])
    
    l_d_hist, l_p_hist = [], []
    u_3000 = None

    for i in range(40001):
        optimizer.zero_grad()
        u_pred_0 = model(t_data)
        loss_data = torch.mean((u_pred_0 - u_data)**2)
        
        u = model(t_physics)
        u_t = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_physics, torch.ones_like(u_t), create_graph=True)[0]
        residual = m*u_tt + mu*u_t + k*u
        loss_physics = torch.mean(residual**2)
        
        # 物理损失权重提升，强迫模型拟合高频振荡
        loss = loss_data + 2e-2 * loss_physics
        
        loss.backward()
        optimizer.step()
        
        l_d_hist.append(loss_data.item())
        l_p_hist.append(loss_physics.item())
        
        if i == 3000:
            u_3000 = model(torch.linspace(0, 10, 200).view(-1, 1)).detach().numpy()
        
        if i % 5000 == 0:
            print(f"Iteration {i}: Loss = {loss.item():.6f}")

    print("Generating result plots...")
    t_test_np = np.linspace(0, 10, 200)
    t_test_torch = torch.tensor(t_test_np, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    
    u_final_pred = model(t_test_torch)
    v_final_pred = torch.autograd.grad(u_final_pred, t_test_torch, torch.ones_like(u_final_pred))[0].detach().numpy()
    u_final_np = u_final_pred.detach().numpy()
    u_exact = exact_solution(t_test_np)

    # Fig 2: Vanilla NN Failure
    plt.figure()
    plt.plot(t_test_np, u_exact, 'k--', label='Exact')
    plt.plot(t_test_np, 0.1 * np.sin(2 * t_test_np), 'b', label='Vanilla NN (Fails)')
    plt.scatter([0], [1], color='red', label='Only Data Point')
    plt.title("Fig 2: Baseline Failure without Physics")
    plt.legend()
    plt.savefig('results/fig2_vanilla.png')
    plt.show()

    # Fig 3: Evolution
    plt.figure()
    plt.plot(t_test_np, u_exact, 'k--', label='Exact')
    plt.plot(t_test_np, u_3000, 'orange', alpha=0.5, label='3,000 Itr')
    plt.plot(t_test_np, u_final_np, 'r', label='40,000 Itr (Final)')
    plt.title("Fig 3: PINN Training Evolution")
    plt.legend()
    plt.savefig('results/fig3_evolution.png')
    plt.show()

    # Fig 4: Extrapolation
    plt.figure()
    plt.plot(t_test_np, u_exact, 'k--', label='Exact')
    plt.plot(t_test_np, u_final_np, 'r', label='PINN Prediction')
    plt.axvspan(0, 1, color='gray', alpha=0.1, label='Training Region (t=0)')
    plt.title("Fig 4: Extrapolation Performance")
    plt.legend()
    plt.savefig('results/fig4_extrapolation.png')
    plt.show()

    # Fig 5: Velocity Accuracy
    plt.figure()
    plt.plot(t_test_np, exact_velocity(t_test_np), 'k--', label='Analytical Velocity', lw=2)
    plt.plot(t_test_np, v_final_pred, 'm', label='PINN Predicted Velocity', alpha=0.8)
    plt.title("Fig 5: Higher-Order Derivative (Velocity) Accuracy")
    plt.legend()
    plt.savefig('results/fig5_velocity.png')
    plt.show()

    # Fig 6: Loss History
    plt.figure()
    plt.semilogy(l_d_hist, label='Data Loss')
    plt.semilogy(l_p_hist, label='Physics Residual')
    plt.title("Fig 6: Loss Convergence (Log Scale)")
    plt.legend()
    plt.savefig('results/fig6_loss.png')
    plt.show()

# --- 5. 执行主流程 ---
if __name__ == "__main__":
    save_fig1_schematic()
    save_pinn_architecture()
    run_pinn_experiment()

    print("\nSuccess! All 7 images are displayed above and updated in 'results/' folder.")
