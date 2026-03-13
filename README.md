## Experimental Results

### 1. Physical System Schematic
<img src="results/fig1_schematic.png" width="500">

### 2. Comparison: PINN vs. Vanilla Neural Network
We demonstrate that without physics constraints, a standard NN fails to generalize from a single data point.
| Vanilla NN (Fail) | PINN Evolution |
| :---: | :---: |
| <img src="results/fig2_vanilla.png" width="400"> | <img src="results/fig3_evolution.png" width="400"> |

### 3. Extrapolation & Derivative Accuracy
PINN shows remarkable performance in regions without training data and accurately predicts the first derivative (velocity).
| Extrapolation Performance | Velocity Prediction |
| :---: | :---: |
| <img src="results/fig4_extrapolation.png" width="400"> | <img src="results/fig5_velocity.png" width="400"> |

### 4. Loss Convergence
The decomposition of loss functions shows the balance between data fitting and physics residuals.
<img src="results/fig6_loss.png" width="500">
