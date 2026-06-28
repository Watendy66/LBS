import sys
import types
import pickle
import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Mock chumpy
chumpy_mod = types.ModuleType('chumpy')

class FakeChArray:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict) and 'x' in state:
            self.data = np.array(state['x'])
        elif isinstance(state, np.ndarray):
            self.data = state
        else:
            self.data = np.zeros(0)
    def __array__(self, dtype=None, copy=None):
        return self.data

chumpy_mod.Ch = FakeChArray
chumpy_mod.array = FakeChArray
sys.modules['chumpy'] = chumpy_mod
sys.modules['chumpy.ch'] = chumpy_mod
sys.modules['chumpy.ch_ops'] = chumpy_mod

# Load model
body_model = smplx.create(model_path='.', model_type='smpl', gender='neutral')

v_template = body_model.v_template.detach()  # (6890, 3)
shapedirs = body_model.shapedirs.detach()    # (6890, 3, 10)
J_regressor = body_model.J_regressor        # sparse (24, 6890)
faces = body_model.faces

# Set non-zero beta: make the person taller and heavier
betas = torch.zeros(1, 10)
betas[0, 0] = 2.0   # first component (overall size/shape variation)
betas[0, 1] = -1.5  # second component

# Compute v_shaped = v_template + blend_shapes(beta, shapedirs)
blend_shape = torch.einsum('bl,mkl->bmk', betas, shapedirs)  # (1, 6890, 3)
v_shaped = (v_template + blend_shape[0]).numpy()  # (6890, 3)
v_template_np = v_template.numpy()

# Compute joints: J = J_regressor @ v_shaped
J_reg_dense = J_regressor.to_dense().numpy()  # (24, 6890)
J = J_reg_dense @ v_shaped  # (24, 3)

# Joint names
joint_names = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Left: template mesh (for comparison)
ax = axes[0]
triang = Triangulation(v_template_np[:, 0], v_template_np[:, 1], faces)
ax.tripcolor(triang, v_template_np[:, 2], cmap='coolwarm', shading='gouraud', alpha=0.7)
ax.set_aspect('equal')
ax.set_title('Template Mesh $\\bar{T}$', fontsize=13)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Right: shaped mesh + joints
ax = axes[1]
triang_s = Triangulation(v_shaped[:, 0], v_shaped[:, 1], faces)
ax.tripcolor(triang_s, v_shaped[:, 2], cmap='coolwarm', shading='gouraud', alpha=0.7)
ax.scatter(J[:, 0], J[:, 1], c='lime', edgecolors='black', s=60, zorder=5, label='Joints')
for i, name in enumerate(joint_names):
    ax.annotate(name, (J[i, 0], J[i, 1]), fontsize=5, ha='center', va='bottom',
                color='white', fontweight='bold')
ax.set_aspect('equal')
ax.set_title('Shaped Mesh $\\bar{T}+B_S(\\beta)$ with Joints $J(\\beta)$\n'
             f'$\\beta_0$={betas[0,0].item():.1f}, $\\beta_1$={betas[0,1].item():.1f}',
             fontsize=13)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('outputs/stage_b_shaped_joints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/stage_b_shaped_joints.png")
