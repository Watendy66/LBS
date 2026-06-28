import sys
import types
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
faces = body_model.faces

# Consistent parameters across all stages
betas = torch.zeros(1, 10)
betas[0, 0] = 2.0
betas[0, 1] = -1.5

# Moderate pose: slight arm raise, not too extreme to avoid 2D projection overlap
body_pose = torch.zeros(1, 69)
body_pose[0, 16*3 + 2] = 0.3    # left shoulder slight outward
body_pose[0, 17*3 + 2] = -0.3   # right shoulder slight outward
body_pose[0, 15*3 + 1] = 0.4    # left elbow slight bend
body_pose[0, 18*3 + 1] = 0.8    # left elbow bend more

global_orient = torch.zeros(1, 3)

# --- Compute all stages from a single forward pass for consistency ---
output = body_model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)

# (a) Template
v_template = body_model.v_template.detach().numpy()
lbs_weights = body_model.lbs_weights.detach().numpy()

# (b) Shaped mesh + joints
shapedirs = body_model.shapedirs.detach()
posedirs = body_model.posedirs.detach()
J_regressor = body_model.J_regressor.to_dense().numpy()

blend_shape = torch.einsum('bl,mkl->bmk', betas, shapedirs)  # (1, 6890, 3)
v_shaped = (body_model.v_template.detach() + blend_shape[0]).numpy()
J = J_regressor @ v_shaped

# (c) Pose offsets
from smplx.lbs import batch_rodrigues
full_pose = torch.cat([global_orient, body_pose], dim=1)
rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, 24, 3, 3)
ident = torch.eye(3, dtype=rot_mats.dtype).unsqueeze(0)
pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
offset_mag = np.linalg.norm(pose_offsets[0].detach().numpy(), axis=1)

# (d) Final verts
verts = output.vertices.detach().numpy()[0]
J_transformed = output.joints.detach().numpy()[0][:24]

# --- Compute unified axis limits from the largest extent (final verts) ---
all_x = np.concatenate([v_template[:, 0], v_shaped[:, 0], verts[:, 0]])
all_y = np.concatenate([v_template[:, 1], v_shaped[:, 1], verts[:, 1]])
margin = 0.1
xlim = (all_x.min() - margin, all_x.max() + margin)
ylim = (all_y.min() - margin, all_y.max() + margin)

# --- Plot 2x2 grid with consistent sizes ---
fig, axes = plt.subplots(2, 2, figsize=(12, 16))

# (a) Template + skinning weights (L_Elbow)
ax = axes[0, 0]
triang_t = Triangulation(v_template[:, 0], v_template[:, 1], faces)
ax.tripcolor(triang_t, lbs_weights[:, 18], cmap='hot', shading='gouraud', vmin=0, vmax=1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_title('(a) Template $\\bar{T}$ + Weights $\\mathcal{W}$\n(L_Elbow shown)', fontsize=11)

# (b) Shaped mesh + joints
ax = axes[0, 1]
triang_s = Triangulation(v_shaped[:, 0], v_shaped[:, 1], faces)
ax.tripcolor(triang_s, v_shaped[:, 2], cmap='coolwarm', shading='gouraud', alpha=0.7)
ax.scatter(J[:, 0], J[:, 1], c='lime', edgecolors='black', s=35, zorder=5)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_title('(b) Shaped $\\bar{T}+B_S(\\beta)$ + Joints $J(\\beta)$', fontsize=11)

# (c) Pose offsets magnitude
ax = axes[1, 0]
ax.tripcolor(triang_s, offset_mag, cmap='magma', shading='gouraud')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_title('(c) Pose Offsets $\\|B_P(\\theta)\\|$\n(before LBS)', fontsize=11)

# (d) Final LBS result
ax = axes[1, 1]
triang_f = Triangulation(verts[:, 0], verts[:, 1], faces)
ax.tripcolor(triang_f, verts[:, 2], cmap='coolwarm', shading='gouraud', alpha=0.8)
ax.scatter(J_transformed[:, 0], J_transformed[:, 1], c='lime', edgecolors='black',
           s=35, zorder=5)
kinematic_tree = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)
]
for (a, b) in kinematic_tree:
    ax.plot([J_transformed[a, 0], J_transformed[b, 0]],
            [J_transformed[a, 1], J_transformed[b, 1]],
            'g-', linewidth=1.2, alpha=0.7)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_title('(d) Final LBS $W(T_P, J, \\theta, \\mathcal{W})$', fontsize=11)

plt.suptitle('SMPL Linear Blend Skinning - Four Stages', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('outputs/comparison_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/comparison_grid.png")
