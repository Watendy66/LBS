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

v_template = body_model.v_template.detach()
shapedirs = body_model.shapedirs.detach()      # (6890, 3, 10)
posedirs = body_model.posedirs.detach()        # (6890, 3, 207)
faces = body_model.faces

# Step 1: Compute v_shaped with non-zero beta
betas = torch.zeros(1, 10)
betas[0, 0] = 2.0
betas[0, 1] = -1.5

blend_shape = torch.einsum('bl,mkl->bmk', betas, shapedirs)
v_shaped = v_template + blend_shape[0]  # (6890, 3)

# Step 2: Set a non-zero pose (72 params = 24 joints * 3 axis-angle)
# Bend left elbow, raise right shoulder, slight torso twist
pose = torch.zeros(1, 72)
pose[0, 47] = -1.2   # left elbow bend (joint 15*3+2 = 47... actually joint index 18 -> 18*3=54,55,56)
pose[0, 55] = -1.0   # left elbow (joint 18, axis z): bend elbow
pose[0, 49] = 0.8    # right shoulder (joint 16*3+1=49... joint 17 -> 51,52,53)
pose[0, 52] = -0.7   # right shoulder raise

# Step 3: Convert axis-angle to rotation matrices using batch_rodrigues
from smplx.lbs import batch_rodrigues

batch_size = 1
rot_mats = batch_rodrigues(pose.view(-1, 3)).view(batch_size, 24, 3, 3)  # (1, 24, 3, 3)

# Step 4: Construct pose_feature = R - I (exclude root joint)
ident = torch.eye(3, dtype=rot_mats.dtype).unsqueeze(0)  # (1, 3, 3)
pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)  # (1, 207)

# Step 5: Compute pose_offsets = pose_feature @ posedirs
# posedirs: (6890, 3, 207) -> reshape to (6890*3, 207)
# posedirs: (207, 20670), pose_feature: (1, 207)
# result: (1, 20670) -> (1, 6890, 3)
pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)

# Step 6: v_posed = v_shaped + pose_offsets
v_posed = (v_shaped + pose_offsets[0]).detach().numpy()  # (6890, 3)
v_shaped_np = v_shaped.detach().numpy()
pose_offsets_np = pose_offsets[0].detach().numpy()  # (6890, 3)

# Magnitude of pose offsets per vertex
offset_magnitude = np.linalg.norm(pose_offsets_np, axis=1)  # (6890,)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Left: v_shaped (before pose correction)
ax = axes[0]
triang = Triangulation(v_shaped_np[:, 0], v_shaped_np[:, 1], faces)
ax.tripcolor(triang, v_shaped_np[:, 2], cmap='coolwarm', shading='gouraud', alpha=0.7)
ax.set_aspect('equal')
ax.set_title('$v_{shaped}$ (before pose correction)', fontsize=13)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Right: pose offset magnitude heatmap on v_shaped mesh
ax = axes[1]
tcf = ax.tripcolor(triang, offset_magnitude, cmap='magma', shading='gouraud')
ax.set_aspect('equal')
ax.set_title('Pose Offset $\\|B_P(\\theta)\\|$ Magnitude\n(before LBS, corrections only)',
             fontsize=13)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(tcf, ax=ax, label='Offset magnitude')

plt.tight_layout()
plt.savefig('outputs/stage_c_pose_offsets.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/stage_c_pose_offsets.png")

print(f"\nPose offset stats:")
print(f"  Max offset magnitude: {offset_magnitude.max():.6f}")
print(f"  Mean offset magnitude: {offset_magnitude.mean():.6f}")
print(f"  Non-zero vertices (>1e-5): {(offset_magnitude > 1e-5).sum()} / {len(offset_magnitude)}")
