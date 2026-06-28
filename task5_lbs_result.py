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

faces = body_model.faces

# Set parameters
betas = torch.zeros(1, 10)
betas[0, 0] = 2.0
betas[0, 1] = -1.5

# Pose: bend left elbow, raise right shoulder
body_pose = torch.zeros(1, 69)  # 23 joints * 3
body_pose[0, 17*3 + 2] = -0.7   # right shoulder Z
body_pose[0, 16*3 + 2] = 0.7    # left shoulder Z
body_pose[0, 17*3 + 1] = -0.5   # right elbow
body_pose[0, 18*3 + 1] = 1.2    # left elbow bend

global_orient = torch.zeros(1, 3)

# Run the full SMPL forward pass
output = body_model(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient,
    return_verts=True
)

verts = output.vertices.detach().numpy()[0]  # (6890, 3)
joints = output.joints.detach().numpy()[0]   # (24+, 3) - transformed joints

# Only take first 24 joints (SMPL joints)
J_transformed = joints[:24]

# Joint names
joint_names = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(8, 12))

triang = Triangulation(verts[:, 0], verts[:, 1], faces)
ax.tripcolor(triang, verts[:, 2], cmap='coolwarm', shading='gouraud', alpha=0.8)
ax.scatter(J_transformed[:, 0], J_transformed[:, 1], c='lime', edgecolors='black',
           s=50, zorder=5, label='Transformed Joints')

# Draw skeleton connections
kinematic_tree = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)
]
for (a, b) in kinematic_tree:
    ax.plot([J_transformed[a, 0], J_transformed[b, 0]],
            [J_transformed[a, 1], J_transformed[b, 1]],
            'g-', linewidth=1.5, alpha=0.7)

ax.set_aspect('equal')
ax.set_title('Final LBS Result: Skinned Mesh + Skeleton', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('outputs/stage_d_lbs_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/stage_d_lbs_result.png")
