import sys
import types
import pickle
import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib import cm

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

# Load clean model (already converted in task1)
body_model = smplx.create(model_path='.', model_type='smpl', gender='neutral')

v_template = body_model.v_template.detach().numpy()  # (6890, 3)
faces = body_model.faces  # (13776, 3)
lbs_weights = body_model.lbs_weights.detach().numpy()  # (6890, 24)

# SMPL joint names for reference
joint_names = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# --- (1) Single joint weight heatmap ---
# Choose left elbow (joint 18) as example - clear spatial distribution
joint_idx = 18
joint_name = joint_names[joint_idx]
weights = lbs_weights[:, joint_idx]

fig, ax = plt.subplots(1, 1, figsize=(8, 12))
x = v_template[:, 0]
y = v_template[:, 1]
triang = Triangulation(x, y, faces)

tcf = ax.tripcolor(triang, weights, cmap='hot', shading='gouraud',
                   vmin=0, vmax=1)
ax.set_aspect('equal')
ax.set_title(f'Joint Weight Heatmap: {joint_name} (joint {joint_idx})',
             fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(tcf, ax=ax, label='Weight')
plt.tight_layout()
plt.savefig('outputs/stage_a_template_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/stage_a_template_weights.png")

# --- (2) All-joint dominant weight distribution ---
dominant_joint = np.argmax(lbs_weights, axis=1)  # (6890,) which joint dominates
dominant_weight = np.max(lbs_weights, axis=1)    # (6890,) how strong

# Per-face: use the dominant joint of the first vertex of each face
face_joint = dominant_joint[faces[:, 0]]
face_weight = dominant_weight[faces[:, 0]]

fig, ax = plt.subplots(1, 1, figsize=(8, 12))

# Color by joint index (hue) modulated by weight strength (brightness)
n_joints = 24
cmap = plt.colormaps['tab20'].resampled(n_joints)
colors = cmap(face_joint / (n_joints - 1))
colors[:, :3] *= face_weight[:, np.newaxis]  # modulate brightness by weight strength

from matplotlib.collections import PolyCollection
verts_2d = v_template[:, :2]
polygons = verts_2d[faces]
poly_collection = PolyCollection(polygons, facecolors=colors, edgecolors='none',
                                  linewidths=0.1)
ax.add_collection(poly_collection)
ax.set_xlim(verts_2d[:, 0].min() - 0.05, verts_2d[:, 0].max() + 0.05)
ax.set_ylim(verts_2d[:, 1].min() - 0.05, verts_2d[:, 1].max() + 0.05)
ax.set_aspect('equal')
ax.set_title('All-Joint Dominant Weight Distribution', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Legend with joint colors
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=cmap(i / (n_joints - 1)), label=joint_names[i])
                  for i in range(n_joints)]
ax.legend(handles=legend_patches, loc='upper right', fontsize=6, ncol=2)

plt.tight_layout()
plt.savefig('outputs/all_joint_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/all_joint_weights.png")
