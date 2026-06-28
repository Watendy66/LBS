import sys
import types
import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import imageio.v2 as imageio
import os

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
lbs_weights = body_model.lbs_weights.detach().numpy()

# Fixed shape
betas = torch.zeros(1, 10)
betas[0, 0] = 1.0

# Animation: left elbow (joint 18) bending from 0 to -2.0 radians and back
n_frames = 30
angles = np.concatenate([
    np.linspace(0, -2.0, n_frames // 2),
    np.linspace(-2.0, 0, n_frames // 2)
])

# Joint 18 = L_Elbow, weight for visualization
elbow_weights = lbs_weights[:, 18]

# Precompute axis limits from extreme pose
body_pose_test = torch.zeros(1, 69)
body_pose_test[0, 17*3 + 2] = -2.0
output_test = body_model(betas=betas, body_pose=body_pose_test, global_orient=torch.zeros(1, 3))
verts_test = output_test.vertices.detach().numpy()[0]

# Also check rest pose
output_rest = body_model(betas=betas, body_pose=torch.zeros(1, 69), global_orient=torch.zeros(1, 3))
verts_rest = output_rest.vertices.detach().numpy()[0]

all_x = np.concatenate([verts_test[:, 0], verts_rest[:, 0]])
all_y = np.concatenate([verts_test[:, 1], verts_rest[:, 1]])
margin = 0.1
xlim = (all_x.min() - margin, all_x.max() + margin)
ylim = (all_y.min() - margin, all_y.max() + margin)

# Generate frames
frame_dir = 'outputs/frames'
os.makedirs(frame_dir, exist_ok=True)

print(f"Generating {n_frames} frames...")
frame_paths = []

for idx, angle in enumerate(angles):
    body_pose = torch.zeros(1, 69)
    body_pose[0, 17*3 + 2] = angle  # L_Elbow rotation around Z axis (bends in XY plane)

    output = body_model(betas=betas, body_pose=body_pose, global_orient=torch.zeros(1, 3))
    verts = output.vertices.detach().numpy()[0]
    J_transformed = output.joints.detach().numpy()[0][:24]

    fig, ax = plt.subplots(1, 1, figsize=(6, 9))

    # Draw mesh colored by elbow weight (use 'YlOrRd' so low-weight is light yellow, not black)
    triang = Triangulation(verts[:, 0], verts[:, 1], faces)
    ax.tripcolor(triang, elbow_weights, cmap='YlOrRd', shading='gouraud',
                 vmin=0, vmax=1, alpha=0.85)

    # Draw mesh outline with light gray for body visibility
    ax.triplot(triang, color='gray', linewidth=0.05, alpha=0.3)

    # Draw skeleton
    kinematic_tree = [
        (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
        (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
        (12, 13), (12, 14), (12, 15), (13, 16), (14, 17),
        (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)
    ]
    for (a, b) in kinematic_tree:
        ax.plot([J_transformed[a, 0], J_transformed[b, 0]],
                [J_transformed[a, 1], J_transformed[b, 1]],
                'b-', linewidth=1.5, alpha=0.7)
    ax.scatter(J_transformed[:, 0], J_transformed[:, 1], c='blue',
               edgecolors='black', s=25, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'L_Elbow Bend Animation (weight heatmap)\n'
                 f'Angle = {np.degrees(angle):.1f}°  (frame {idx+1}/{n_frames})',
                 fontsize=11)

    frame_path = os.path.join(frame_dir, f'frame_{idx:03d}.png')
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close()
    frame_paths.append(frame_path)

# Assemble GIF
print("Assembling GIF...")
images = [imageio.imread(p) for p in frame_paths]
imageio.mimsave('outputs/pose_animation.gif', images, duration=0.1, loop=0)
print("Saved: outputs/pose_animation.gif")

# Clean up frames
for p in frame_paths:
    os.remove(p)
os.rmdir(frame_dir)
print("Cleaned up frame files.")
