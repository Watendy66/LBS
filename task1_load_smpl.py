import sys
import types
import pickle
import numpy as np
import torch
import smplx

# Mock chumpy module so pickle can load SMPL pkl without chumpy installed
chumpy_mod = types.ModuleType('chumpy')

class FakeChArray:
    """Stand-in for chumpy arrays during unpickling."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict) and 'x' in state:
            self.data = np.array(state['x'])
        elif isinstance(state, np.ndarray):
            self.data = state
        else:
            self.data = np.zeros(0)
    def __array__(self):
        return self.data

chumpy_mod.Ch = FakeChArray
chumpy_mod.array = FakeChArray
sys.modules['chumpy'] = chumpy_mod
sys.modules['chumpy.ch'] = chumpy_mod
sys.modules['chumpy.ch_ops'] = chumpy_mod

# Load pkl
pkl_path = 'SMPL_NEUTRAL.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Convert chumpy-like objects to numpy
np_data = {}
for key, val in data.items():
    if isinstance(val, FakeChArray):
        np_data[key] = np.array(val)
    elif hasattr(val, 'r'):
        np_data[key] = np.array(val.r)
    elif isinstance(val, np.ndarray):
        np_data[key] = val
    else:
        np_data[key] = val

# Save clean pkl for smplx
clean_pkl = 'smpl/SMPL_NEUTRAL.pkl'
with open(clean_pkl, 'wb') as f:
    pickle.dump(np_data, f)

# Now load with smplx
body_model = smplx.create(
    model_path='.',
    model_type='smpl',
    gender='neutral'
)

print("=== SMPL 模型基础信息 ===")
print(f"顶点数: {body_model.get_num_verts()}")
print(f"面片数: {body_model.faces.shape[0]}")
print(f"关节数: {body_model.J_regressor.shape[0]}")
print(f"betas 维度: {body_model.shapedirs.shape[-1]}")
