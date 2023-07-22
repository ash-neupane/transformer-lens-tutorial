import os
import sys
import plotly.express as px
import torch
from pathlib import Path
import numpy as np
import einops
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
# from IPython.display import display
# import webbrowser
# import gdown

from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

import os, sys, json
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_grad_enabled(False)
device_name = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
torch.set_grad_enabled(False)
print(f"{device=}")

# GPT-2
gpt2_small = HookedTransformer.from_pretrained("gpt2-small")
print(type(gpt2_small))
print(gpt2_small.cfg)