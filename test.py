import time
import numpy as np
import jax
from model import GPT, GPTConfig, convert_hf_params
from flax.traverse_util import flatten_dict

def test_nanogpt():
    key = jax.random.PRNGKey(0)
    key, key_idxs, key_params = jax.random.split(key, 3)

    model = GPT(config)

    params = model.init(key_params)

    idxs = jax.random.randint(key_idxs, (2, 32), 0, 256)

    T=[]
    for _ in range(5):
      t1 = time.time()
      y = model.apply(params, idxs, True)
      t2 = time.time()
      T.append(t2 - t1)

    print("Average inference time", np.mean(np.array(T[2:]))) # Ignore the 2 initial executions as they normally takes longer

test_nanogpt()
