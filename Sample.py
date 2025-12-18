import h5py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any

'''
def get_conditional_event(
    cache_file: str,
    mode: str,
    event_id: int,
    eps: float = 1e-6, 
    dtype: Optional[tf.DType] = tf.float32
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Open the HDF5 cache, pick out a single event’s raw conditional features,
    and apply the same log‐standardize transform used in training.

    Args:
      - cache_file: path to the preproc_cache .hdf5 file
      - mode:       'A' or 'B' (if 'A' ⇒ V = 648 coarse voxels; if 'B' ⇒ V = 540)
      - event_id:   which event (0 ≤ event_id < N)
      - eps:        small constant to avoid log(0), same as in training

    Returns:
      - cond_event_Z: tf.Tensor of shape [V, C], where each entry =
          ( log(raw_cond + eps) − μ_cond ) / σ_cond
      - E_event_Z:    tf.Tensor scalar = cond_event_Z[0, 0]
    """
    # 1) Set V based on mode
    if mode == 'A':
        V = 648
    elif mode == 'B':
        V = 540
    else:
        raise ValueError("mode must be 'A' (V=648) or 'B' (V=540).")

    # 2) Open the cache and read cond_flat, scale_mu, scale_sigma
    with h5py.File(cache_file, 'r') as f:
        cond_flat_np = f['cond_trans'][:]        # shape [N*V, C], raw (unscaled) conditionals


    # 4) Extract exactly the block [event_id*V : (event_id+1)*V] from cond_flat
    start = event_id * V
    end   = start + V
    cond_event_raw_np = cond_flat_np[start:end, :]  # shape [V, C], still NumPy

    # 5) Convert raw conditionals to tf.Tensor
    cond_event_raw = tf.convert_to_tensor(cond_event_raw_np, dtype=dtype)  # [V, C]

    # 6) Extract the energy for this event (first column, first row)
    E_event = tf.cast(cond_event_raw[0, 0], dtype=dtype)  # scalar, first coarse voxel's energy

    return cond_event_raw, E_event
'''


def get_conditional_event(
    cache_file: str,
    mode: str,
    event_id: int,
    eps: float = 1e-6,
    dtype: tf.DType = tf.float32
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    … docstring unchanged, but now also returns coarse_E_flat …
    """
    # 1) geometry (unchanged)
    if mode == 'A':
        V = 648
    elif mode == 'B':
        V = 540
    else:
        raise ValueError("mode must be 'A' or 'B'.")

    with h5py.File(cache_file, 'r') as f:
        cond_flat_np   = f['cond_trans'][:]            # [N*V, C]
        coarse_showers = f['coarse_showers'][event_id] # shape [nz, na, nr]

    # --- existing block extracting cond_event_raw ---
    start = event_id * V
    end   = start + V
    cond_event_raw_np = cond_flat_np[start:end, :]      # [V, C]
    cond_event_raw    = tf.convert_to_tensor(cond_event_raw_np, dtype=dtype)

    E_event = tf.cast(cond_event_raw[0, 0], dtype=dtype)

    # NEW: flatten the coarse energies for the same event
    coarse_E_flat = tf.convert_to_tensor(coarse_showers.reshape(-1), dtype=dtype) # [V]

    return cond_event_raw, E_event, coarse_E_flat


def sample_event(
    flow: tfp.bijectors.Chain,
    base_dist: tfp.distributions.Distribution,
    cond_event: tf.Tensor,
    clip_latent: bool = False,
    latent_clip_value: float = 3.0,
    verbose: bool = False
) -> tf.Tensor:
    """
    Draw one fine‐voxel sample per coarse voxel given cond_event [V, C].

    Args:
      - flow:                a tfp.bijectors.Chain for the fine‐voxel flow
      - base_dist:           e.g. MultivariateNormalDiag(loc=0, scale=1)
      - cond_event:          tf.Tensor of shape [V, C]
      - clip_latent:         whether to clamp latent z to [-latent_clip_value, +latent_clip_value]
      - latent_clip_value:   maximum |z| allowed (default 3.0)
      - verbose:             if True, print debug info

    Returns:
      - x: tf.Tensor of shape [V, D_fine], the flow’s output in latent‐space
    """
    V = tf.shape(cond_event)[0]   # number of coarse voxels

    # 1) Draw V independent latent samples from N(0, I)
    z = base_dist.sample(sample_shape=V)  # shape [V, D_fine]

    if verbose:
        z_min = tf.reduce_min(z)
        z_max = tf.reduce_max(z)
        tf.print("DEBUG • base_dist.sample() → z range before clip:", z_min, z_max)

    # 2) Optionally clamp each coordinate of z to [-latent_clip_value, +latent_clip_value]
    if clip_latent:
        z = tf.clip_by_value(z, -latent_clip_value, latent_clip_value)
        if verbose:
            z_min_c = tf.reduce_min(z)
            z_max_c = tf.reduce_max(z)
            tf.print(f"DEBUG • z range after clip to ±{latent_clip_value}:", z_min_c, z_max_c)

    # 3) Push through each bijector (in reverse order)
    x = z
    for i, bij in enumerate(reversed(flow.bijectors)):
        '''
        try:
            x = bij.forward(x, conditional_input=cond_event)
        except (TypeError, ValueError):
            x = bij.forward(x)

        if verbose:
            x_min = tf.reduce_min(x)
            x_max = tf.reduce_max(x)
            tf.print(f"DEBUG • after bijector #{i} ({type(bij).__name__}):",
                     "min=", x_min, "max=", x_max)
        '''
        cond_fn = getattr(bij, "_shift_and_log_scale_fn", None)
        is_conditional = getattr(cond_fn, "_conditional", False)

        if is_conditional:
            x = bij.forward(x, conditional_input=cond_event)
        else:
            x = bij.forward(x)

        if verbose:
            tf.print(f"[DEBUG] after {type(bij).__name__} #{i}:",
                     "min=", tf.reduce_min(x),
                     " max=", tf.reduce_max(x))
    return x


'''
def reshape_to_calo_flat(fine_samples: tf.Tensor) -> np.ndarray:
    """
    Given `fine_samples` of shape [V, D_fine], return a NumPy array of shape [1, V * D_fine].
    This flattens all sub-voxels from each coarse cell in row-major (C-order) and then
    adds a leading batch dimension.
    """
    # 1) Get V and D_fine
    V, Df = fine_samples.shape
    
    # 2) Flatten to a 1D tensor of length (V * D_fine)
    flat_tensor = tf.reshape(fine_samples, [V * Df])  # shape: (V*D_fine,)
    
    # 3) Convert to NumPy and add batch dimension
    flat_np = flat_tensor.numpy()            # shape: (V*D_fine,)
    batch_np = flat_np[np.newaxis, :]        # shape: (1, V*D_fine)
    
    return batch_np
'''

def reshape_to_calo_flat(fine_samples: tf.Tensor,
                         mode: str) -> np.ndarray:
    """
    fine_samples: [V, zf*af*rf] where V = nz*na*nr
    mode: 'A' or 'B' (to infer zf, af, rf)
    Returns: np.ndarray of shape [1, 45*16*9] in the dataset's (L, A, R) order.
    """
    if mode == 'A':
        zf, af, rf = 5, 2, 1
    elif mode == 'B':
        zf, af, rf = 1, 4, 3
    else:
        raise ValueError("mode must be 'A' or 'B'.")

    L, A, R = 45, 16, 9
    nz, na, nr = L // zf, A // af, R // rf

    # sanity checks
    V = tf.shape(fine_samples)[0]
    Df = tf.shape(fine_samples)[1]
    tf.debugging.assert_equal(V, nz*na*nr, message="V mismatch with nz*na*nr")
    tf.debugging.assert_equal(Df, zf*af*rf, message="Df mismatch with zf*af*rf")

    # [V, Df] → [nz, na, nr, zf, af, rf]
    arr = tf.reshape(fine_samples, [nz, na, nr, zf, af, rf])
    # reorder to [nz, zf, na, af, nr, rf]
    arr = tf.transpose(arr, [0, 3, 1, 4, 2, 5])
    # collapse pairs → [L, A, R]
    arr = tf.reshape(arr, [L, A, R])

    # final flatten with R fastest (C-order)
    flat = tf.reshape(arr, [L*A*R])
    return flat[tf.newaxis, :].numpy()



def invert_transform_fine(
    fine_trans: tf.Tensor | np.ndarray,
    coarse_E_flat: tf.Tensor | np.ndarray,
    alpha: float = 1e-6,
    dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """
    Invert the `transform_fine` / `transform_fine_debug` preprocessing.

    Args
    ----
      fine_trans      : [V, D_fine] tensor – network output in logit space
      coarse_E_flat   : [V] tensor     – *raw* coarse voxel energies (same event)
      alpha           : clip constant used during training (default 1 e-6)

    Returns
    -------
      fine_E          : tf.Tensor, shape [V, D_fine] – physical fine-voxel
                        energies (still contains the tiny de-quantisation noise)
    """
    # 1) Cast both inputs to the same dtype (float32/64)
    fine_trans   = tf.cast(fine_trans,   dtype)
    coarse_E_flat = tf.cast(coarse_E_flat, dtype)                      # [V]

    # 2) σ(t)  →  u  (undo logit)
    u = tf.math.sigmoid(fine_trans)                                    # [V, Df]

    # 3) u  →  z  (undo affine part)
    denom = tf.constant(1.0 - 2.0 * alpha, dtype)
    z = (u - alpha) / denom                                            # in [0,1]

    # 4) z  →  E_fine   (multiply by parent coarse energy)
    fine_E = z * coarse_E_flat[:, None]                                # broadcast
    fine_E = tf.maximum(fine_E, 0.0) 
    
    # Optional hard safety clamp (rarely needed):
    # fine_E = tf.clip_by_value(fine_E, clip_value_min=0.0, clip_value_max=np.inf)

    return fine_E

