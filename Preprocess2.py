import h5py
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
from tqdm import tqdm
import time
import os
from typing import Tuple
from math import ceil

# ─────── Updated build_and_cache ───────


def build_and_cache(
    raw_h5: str,
    cache_file: str,
    mode: str,
    E_coarse_max_ref: float | None = None,
    force_rebuild: bool = False,
    verbose: bool = False,
    max_events: int = None,
    seed: int = None,
    noise_scale: float = 1e-4,
    compression: str | None = None,
    layer_energies: bool = True,
):
    """
    Final, corrected version. Reads raw data, dequantizes it, builds all
    features using the detailed helper functions, transforms them, and saves
    to a cache file.
    """
    cache = Path(cache_file)
    # Fast‐load if present
    if cache.exists() and not force_rebuild:
        if verbose: print(f"[build_and_cache] Loading from cache.")
        with h5py.File(cache_file,'r') as f:
            return (
                f['showers_raw'][:], f['coarse_showers'][:], f['incident_energies'][:],
                f['fine_trans'][:], f['cond_trans'][:],
                tuple(f['factors'][:]) + (f['E_coarse_max'][()],)
            )

    if verbose: print("Building cache from scratch...")
    rng = np.random.default_rng(seed)

    # 1. READ RAW DATA
    with h5py.File(raw_h5,'r') as f:
        total = f['showers'].shape[0]
        N = min(total, max_events) if max_events else total
        raw = f['showers'][:N]
        E_inc = np.asarray(f['incident_energies'][:N], dtype=np.float32)

    # 2. DEQUANTIZE DATA
    showers_raw = np.asarray(raw.reshape(N, 45, 16, 9), dtype=np.float32)
    noise = rng.uniform(0.0, noise_scale, size=showers_raw.shape)
    showers_noisy = showers_raw + noise

    # 3. DEFINE GEOMETRY AND CREATE NOISY COARSE SHOWERS
    if mode=='A': zf,af,rf = 5,2,1
    else:         zf,af,rf = 1,4,3
    nz,na,nr = 45//zf, 16//af, 9//rf

    cs = showers_noisy.reshape(N, nz, zf, na, af, nr, rf).sum(axis=(2,4,6))
    cs = np.asarray(cs, dtype=np.float32)

    #E_coarse_max = float(cs.max())
    if E_coarse_max_ref is not None:
        E_coarse_max = float(E_coarse_max_ref)
    else:
        E_coarse_max = float(cs.max())

    # 4. CREATE ALL DATA AND CONDITIONAL PIECES USING HELPER FUNCTIONS
    blocks = showers_noisy.reshape(N, nz, zf, na, af, nr, rf).transpose(0,1,3,5,2,4,6)
    
    # These helpers produce already-flattened [M, num_features] arrays
    fine_flat = np.asarray(flatten_fine_blocks(blocks), dtype=np.float32)
    ce = build_cond_event(E_inc, N, nz, na, nr)
    cc = np.asarray(build_cond_coarse(cs), dtype=np.float32)
    cl = None
    if layer_energies:
        cl = np.asarray(build_cond_fine_layer_sums(blocks), dtype=np.float32)
    cn = np.asarray(build_cond_neighbors(cs), dtype=np.float32)
    
    _z, _, _r = np.indices((nz, na, nr))
    layer_idx = np.tile(_z[None], (N, 1, 1, 1))
    radial_bin = np.tile(_r[None], (N, 1, 1, 1))
    lo, ro = build_cond_onehots(layer_idx, radial_bin)
    
    # 5. TRANSFORM THE DATA
    # The transform functions receive the correctly shaped, flattened arrays directly.
    if verbose:
        fine_trans = transform_fine_debug(fine_flat, cc.flatten())
    else:
        fine_trans = transform_fine(fine_flat, cc.flatten())
    
    cond_trans = transform_cond(ce, cc, cn, lo, ro, E_coarse_max=E_coarse_max, cl=cl)
    cond_trans = np.asarray(cond_trans, dtype=np.float32)

    # 6. SAVE TO CACHE
    with h5py.File(cache_file,'w') as f:
        f.create_dataset('showers_raw', data=showers_raw, compression=compression)
        f.create_dataset('coarse_showers', data=cs, compression=compression)
        f.create_dataset('incident_energies', data=E_inc, compression=compression)
        f.create_dataset('fine_trans', data=fine_trans, compression=compression)
        f.create_dataset('cond_trans', data=cond_trans, compression=compression)
        f.create_dataset('factors', data=np.array([zf,af,rf]), compression=compression)
        f.create_dataset('E_coarse_max', data=E_coarse_max)
    
    print(showers_raw.dtype, cs.dtype, E_inc.dtype, fine_trans.dtype, cond_trans.dtype)

    if verbose: print(f"[build_and_cache] Done. E_coarse_max={E_coarse_max:.6g}")

    return (showers_raw, cs, E_inc, fine_trans, cond_trans,
            ((zf,af,rf), E_coarse_max))


def transform_fine(
    fine_flat: np.ndarray,
    cs_flat:   np.ndarray,
    alpha:     float = 1e-6,
) -> np.ndarray:
    """
    Simplified logit transform.
    Assumes `fine_flat` and `cs_flat` are already dequantized (noisy)
    and that `cs_flat` is never zero.
    """
    # Numerator is the already-noisy fine voxel energy
    numerator = fine_flat
    
    # Denominator is the already-noisy coarse voxel energy (guaranteed non-zero)
    denominator = cs_flat[:, None]

    # Compute z. This ratio is now stable.
    z = numerator / denominator
    
    # Apply logit transform with a clip for ultimate safety
    u = alpha + (1.0 - 2.0 * alpha) * z
    #u_clipped = np.clip(u, a_min=alpha, a_max=1.0 - alpha)
    
    return np.log(u / (1.0 - u))


def transform_fine_debug(
    fine_flat: np.ndarray,
    cs_flat:   np.ndarray,
    alpha:     float = 1e-6,
) -> np.ndarray:
    """
    Simplified logit transform.
    Assumes fine_flat and cs_flat are already dequantized (noisy)
    and that cs_flat is never zero.
    """
    numerator   = fine_flat             # [M, Df]
    denominator = cs_flat[:, None]      # [M, 1]

    # Compute z. This ratio is now stable.
    z = numerator / denominator         # [M, Df]

    # Apply logit transform with a clip for ultimate safety
    u         = alpha + (1.0 - 2.0*alpha) * z
    u_clipped = np.clip(u, a_min=alpha, a_max=1.0 - alpha)

    # --- DEBUG: find anything hitting the lower tail alpha ---
    # flatten all M*Df voxels
    flat_u       = u.ravel()
    flat_u_clip  = u_clipped.ravel()
    flat_z       = z.ravel()
    flat_num     = numerator.ravel()
    flat_den     = np.repeat(denominator, numerator.shape[1], axis=1).ravel()

    # mask for those at the exact lower boundary
    mask_low = np.isclose(flat_u_clip, alpha, atol=1e-12)
    if mask_low.any():
        print(f"[transform_fine DEBUG] Detected {mask_low.sum()} voxels at lower clip")
        # print up to first 10 for inspection
        idxs = np.where(mask_low)[0][:10]
        for idx in idxs:
            print(f"  idx={idx}: num={flat_num[idx]:.6g}, den={flat_den[idx]:.6g}, "
                  f"z={flat_z[idx]:.6g}, u={flat_u[idx]:.6g}")

    #return np.log(u_clipped / (1.0 - u_clipped))
    return np.log(u / (1.0 - u))



def transform_cond(
    ce: np.ndarray,
    cc: np.ndarray,
    cn: np.ndarray,
    lo: np.ndarray,
    ro: np.ndarray,
    E_coarse_max: float,
    alpha: float = 1e-6,
    cl = None,
) -> np.ndarray:
    """
    Apply per‐feature transforms.
    The `cc` (coarse-energy) input is assumed to be already dequantized.
    """
    M = cc.shape[0]

    # 1) event energy
    Eev = ce[:,0]
    ce_t = np.log10(Eev / (10**4.5)).reshape(M,1)

    # 2) coarse energy - NO NEED to add more noise
    zc = cc / E_coarse_max
    uc = alpha + (1 - 2*alpha)*zc
    uc = np.clip(uc, alpha, 1-alpha)
    cc_t = np.log(uc / (1 - uc))



    # 4) neighbors - Also dequantized
    zn = cn / E_coarse_max
    un = alpha + (1 - 2*alpha)*zn
    un = np.clip(un, alpha, 1-alpha)
    cn_t = np.log(un / (1 - un))

    # 5) one‐hots: unchanged
    lo_t = lo
    ro_t = ro

    # 3) layer sums - These are sums of noisy fine-voxels, so also dequantized
    if cl is not None:
        zl = cl / 65000.0
        ul = alpha + (1 - 2*alpha)*zl
        ul = np.clip(ul, alpha, 1-alpha)
        cl_t = np.log(ul / (1 - ul))

        return np.concatenate([ce_t, cc_t, cl_t, cn_t, lo_t, ro_t], axis=1)
    else:
        return np.concatenate([ce_t, cc_t, cn_t, lo_t, ro_t], axis=1)



# Helper functions
def flatten_fine_blocks(blocks): return blocks.reshape(-1, blocks.shape[-3]*blocks.shape[-2]*blocks.shape[-1])
def build_cond_event(E_inc, N, nz, na, nr): return np.repeat(E_inc, nz*na*nr).reshape(-1,1)
def build_cond_coarse(cs): return cs.reshape(-1,1)

def build_cond_fine_layer_sums(blocks):
    """
    From blocks [N,nz,na,nr,zf,af,rf], sum over af,rf → per-layer sums:
    → [N,nz,na,nr,zf] → flatten→ [N*540, zf]
    """
    sums = blocks.sum(axis=(5,6))                        # sum af,rf: [N,nz,na,nr,zf]
    return sums.reshape(-1, sums.shape[-1])


def build_cond_neighbors(cs: np.ndarray) -> np.ndarray:
    """
    From cs [N, nz, na, nr], build a neighbor‐energy array:
      for each (z,i,j) index you get energies of
        [z-1, i, j], [z+1, i, j],  (±z)
        [z,   i-1, j], [z,   i+1, j],  (±alpha)
        [z,   i,   j-1], [z,   i,   j+1]  (±r)
      out‐of‐bounds → 0.
    Returns:
      cn: np.ndarray, shape [N * nz * na * nr, 6]
          ordering: [z-, z+, α-, α+, r-, r+]
    """
    N, nz, na, nr = cs.shape
    # pad with one‐voxel border of zeros on all 3 dims
    padded = np.zeros((N, nz+2, na+2, nr+2), dtype=cs.dtype)
    padded[:, 1:nz+1, 1:na+1, 1:nr+1] = cs

    # for each direction, slice the padded array
    n_zm = padded[:, 0: nz   , 1:na+1, 1:nr+1]  # z-1
    n_zp = padded[:, 2:nz+2 , 1:na+1, 1:nr+1]  # z+1
    n_am = padded[:, 1:nz+1 , 0:na   , 1:nr+1]  # α-1
    n_ap = padded[:, 1:nz+1 , 2:na+2, 1:nr+1]  # α+1
    n_rm = padded[:, 1:nz+1 , 1:na+1, 0:nr   ]  # r-1
    n_rp = padded[:, 1:nz+1 , 1:na+1, 2:nr+2]  # r+1

    # stack in the desired order → shape [N, nz, na, nr, 6]
    neighbors = np.stack([n_zm, n_zp, n_am, n_ap, n_rm, n_rp], axis=-1)

    # flatten to [N*nz*na*nr, 6]
    return neighbors.reshape(-1, 6)


def build_cond_onehots(layer_idx, radial_bin):
    """
    layer_idx, radial_bin: [N,nz,na,nr] ints
    → produce one‐hot [N,nz,na,nr,L], [N,nz,na,nr,R]
    → flatten both to [N*540,L] and [N*540,R]
    """
    N,nz,na,nr = layer_idx.shape
    L = layer_idx.max()+1
    R = radial_bin.max()+1

    lo = np.eye(L)[layer_idx]     # [N,nz,na,nr,L]
    ro = np.eye(R)[radial_bin]    # [N,nz,na,nr,R]

    lo = lo.reshape(-1, L)
    ro = ro.reshape(-1, R)
    return lo, ro


def load_train_val_tensors(
    cache_file: str,
    val_split: float = 0.3,
    shuffle: bool = True,
    seed: int = None,
    precision = tf.float32
):
    """
    Corrected version that concatenates fine-voxel data and conditional data
    into a single input tensor 'x', as requested.

    Returns:
      x_train: tf.Tensor, shape [N_train, D_fine + D_cond], the combined input.
      y_train: tf.Tensor, zeros, shape [N_train, 1], a dummy target.
      x_val:   tf.Tensor, shape [N_val, D_fine + D_cond], the validation input.
      y_val:   tf.Tensor, zeros, shape [N_val, 1], a dummy validation target.
    """
    # 1) Load the two separate transformed arrays
    with h5py.File(cache_file, 'r') as f:
        fine_data = f['fine_trans'][:]    # Shape: [M, 12]
        cond_data = f['cond_trans'][:]    # Shape: [M, 57]

    M = fine_data.shape[0]
    assert M == cond_data.shape[0]

    # 2) Sanity check the loaded data
    assert not np.isnan(fine_data).any(), "fine_data has NaNs!"
    assert not np.isinf(fine_data).any(), "fine_data has Infs!"
    assert not np.isnan(cond_data).any(), "cond_data has NaNs!"
    assert not np.isinf(cond_data).any(), "cond_data has Infs!"

    # =============================================================
    #                 START OF THE FIX
    # =============================================================
    # 3) Concatenate the data and conditions into a single array X
    X_full = np.concatenate([fine_data, cond_data], axis=1)
    # =============================================================
    #                  END OF THE FIX
    # =============================================================

    # 4) Create shuffled indices for consistent splitting
    idx = np.arange(M)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    # 5) Split indices into training and validation sets
    n_val     = int(M * val_split)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    # 6) Create the final NumPy arrays
    # x_train and x_val are slices of the concatenated array
    x_train_np = X_full[train_idx]
    x_val_np   = X_full[val_idx]

    # y_train and y_val are dummy zero targets. A shape of (N, 1) is standard.
    if precision == tf.float64:
        y_train_np = np.zeros_like(x_train_np, dtype=np.float64)
        y_val_np   = np.zeros_like(x_val_np,   dtype=np.float64)
    elif precision == tf.float32:
        y_train_np = np.zeros_like(x_train_np, dtype=np.float32)
        y_val_np   = np.zeros_like(x_val_np,   dtype=np.float32)
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use tf.float32 or tf.float64.")

    # 7) Convert to tf.Tensor
    x_train = tf.convert_to_tensor(x_train_np, dtype=precision)
    y_train = tf.convert_to_tensor(y_train_np, dtype=precision)
    x_val   = tf.convert_to_tensor(x_val_np,   dtype=precision)
    y_val   = tf.convert_to_tensor(y_val_np,   dtype=precision)

    # 8) Sanity check & report final shapes
    print(f"Shapes prepared for Trainer (Concatenated format):")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  x_val:   {x_val.shape}")
    print(f"  y_val:   {y_val.shape}")

    return x_train, y_train, x_val, y_val

'''
def build_train_val_datasets(
    cache_file : str,
    val_split  : float       = 0.30,
    batch_size : int         = 4096,
    shuffle    : bool        = True,
    seed       : int | None  = None,
    precision  : tf.dtypes.DType = tf.float32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset,
           tf.data.Dataset, tf.data.Dataset]:
    """
    Returns four tf.data.Dataset objects:
        x_train_ds, y_train_ds, x_val_ds, y_val_ds
    Each *x* element is a (69,) float32 vector.
    Each *y* element is a (69,) float32 vector of zeros.
    """

    hf   = h5py.File(cache_file, "r")       # keep open; datasets use it
    fine = hf["fine_trans"]                 # [M, 12] if mode B
    cond = hf["cond_trans"]                 # [M, 57] if mode B
    M    = fine.shape[0]

    total_dim = fine.shape[1] + cond.shape[1]

    print(cond.shape, fine.shape)
    rng  = np.random.default_rng(seed)
    idx  = np.arange(M)
    if shuffle:
        rng.shuffle(idx)

    n_val   = int(M * val_split)
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    def make_xy_dataset(indices, do_shuffle):
        # local copy so val set stays ordered if desired
        loc = np.array(indices)
        if do_shuffle:
            rng.shuffle(loc)

        def gen():
            for i in loc:
                x = np.concatenate([fine[i], cond[i]]).astype(
                        precision.as_numpy_dtype)
                y = np.zeros_like(x)        # y has same shape as x
                yield x, y

        sig_x = tf.TensorSpec([total_dim], precision)
        sig_y = tf.TensorSpec([total_dim], precision)
        ds = (tf.data.Dataset.from_generator(gen, output_signature=(sig_x, sig_y))
                .batch(batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE))
        return ds

    train_ds = make_xy_dataset(trn_idx, do_shuffle=True)
    val_ds   = make_xy_dataset(val_idx, do_shuffle=False)

    # unpack into four datasets expected by your Trainer ctor
    x_train_ds = train_ds.map(lambda x, y: x)
    y_train_ds = train_ds.map(lambda x, y: y)
    x_val_ds   = val_ds.map(lambda x, y: x)
    y_val_ds   = val_ds.map(lambda x, y: y)

    return x_train_ds, y_train_ds, x_val_ds, y_val_ds
'''

def build_train_val_datasets(
    cache_file : str,
    val_split  : float       = 0.30,
    batch_size : int         = 4096,
    shuffle    : bool        = True,
    seed       : int | None  = None,
    conditionals: bool = True,
    precision  : tf.dtypes.DType = tf.float32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset,
           tf.data.Dataset, tf.data.Dataset]:
    import numpy as np
    import tensorflow as tf
    import h5py

    # 1) Load entire arrays into RAM once
    with h5py.File(cache_file, "r") as hf:
        fine_np = hf["fine_trans"][...] .astype(precision.as_numpy_dtype)
        cond_np = hf["cond_trans"][...] .astype(precision.as_numpy_dtype)

    # 2) Build X and Y arrays
    if conditionals: 
        X_all = np.concatenate([fine_np, cond_np], axis=1)
    else: 
        X_all = fine_np
    Y_all = np.zeros_like(X_all)

    # 3) Shuffle + split indices
    M = X_all.shape[0]
    idx = np.arange(M)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)

    n_val   = int(M * val_split)
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_tr, Y_tr = X_all[trn_idx], Y_all[trn_idx]
    X_va, Y_va = X_all[val_idx], Y_all[val_idx]

    # 4) Create Dataset objects via from_tensor_slices
    with tf.device('/CPU:0'):
        train_ds = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
        val_ds   = tf.data.Dataset.from_tensor_slices((X_va, Y_va))
        
    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=len(X_tr), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 5) If you really need four separate datasets:
    x_train_ds = train_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
    y_train_ds = train_ds.map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE)
    x_val_ds   = val_ds  .map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
    y_val_ds   = val_ds  .map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE)

    return x_train_ds, y_train_ds, x_val_ds, y_val_ds