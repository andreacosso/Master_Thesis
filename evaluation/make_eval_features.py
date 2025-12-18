# make_eval_features.py
from __future__ import annotations
import gc
import math
from pathlib import Path
from typing import Optional, Dict, Tuple
import h5py
import numpy as np
import os
import time
from contextlib import nullcontext


# --------- small helpers ---------
def _read_geometry_and_factors(
    hf_res: h5py.File,
    hf_truth: Optional[h5py.File]
) -> Tuple[int, int, int, Tuple[int, int, int]]:
    """
    Retrieve geometry (nz, na, nr) and fine split factors (zf, af, rf).

    Priority:
      - (nz,na,nr): first try results attrs; else fall back to truth['/coarse_showers'].shape[1:4]
      - (zf,af,rf): from truth['/factors'] (this is where Preprocess2 writes them)

    Raises with a clear message if anything essential is missing.
    """
    # --- geometry ---
    def _get_attr_int(attrs, key):
        v = attrs.get(key, None)
        return int(v) if v is not None else None

    nz = _get_attr_int(hf_res.attrs, "nz")
    na = _get_attr_int(hf_res.attrs, "na")
    nr = _get_attr_int(hf_res.attrs, "nr")

    if nz is None or na is None or nr is None:
        if hf_truth is None or "/coarse_showers" not in hf_truth:
            raise KeyError(
                "Could not find (nz,na,nr) in results attrs and no '/coarse_showers' in truth file."
            )
        _, nz, na, nr = hf_truth["/coarse_showers"].shape  # first dim is N

    # --- fine split factors ---
    if hf_truth is None or "/factors" not in hf_truth:
        raise KeyError(
            "Missing '/factors' in truth cache. Preprocess2.build_and_cache should write it as [zf, af, rf]."
        )
    factors = np.asarray(hf_truth["/factors"][...], dtype=np.int64).tolist()
    if len(factors) != 3:
        raise ValueError(f"Expected '/factors' to have length 3; got {len(factors)}")
    zf, af, rf = map(int, factors)

    return int(nz), int(na), int(nr), (zf, af, rf)



def _compute_total_energy(E: np.ndarray) -> np.ndarray:
    """Total energy per event: sum over all fine voxels. E: [B, Df] → [B]."""
    return E.sum(axis=1)

'''
def _compute_layer_energies(
    E: np.ndarray, nz: int, na: int, nr: int, zf: int, af: int, rf: int
) -> np.ndarray:
    """
    Per-layer energy per event, using the known reshape order from Preprocess2:

    Preprocess2 builds 'blocks' as:
        blocks = showers_noisy.reshape(N, nz, zf, na, af, nr, rf).transpose(0,1,3,5,2,4,6)
    so per event the fine vector corresponds to:
        [nz, na, nr, zf, af, rf] flattened in this exact order.

    Here we invert that flattening:
       E [B,Df] → E6 [B,nz,na,nr,zf,af,rf]
       then sum over (na, nr, zf, af, rf) → [B, nz].
    """
    B, Df = E.shape
    V = nz * na * nr
    K = zf * af * rf
    if Df != V * K:
        raise ValueError(
            f"Shape mismatch: Df={Df} but nz*na*nr*zf*af*rf={V*K} "
            f"(nz={nz},na={na},nr={nr}, zf={zf},af={af},rf={rf})."
        )
    E6 = E.reshape(B, nz, na, nr, zf, af, rf)
    # sum over inner dims to get per-layer energy
    return E6.sum(axis=(2, 3, 4, 5, 6))  # → [B, nz]
'''

def _compute_layer_energies(
    E: np.ndarray, nz: int, na: int, nr: int, zf: int, af: int, rf: int
) -> np.ndarray:
    """
    Per-*fine*-layer energy per event.

    Preprocess2 builds 'blocks' as:
        blocks = showers_noisy.reshape(N, nz, zf, na, af, nr, rf).transpose(0,1,3,5,2,4,6)
    so per event the fine vector corresponds to:
        [nz, na, nr, zf, af, rf] flattened in this exact order.

    We invert that flattening:
       E [B,Df] → E6 [B,nz,na,nr,zf,af,rf]

    For *fine*-layer energies we must KEEP zf and sum over (na, nr, af, rf):
       sum over axes (2, 3, 5, 6) → [B, nz, zf] → reshape [B, nz*zf].
    """
    B, Df = E.shape
    V = nz * na * nr
    K = zf * af * rf
    if Df != V * K:
        raise ValueError(
            f"Shape mismatch: Df={Df} but nz*na*nr*zf*af*rf={V*K} "
            f"(nz={nz},na={na},nr={nr}, zf={zf},af={af},rf={rf})."
        )
    E6 = E.reshape(B, nz, na, nr, zf, af, rf)
    # sum over na(2), nr(3), af(5), rf(6); KEEP zf(4)
    L = E6.sum(axis=(2, 3, 5, 6))       # [B, nz, zf]
    return L.reshape(B, nz * zf)        # [B, nz*zf]


# ---------------------------
# Geometry-index helpers
# ---------------------------
def _make_fine_indices(
    nz: int, na: int, nr: int, zf: int, af: int, rf: int,
    *,
    # Optional: convert index -> physical coordinate = offset + scale * index
    z_offset: float = 0.0, z_scale: float = 1.0,
    a_offset: float = 0.0, a_scale: float = 1.0,
    r_offset: float = 0.0, r_scale: float = 1.0,
    dtype=np.float64,
):
    """
    Build broadcastable coordinate index arrays matching the fine layout:
      E6 has shape [B, nz, na, nr, zf, af, rf]

    This returns three arrays with shapes:
      Zidx6: [1, nz,  1,  1, zf, 1, 1]  with values (z * zf + kz)
      Aidx6: [1,  1, na,  1,  1, af, 1] with values (a * af + ka)
      Ridx6: [1,  1,  1, nr,  1,  1, rf] with values (r * rf + kr)

    If you have physical spacings, pass (offset, scale) to map to physical coords.
    """
    # Coarse indices
    zc = np.arange(nz, dtype=np.int64).reshape(1, nz, 1, 1, 1, 1, 1)  # [1,nz,1,1,1,1,1]
    ac = np.arange(na, dtype=np.int64).reshape(1, 1, na, 1, 1, 1, 1)  # [1,1,na,1,1,1,1]
    rc = np.arange(nr, dtype=np.int64).reshape(1, 1, 1, nr, 1, 1, 1)  # [1,1,1,nr,1,1,1]

    # Fine-within-coarse indices
    kz = np.arange(zf, dtype=np.int64).reshape(1, 1, 1, 1, zf, 1, 1)  # [1,1,1,1,zf,1,1]
    ka = np.arange(af, dtype=np.int64).reshape(1, 1, 1, 1, 1, af, 1)  # [1,1,1,1,1,af,1]
    kr = np.arange(rf, dtype=np.int64).reshape(1, 1, 1, 1, 1, 1, rf)  # [1,1,1,1,1,1,rf]

    # Flattened fine indices along each axis
    z_index6 = (zc * zf + kz).astype(dtype)  # [1,nz,1,1,zf,1,1]
    a_index6 = (ac * af + ka).astype(dtype)  # [1,1,na,1,1,af,1]
    r_index6 = (rc * rf + kr).astype(dtype)  # [1,1,1,nr,1,1,rf]

    # Apply optional physical scaling
    z_coord6 = z_offset + z_scale * z_index6
    a_coord6 = a_offset + a_scale * a_index6
    r_coord6 = r_offset + r_scale * r_index6

    return z_coord6, a_coord6, r_coord6


# ---------------------------
# Centroid & RMS (index- or physical-space)
# ---------------------------
def _compute_centroid_and_rms(
    E: np.ndarray,  # [B, Df]
    nz: int, na: int, nr: int, zf: int, af: int, rf: int,
    *,
    z_offset: float = 0.0, z_scale: float = 1.0,
    a_offset: float = 0.0, a_scale: float = 1.0,
    r_offset: float = 0.0, r_scale: float = 1.0,
    eps: float = 1e-12,
    out_dtype = np.float64,
):
    """
    Compute energy-weighted centroids and RMS.

    - Longitudinal (z) centroid and RMS are computed along z directly.
    - Lateral RMS is computed in *Cartesian* coordinates (x = r cos a, y = r sin a),
      so it has consistent physical units (mm) and handles angular periodicity.
    - For diagnostics/compatibility, per-axis RMS in (a, r) are still returned.

    Returns 7 arrays of shape [B]:
      z_bar, a_bar, r_bar, rms_z, rms_a, rms_r, lateral_rms

    Notes
    -----
    * z_offset/scale, a_offset/scale, r_offset/scale map index -> physical coords:
        coord = offset + scale * index
      Typical units: z,r in mm; a in radians.
    * lateral_rms = sqrt(RMS_x^2 + RMS_y^2) with x=r cos a, y=r sin a (in mm).
    """
    B, Df = E.shape
    V = nz * na * nr
    K = zf * af * rf
    if Df != V * K:
        raise ValueError(
            f"Expected Df={V*K} but got Df={Df} (nz={nz},na={na},nr={nr}, zf={zf},af={af},rf={rf})"
        )

    # Reshape E to [B, nz, na, nr, zf, af, rf]
    E6 = E.reshape(B, nz, na, nr, zf, af, rf)

    # Build broadcastable coordinate grids matching E6
    Z6, A6, R6 = _make_fine_indices(
        nz, na, nr, zf, af, rf,
        z_offset=z_offset, z_scale=z_scale,
        a_offset=a_offset, a_scale=a_scale,
        r_offset=r_offset, r_scale=r_scale,
        dtype=E6.dtype,
    )

    # Denominator per event (avoid divide-by-zero)
    S = E6.sum(axis=(1, 2, 3, 4, 5, 6)) + eps  # [B]

    # ---------- Centroids in (z, a, r) space ----------
    z_bar = (E6 * Z6).sum(axis=(1, 2, 3, 4, 5, 6)) / S  # [B]
    a_bar = (E6 * A6).sum(axis=(1, 2, 3, 4, 5, 6)) / S  # [B]
    r_bar = (E6 * R6).sum(axis=(1, 2, 3, 4, 5, 6)) / S  # [B]

    # ---------- Variances in (z, a, r) space (diagnostics) ----------
    z_var = (E6 * (Z6 - z_bar.reshape(B, 1, 1, 1, 1, 1, 1))**2).sum(axis=(1, 2, 3, 4, 5, 6)) / S
    a_var = (E6 * (A6 - a_bar.reshape(B, 1, 1, 1, 1, 1, 1))**2).sum(axis=(1, 2, 3, 4, 5, 6)) / S
    r_var = (E6 * (R6 - r_bar.reshape(B, 1, 1, 1, 1, 1, 1))**2).sum(axis=(1, 2, 3, 4, 5, 6)) / S

    rms_z = np.sqrt(np.maximum(z_var, 0.0)).astype(out_dtype, copy=False)
    rms_a = np.sqrt(np.maximum(a_var, 0.0)).astype(out_dtype, copy=False)  # radians if a_scale is rad
    rms_r = np.sqrt(np.maximum(r_var, 0.0)).astype(out_dtype, copy=False)  # mm if r_scale is mm

    # ---------- Cartesian (x,y) for physically consistent lateral RMS ----------
    # x = r cos(a), y = r sin(a)  (units of mm if r is mm and a is radians)
    X6 = R6 * np.cos(A6)
    Y6 = R6 * np.sin(A6)

    x_bar = (E6 * X6).sum(axis=(1, 2, 3, 4, 5, 6)) / S  # [B]
    y_bar = (E6 * Y6).sum(axis=(1, 2, 3, 4, 5, 6)) / S  # [B]

    x_var = (E6 * (X6 - x_bar.reshape(B, 1, 1, 1, 1, 1, 1))**2).sum(axis=(1, 2, 3, 4, 5, 6)) / S
    y_var = (E6 * (Y6 - y_bar.reshape(B, 1, 1, 1, 1, 1, 1))**2).sum(axis=(1, 2, 3, 4, 5, 6)) / S

    rms_x = np.sqrt(np.maximum(x_var, 0.0)).astype(out_dtype, copy=False)  # mm
    rms_y = np.sqrt(np.maximum(y_var, 0.0)).astype(out_dtype, copy=False)  # mm

    # Physically meaningful lateral RMS (in mm)
    lateral_rms = np.sqrt(rms_x**2 + rms_y**2).astype(out_dtype, copy=False)

    return (
        z_bar.astype(out_dtype, copy=False),
        a_bar.astype(out_dtype, copy=False),
        r_bar.astype(out_dtype, copy=False),
        rms_z,
        rms_a,
        rms_r,
        lateral_rms,
    )


'''
# --------- main entry point ---------
def build_eval_features(
    results_h5: str = "/teo_fs_fast/users/acosso/Dataset/eval_results_A.hdf5",
    truth_h5: Optional[str] = "/teo_fs_fast/users/acosso/Dataset/eval_dataset_A.hdf5",
    out_h5: str = "/teo_fs_fast/users/acosso/Dataset/eval_features_A.hdf5",
    *,
    io_chunk_events: int = 100_000,
    compression: Optional[str] = "lzf",
    compute_dtype = np.float64,   # accumulate in float64 for safety
    store_dtype = np.float32,     # save compact float32
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Stream over eval_results (energies) and compute compact features:
      - /per_event/sum_E_true, sum_E_gen          (shape [N])
      - /per_event/layer_E_true, layer_E_gen      (shape [N, nz])

    Geometry (nz,na,nr) is taken from results attrs if present,
    otherwise from truth['/coarse_showers'].shape. Fine split (zf,af,rf)
    is taken from truth['/factors'] as written by Preprocess2.

    The function writes a small 'eval_features' HDF5 you can extend later.
    """
    if Path(out_h5).exists() and not overwrite:
        print(f"{out_h5} exists; pass overwrite=True to replace.")
        #raise FileExistsError(f"{out_h5} exists; pass overwrite=True to replace.")

    with h5py.File(results_h5, "r") as hf_res:

        # Must exist (written by evaluate_flow_and_cache)
        if "/truth_E" not in hf_res or "/model_E" not in hf_res:
            raise KeyError(
                f"{results_h5} must contain '/truth_E' and '/model_E'. "
                "Did evaluate_flow_and_cache finish correctly?"
            )
        truth_E_ds = hf_res["/truth_E"]   # [N, Df]
        model_E_ds = hf_res["/model_E"]   # [N, Df]
        N, Df = truth_E_ds.shape
        if model_E_ds.shape != (N, Df):
            raise ValueError("truth_E and model_E shapes differ.")

        # Open truth cache only if provided/exists (for geometry/factors)
        hf_truth = h5py.File(truth_h5, "r") if (truth_h5 and Path(truth_h5).exists()) else None
        try:
            nz, na, nr, (zf, af, rf) = _read_geometry_and_factors(hf_res, hf_truth)

            # Create output
            with h5py.File(out_h5, "w") as dout:
                # provenance
                dout.attrs["source_results"] = Path(results_h5).name
                if hf_truth is not None:
                    dout.attrs["source_truth"] = Path(truth_h5).name
                dout.attrs["N_events"] = int(N)
                dout.attrs["Df"] = int(Df)
                dout.attrs["nz"] = int(nz)
                dout.attrs["na"] = int(na)
                dout.attrs["nr"] = int(nr)
                dout.attrs["zf"] = int(zf)
                dout.attrs["af"] = int(af)
                dout.attrs["rf"] = int(rf)
                dout.attrs["compute_dtype"] = str(np.dtype(compute_dtype))
                dout.attrs["store_dtype"] = str(np.dtype(store_dtype))

                # datasets
                ds_sum_true = dout.create_dataset(
                    "/per_event/sum_E_true", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_sum_gen = dout.create_dataset(
                    "/per_event/sum_E_gen", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_lay_true = dout.create_dataset(
                    "/per_event/layer_E_true", shape=(N, nz), maxshape=(None, nz),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_lay_gen = dout.create_dataset(
                    "/per_event/layer_E_gen", shape=(N, nz), maxshape=(None, nz),
                    dtype=store_dtype, compression=compression, chunks=True
                )

                # --- centroids (z,a,r) for truth & gen ---
                ds_centroid_true = dout.create_dataset(
                    "/per_event/centroid_true", shape=(N, 3), maxshape=(None, 3),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_centroid_gen = dout.create_dataset(
                    "/per_event/centroid_gen", shape=(N, 3), maxshape=(None, 3),
                    dtype=store_dtype, compression=compression, chunks=True
                )

                #--- RMS (longitudinal=z, lateral) for truth & gen ---
                # Store per-axis RMS too (useful for diagnostics)
                ds_rms_long_true = dout.create_dataset(
                    "/per_event/rms_longitudinal_true", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_rms_long_gen = dout.create_dataset(
                    "/per_event/rms_longitudinal_gen", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_rms_lat_true = dout.create_dataset(
                    "/per_event/rms_lateral_true", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_rms_lat_gen = dout.create_dataset(
                    "/per_event/rms_lateral_gen", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )

                # Optional (nice to have): per-axis RMS for r and a separately
                ds_rms_r_true = dout.create_dataset(
                    "/per_event/rms_r_true", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_rms_r_gen = dout.create_dataset(
                    "/per_event/rms_r_gen", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_rms_a_true = dout.create_dataset(
                    "/per_event/rms_a_true", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )
                ds_rms_a_gen = dout.create_dataset(
                    "/per_event/rms_a_gen", shape=(N,), maxshape=(None,),
                    dtype=store_dtype, compression=compression, chunks=True
                )


                # stream in chunks
                pos = 0
                while pos < N:
                    i0 = pos
                    i1 = min(N, pos + io_chunk_events)
                    B = i1 - i0

                    # load as float64 for numerically stable sums; no copy if already that dtype
                    E_true = truth_E_ds[i0:i1].astype(compute_dtype, copy=False)
                    E_gen  = model_E_ds[i0:i1].astype(compute_dtype,  copy=False)

                    # totals
                    sum_true = _compute_total_energy(E_true)   # [B]
                    sum_gen  = _compute_total_energy(E_gen)    # [B]

                    # layer sums: reshape back to [nz,na,nr,zf,af,rf] and sum over inner dims
                    lay_true = _compute_layer_energies(E_true, nz, na, nr, zf, af, rf)  # [B, nz]
                    lay_gen  = _compute_layer_energies(E_gen,  nz, na, nr, zf, af, rf)  # [B, nz]

                    # ---- NEW: centroids & RMS (index-space by default) ----
                    (zbar_t, abar_t, rbar_t,
                        rmsz_t, rmsa_t, rmsr_t, lat_t) = _compute_centroid_and_rms(E_true, nz, na, nr, zf, af, rf,
                                                                                   z_offset = 1.7, z_scale = 3.4,
                                                                                   a_offset = 0.1965, a_scale = 0.393,
                                                                                   r_offset = 2.325, r_scale = 4.65,
                                                                                    # If you later have physical spacings, pass z_scale/a_scale/r_scale here
                                                                                    out_dtype=compute_dtype)

                    (zbar_g, abar_g, rbar_g,
                        rmsz_g, rmsa_g, rmsr_g, lat_g) = _compute_centroid_and_rms(E_gen, nz, na, nr, zf, af, rf,
                                                                                   z_offset = 1.7, z_scale = 3.4,
                                                                                   a_offset = math.pi/16, a_scale = 2*math.pi/16,
                                                                                   r_offset = 2.325, r_scale = 4.65,
                                                                                    out_dtype=compute_dtype)

                    # write (cast to store dtype)
                    ds_sum_true[i0:i1] = sum_true.astype(store_dtype, copy=False)
                    ds_sum_gen[i0:i1]  = sum_gen.astype(store_dtype,  copy=False)
                    ds_lay_true[i0:i1] = lay_true.astype(store_dtype, copy=False)
                    ds_lay_gen[i0:i1]  = lay_gen.astype(store_dtype,  copy=False)
                    # Write (cast to store dtype)
                    ds_centroid_true[i0:i1, :] = np.stack([rbar_t, abar_t, zbar_t], axis=1).astype(store_dtype, copy=False)
                    ds_centroid_gen[i0:i1, :]  = np.stack([rbar_g, abar_g, zbar_g], axis=1).astype(store_dtype, copy=False)

                    ds_rms_long_true[i0:i1] = rmsz_t.astype(store_dtype, copy=False)
                    ds_rms_long_gen[i0:i1]  = rmsz_g.astype(store_dtype, copy=False)
                    ds_rms_lat_true[i0:i1]  = lat_t.astype(store_dtype, copy=False)
                    ds_rms_lat_gen[i0:i1]   = lat_g.astype(store_dtype, copy=False)

                    # Optional: per-axis RMS for diagnostics
                    ds_rms_r_true[i0:i1] = rmsr_t.astype(store_dtype, copy=False)
                    ds_rms_r_gen[i0:i1]  = rmsr_g.astype(store_dtype, copy=False)
                    ds_rms_a_true[i0:i1] = rmsa_t.astype(store_dtype, copy=False)
                    ds_rms_a_gen[i0:i1]  = rmsa_g.astype(store_dtype, copy=False)
                    dout.flush()

                    # free
                    del E_true, E_gen, sum_true, sum_gen, lay_true, lay_gen, rmsz_t, rmsz_g, lat_t, lat_g, rmsr_t, rmsr_g, rmsa_t, rmsa_g
                    gc.collect()

                    pos = i1
        finally:
            if hf_truth is not None:
                hf_truth.close()
    return {"N_events": int(N), "Df": int(Df), "nz": int(nz), "na": int(na), "nr": int(nr), "zf": int(zf), "af": int(af), "rf": int(rf)}
'''



def build_eval_features(
    results_h5: str = "/teo_fs_fast/users/acosso/Dataset/eval_results_A.hdf5",
    truth_h5: Optional[str] = "/teo_fs_fast/users/acosso/Dataset/eval_dataset_A.hdf5",
    out_h5: str = "/teo_fs_fast/users/acosso/Dataset/eval_features_A.hdf5",
    *,
    io_chunk_events: int = 100_000,
    compression: Optional[str] = "lzf",
    compute_dtype = np.float64,   # accumulate in float64 for safety
    store_dtype = np.float32,     # save compact float32
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Stream over eval_results (energies) and compute compact features:
      - /per_event/sum_E_true, sum_E_gen          (shape [N])
      - /per_event/layer_E_true, layer_E_gen      (shape [N, nz])
      - /per_event/centroid_true|gen              (shape [N, 3]) -> (z̄, ā, r̄)
      - /per_event/rms_longitudinal_true|gen      (shape [N])    -> along z (mm)
      - /per_event/rms_lateral_true|gen           (shape [N])    -> sqrt(RMS_x^2+RMS_y^2) (mm)
      - /per_event/rms_a_true|gen, rms_r_true|gen (shape [N])    -> diagnostics

    Supports BOTH storage layouts for /truth_E and /model_E in results_h5:
      1) Per-voxel rows: [TOTAL, K] where K = zf*af*rf and TOTAL = N_events * V with V = nz*na*nr
      2) Per-event rows: [N_events, Df] where Df = V*K
    """

    t0 = time.time()
    # --- NEW: skip if file exists and overwrite=False
    if os.path.exists(out_h5) and not overwrite:
        print(f"[build_eval_features] Output file already exists: {out_h5}")
        print("  overwrite flag is False → skipping feature computation.")
        with h5py.File(out_h5, "r") as f:
            # copy attrs while file is open
            attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        # safely use attrs after the file is closed
        N  = int(attrs.get("N_events", -1))
        nz = int(attrs.get("nz", -1))
        zf = int(attrs.get("zf", -1))
        print(f"  Existing file has N_events={N}, nz={nz}, zf={zf}, fine layers={nz*zf}")

        return {
            "N_events": int(attrs.get("N_events", -1)),
            "Df":       int(attrs.get("Df", -1)),
            "nz":       int(attrs.get("nz", -1)),
            "na":       int(attrs.get("na", -1)),
            "nr":       int(attrs.get("nr", -1)),
            "zf":       int(attrs.get("zf", -1)),
            "af":       int(attrs.get("af", -1)),
            "rf":       int(attrs.get("rf", -1)),
            "storage_mode": attrs.get("storage_mode", "unknown"),
            "duration_sec": 0.0,
            "out_h5": out_h5,
            "status": "skipped_existing_file",
        }


    # Overwrite handling
    if overwrite and os.path.exists(out_h5):
        os.remove(out_h5)

    # ---- open files
    hf_truth_ctx = h5py.File(truth_h5, "r") if truth_h5 is not None else nullcontext()
    try:
        with h5py.File(results_h5, "r") as hf_res, \
             (hf_truth_ctx if truth_h5 is not None else nullcontext()) as hf_truth, \
             h5py.File(out_h5, "w") as dout:

            # sanity
            if "truth_E" not in hf_res or "model_E" not in hf_res:
                raise KeyError("results_h5 must contain datasets 'truth_E' and 'model_E'.")

            truth_E_ds = hf_res["truth_E"]
            model_E_ds = hf_res["model_E"]

            # geometry & factors (reads from results first, then falls back to truth)
            nz, na, nr, (zf, af, rf) = _read_geometry_and_factors(hf_res, hf_truth if truth_h5 is not None else None)
            V  = int(nz * na * nr)
            K  = int(zf * af * rf)

            # detect storage layout
            N_rows, Df_col = truth_E_ds.shape
            per_voxel_rows = (Df_col == K) and (N_rows % V == 0)
            if per_voxel_rows:
                N = N_rows // V
                Df = V * K
                storage_mode = "per_voxel_rows"
            else:
                N = N_rows
                Df = Df_col
                storage_mode = "per_event_rows"
                if Df != V * K:
                    raise ValueError(
                        f"Shape mismatch: Df={Df} but nz*na*nr*zf*af*rf={V*K} "
                        f"(nz={nz},na={na},nr={nr}, zf={zf},af={af},rf={rf})."
                    )

            # chunk helper: always returns [B_eff, Df]
            def _load_event_chunk(i_evt: int, B: int):
                """
                Return (E_true, E_gen) with shape [B_eff, Df].
                """
                if storage_mode == "per_event_rows":
                    i1 = min(i_evt + B, N)
                    E_true = truth_E_ds[i_evt:i1].astype(compute_dtype, copy=False)
                    E_gen  = model_E_ds[i_evt:i1].astype(compute_dtype, copy=False)
                    return E_true, E_gen

                # storage_mode == per_voxel_rows
                i0_rows = i_evt * V
                i1_rows = min((i_evt + B) * V, N_rows)
                B_eff = (i1_rows - i0_rows) // V
                if B_eff <= 0:
                    return np.empty((0, Df), dtype=compute_dtype), np.empty((0, Df), dtype=compute_dtype)

                block_true = truth_E_ds[i0_rows:i0_rows + B_eff * V].astype(compute_dtype, copy=False)  # [B_eff*V, K]
                block_gen  = model_E_ds[i0_rows:i0_rows + B_eff * V].astype(compute_dtype, copy=False)

                # [B_eff*V, K] -> [B_eff, nz, na, nr, zf, af, rf] -> [B_eff, V*K]
                E_true6 = block_true.reshape(B_eff, nz, na, nr, zf, af, rf)
                E_gen6  = block_gen .reshape(B_eff, nz, na, nr, zf, af, rf)
                E_true_evt = E_true6.reshape(B_eff, V * K)
                E_gen_evt  = E_gen6 .reshape(B_eff, V * K)
                return E_true_evt, E_gen_evt

            # ---- attrs / provenance
            dout.attrs["source_results"]  = os.path.abspath(results_h5)
            if truth_h5 is not None:
                dout.attrs["source_truth"] = os.path.abspath(truth_h5)
            dout.attrs["storage_mode"]  = storage_mode
            dout.attrs["nz"] = int(nz); dout.attrs["na"] = int(na); dout.attrs["nr"] = int(nr)
            dout.attrs["zf"] = int(zf); dout.attrs["af"] = int(af); dout.attrs["rf"] = int(rf)
            dout.attrs["V"]  = int(V);  dout.attrs["K"]  = int(K);  dout.attrs["Df"] = int(Df)
            dout.attrs["N_events"] = int(N)
            dout.attrs["compute_dtype"] = str(np.dtype(compute_dtype))
            dout.attrs["store_dtype"]   = str(np.dtype(store_dtype))
            dout.attrs["compression"]   = str(compression)

            # coordinate scales (mm / rad) as per your spec
            dz = 3.4;  z0 = 1.7
            da = 2*math.pi/16.0; a0 = da/2.0
            dr = 4.65; r0 = dr/2.0
            dout.attrs["z_offset_mm"] = float(z0); dout.attrs["z_scale_mm"] = float(dz)
            dout.attrs["a_offset_rad"] = float(a0); dout.attrs["a_scale_rad"] = float(da)
            dout.attrs["r_offset_mm"] = float(r0); dout.attrs["r_scale_mm"] = float(dr)
            dout.attrs["n_finelayers"] = int(nz * zf)


            # ---- create per-event datasets
            cB = max(1, min(io_chunk_events, N))
            ds_sum_true = dout.create_dataset(
                "/per_event/sum_E_true", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_sum_gen  = dout.create_dataset(
                "/per_event/sum_E_gen", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            '''
            ds_lay_true = dout.create_dataset(
                "/per_event/layer_E_true", shape=(N, nz), maxshape=(None, nz),
                dtype=store_dtype, compression=compression, chunks=(cB, nz)
            )
            ds_lay_gen  = dout.create_dataset(
                "/per_event/layer_E_gen", shape=(N, nz), maxshape=(None, nz),
                dtype=store_dtype, compression=compression, chunks=(cB, nz)
            )
            '''
            n_finelayers = nz * zf
            ds_lay_true = dout.create_dataset(
                "/per_event/layer_E_true", shape=(N, n_finelayers), maxshape=(None, n_finelayers),
                dtype=store_dtype, compression=compression, chunks=True
            )
            ds_lay_gen = dout.create_dataset(
                "/per_event/layer_E_gen", shape=(N, n_finelayers), maxshape=(None, n_finelayers),
                dtype=store_dtype, compression=compression, chunks=True
            )

            ds_centroid_true = dout.create_dataset(
                "/per_event/centroid_true", shape=(N, 3), maxshape=(None, 3),
                dtype=store_dtype, compression=compression, chunks=(cB, 3)
            )
            ds_centroid_gen  = dout.create_dataset(
                "/per_event/centroid_gen", shape=(N, 3), maxshape=(None, 3),
                dtype=store_dtype, compression=compression, chunks=(cB, 3)
            )
            ds_rms_long_true = dout.create_dataset(
                "/per_event/rms_longitudinal_true", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_long_gen  = dout.create_dataset(
                "/per_event/rms_longitudinal_gen", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_lat_true = dout.create_dataset(
                "/per_event/rms_lateral_true", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_lat_gen  = dout.create_dataset(
                "/per_event/rms_lateral_gen", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_a_true = dout.create_dataset(
                "/per_event/rms_a_true", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_a_gen  = dout.create_dataset(
                "/per_event/rms_a_gen", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_r_true = dout.create_dataset(
                "/per_event/rms_r_true", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )
            ds_rms_r_gen  = dout.create_dataset(
                "/per_event/rms_r_gen", shape=(N,), maxshape=(None,),
                dtype=store_dtype, compression=compression, chunks=(cB,)
            )

            # ---- stream computation
            pos = 0
            while pos < N:
                i0 = pos
                i1 = min(N, pos + io_chunk_events)
                B  = i1 - i0

                # load energies [B, Df] (assemble if per-voxel rows)
                E_true, E_gen = _load_event_chunk(i0, B)
                if E_true.shape[0] == 0:
                    break

                # totals
                sum_true = _compute_total_energy(E_true)   # [B]
                sum_gen  = _compute_total_energy(E_gen)    # [B]

                # per-layer energy [B, nz]
                lay_true = _compute_layer_energies(E_true, nz, na, nr, zf, af, rf)
                lay_gen  = _compute_layer_energies(E_gen,  nz, na, nr, zf, af, rf)

                # centroids & RMS (your 7-tuple)
                (zbar_t, abar_t, rbar_t,
                 rmsz_t, rmsa_t, rmsr_t, lat_t) = _compute_centroid_and_rms(
                    E_true, nz, na, nr, zf, af, rf,
                    z_offset=z0, z_scale=dz,
                    a_offset=a0, a_scale=da,
                    r_offset=r0, r_scale=dr,
                    eps=1e-12, out_dtype=compute_dtype
                )

                (zbar_g, abar_g, rbar_g,
                 rmsz_g, rmsa_g, rmsr_g, lat_g) = _compute_centroid_and_rms(
                    E_gen, nz, na, nr, zf, af, rf,
                    z_offset=z0, z_scale=dz,
                    a_offset=a0, a_scale=da,
                    r_offset=r0, r_scale=dr,
                    eps=1e-12, out_dtype=compute_dtype
                )

                # write chunk
                sl = slice(i0, i1)
                ds_sum_true[sl] = sum_true.astype(store_dtype, copy=False)
                ds_sum_gen [sl] = sum_gen .astype(store_dtype, copy=False)

                ds_lay_true[sl, :] = lay_true.astype(store_dtype, copy=False)
                ds_lay_gen [sl, :] = lay_gen .astype(store_dtype, copy=False)

                ds_centroid_true[sl, 0] = zbar_t.astype(store_dtype, copy=False)
                ds_centroid_true[sl, 1] = abar_t.astype(store_dtype, copy=False)
                ds_centroid_true[sl, 2] = rbar_t.astype(store_dtype, copy=False)
                ds_centroid_gen [sl, 0] = zbar_g.astype(store_dtype, copy=False)
                ds_centroid_gen [sl, 1] = abar_g.astype(store_dtype, copy=False)
                ds_centroid_gen [sl, 2] = rbar_g.astype(store_dtype, copy=False)

                ds_rms_long_true[sl] = rmsz_t.astype(store_dtype, copy=False)
                ds_rms_long_gen [sl] = rmsz_g.astype(store_dtype, copy=False)
                ds_rms_lat_true [sl] = lat_t.astype(store_dtype, copy=False)
                ds_rms_lat_gen  [sl] = lat_g.astype(store_dtype, copy=False)
                ds_rms_a_true   [sl] = rmsa_t.astype(store_dtype, copy=False)
                ds_rms_a_gen    [sl] = rmsa_g.astype(store_dtype, copy=False)
                ds_rms_r_true   [sl] = rmsr_t.astype(store_dtype, copy=False)
                ds_rms_r_gen    [sl] = rmsr_g.astype(store_dtype, copy=False)

                dout.flush()

                # free
                del E_true, E_gen, sum_true, sum_gen, lay_true, lay_gen
                del zbar_t, abar_t, rbar_t, zbar_g, abar_g, rbar_g
                del rmsz_t, rmsz_g, lat_t, lat_g, rmsr_t, rmsr_g, rmsa_t, rmsa_g
                gc.collect()

                pos = i1

    finally:
        if truth_h5 is not None and hasattr(hf_truth_ctx, "close"):
            try:
                hf_truth_ctx.close()
            except Exception:
                pass

    return {
        "N_events": int(N),
        "Df": int(Df),
        "nz": int(nz), "na": int(na), "nr": int(nr),
        "zf": int(zf), "af": int(af), "rf": int(rf),
        "storage_mode": storage_mode,
        "duration_sec": float(time.time() - t0),
        "out_h5": out_h5,
    }


