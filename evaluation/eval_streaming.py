# eval_streaming.py
from __future__ import annotations
import math
import gc
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf

# You have this in your codebase:
# from Sample import invert_transform_fine
# If your project layout differs, adjust the import accordingly.
import Sample  # assumes Sample.invert_transform_fine exists

def enable_gpu_memory_growth():
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

@tf.function(jit_compile=False, reduce_retracing=True)
def _flow_sample_logits(flow, base_dist, conds,
                        clip_latent=False, latent_clip_value=4.0) -> tf.Tensor:
    """
    Sample in the transformed/logit space given conditional inputs.
    Mirrors your per-voxel forward path. Returns [n, Df] float64.
    """
    n = tf.shape(conds)[0]
    z = base_dist.sample(sample_shape=n)  # [n, Df]
    if clip_latent:
        z = tf.clip_by_value(z, -latent_clip_value, +latent_clip_value)

    x = z
    # Apply bijectors in the same direction you used at sampling time.
    # Your snippet used reversed(flow.bijectors) with forward().
    for bij in reversed(flow.bijectors):
        # Detect conditionals like in your code
        cond_fn = getattr(bij, "_shift_and_log_scale_fn", None)
        is_conditional = getattr(cond_fn, "_conditional", False)
        x = bij.forward(x, conditional_input=conds) if is_conditional else bij.forward(x)

    # keep double precision like the rest of your pipeline
    return tf.ensure_shape(tf.cast(x, tf.float64), [None, None])

def _slice_coarse_flat(hf: h5py.File, i0: int, i1: int, V: int) -> np.ndarray:
    """
    Return coarse energies flattened for rows [i0:i1) without loading all coarse_showers.
    We grab only the events spanned by the requested range and then slice inside.
    """
    N, nz, na, nr = hf["coarse_showers"].shape
    evt_s = i0 // V
    evt_e = (i1 - 1) // V + 1  # exclusive
    # read the minimal block of events that covers [i0, i1)
    cs_block = hf["coarse_showers"][evt_s:evt_e]     # shape [(evt_e-evt_s), nz, na, nr]
    cs_flat  = cs_block.reshape(-1)                  # contiguous in event-major, z,a,r order
    offset   = i0 - evt_s * V
    return cs_flat[offset: offset + (i1 - i0)]       # shape [m]

def _maybe_create_out(
    out_h5: str,
    n_rows: int,
    Df: int,
    compression: str | None,
    dtype_save: np.dtype,
    attrs: dict,
):
    mode = "a" if Path(out_h5).exists() else "w"
    out = h5py.File(out_h5, mode)
    if "truth_E" not in out:
        out.create_dataset("truth_E", shape=(n_rows, Df), dtype=dtype_save,
                           chunks=(max(1, min(262144 // max(Df,1), n_rows)), Df),
                           compression=compression)
    if "model_E" not in out:
        out.create_dataset("model_E", shape=(n_rows, Df), dtype=dtype_save,
                           chunks=(max(1, min(262144 // max(Df,1), n_rows)), Df),
                           compression=compression)
    if "rows_done" not in out.attrs:
        out.attrs["rows_done"] = 0
    # Store a few useful attributes for provenance
    for k, v in attrs.items():
        out.attrs[k] = v
    return out

def evaluate_flow_and_cache(
    in_cache_h5: str,
    out_h5: str,
    flow,                      # your tfp.bijectors.Chain (MAF) with loaded weights
    base_dist,                 # the exact base distribution used at train time
    *,
    # GPU / compute control
    gpu_batch: int = 65536,    # rows hitting GPU at once
    io_chunk_rows: int = 2_000_000,   # rows read from disk and written back at a time
    # ranges
    start_row: int = 0,
    n_rows: int | None = None,       # None -> process all rows from start_row
    # numerics / dtype
    tf_dtype=tf.float64,
    save_as_float32: bool = True,
    # latent tricks
    clip_latent: bool = False,
    latent_clip_value: float = 4.0,
    # I/O
    compression: str | None = "lzf",
    seed: int | None = 42,
    overwrite: bool = False,
):
    """
    Stream evaluation: reads [fine_trans, cond_trans, coarse_showers] slice-by-slice,
    samples model logits, inverts to energies, and writes to out_h5.

    Output file contains:
      - truth_E: [n_rows, Df] energies (de-logitized truth)
      - model_E: [n_rows, Df] energies (from flow sampling)
      - attrs: shapes, TOTAL, V, etc., and 'rows_done' to support resume.
    """
    enable_gpu_memory_growth()
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    with h5py.File(in_cache_h5, "r") as hf:
        TOTAL, Df = hf["fine_trans"].shape
        C = hf["cond_trans"].shape[1]
        N, nz, na, nr = hf["coarse_showers"].shape
        V = nz * na * nr

        if n_rows is None:
            n_rows = TOTAL - start_row
        stop_row = min(TOTAL, start_row + n_rows)
        n_rows = stop_row - start_row
        if n_rows <= 0:
            raise ValueError("No rows selected (check start_row / n_rows).")

        # sanity on base_dist
        try:
            ev = int(base_dist.event_shape[0])
            if ev != Df:
                raise ValueError(f"base_dist.event_shape={ev} but Df={Df}")
        except Exception:
            pass

        # open/create out
        dtype_save = np.float32 if save_as_float32 else np.float64
        attrs = dict(
            Df=Df, C=C, N=N, nz=nz, na=na, nr=nr, V=V,
            start_row=start_row, stop_row=stop_row, TOTAL=TOTAL,
            in_cache=Path(in_cache_h5).name
        )
        out = _maybe_create_out(out_h5, n_rows, Df, compression, dtype_save, attrs)
        try:
            # resume support
            rows_done = int(out.attrs.get("rows_done", 0))
            if overwrite and rows_done > 0:
                rows_done = 0
                out.attrs["rows_done"] = 0
            pos = rows_done
            while pos < n_rows:
                # ---- choose I/O window ----
                i0 = start_row + pos
                i1 = min(start_row + pos + io_chunk_rows, stop_row)
                m = i1 - i0
                print(f"[I/O] rows {i0:,d}:{i1:,d}  (chunk m={m:,d}; {100*(i1-start_row)/n_rows:.1f}% of target)")

                # ---- load transformed TRUTH logits & CONDS ----
                # These are exactly aligned row-wise by your cache builder.
                fine_logits_true = hf["fine_trans"][i0:i1].astype(np.float64, copy=False)  # [m, Df]
                conds_chunk      = hf["cond_trans"][i0:i1].astype(np.float64, copy=False)  # [m, C]
                # Coarse energies aligned to rows [i0:i1)
                coarse_chunk     = _slice_coarse_flat(hf, i0, i1, V).astype(np.float64, copy=False)  # [m]

                # ---- invert TRUTH to energies on CPU ----
                with tf.device("/CPU:0"):
                    truth_E = Sample.invert_transform_fine(
                        tf.constant(fine_logits_true, dtype=tf_dtype),
                        tf.constant(coarse_chunk,     dtype=tf_dtype),
                        alpha=1e-6, dtype=tf_dtype
                    )
                    truth_E = tf.cast(truth_E, tf.float32 if save_as_float32 else tf.float64).numpy()

                # ---- MODEL sampling on GPU in sub-batches ----
                model_E_out = np.empty((m, Df), dtype=dtype_save)
                # do small slices to control GPU memory
                for j in range(0, m, gpu_batch):
                    jb = min(gpu_batch, m - j)
                    conds_j   = tf.constant(conds_chunk[j:j+jb], dtype=tf_dtype)
                    logits_j  = _flow_sample_logits(
                        flow=flow,
                        base_dist=base_dist,
                        conds=conds_j,
                        clip_latent=clip_latent,
                        latent_clip_value=latent_clip_value
                    )  # [jb, Df], float64

                    # invert logits -> energies (uses coarse energy per coarse voxel)
                    with tf.device("/CPU:0"):
                        model_E_j = Sample.invert_transform_fine(
                            logits_j,
                            tf.constant(coarse_chunk[j:j+jb], dtype=tf_dtype),
                            alpha=1e-6, dtype=tf_dtype
                        )
                        model_E_j = tf.cast(
                            model_E_j, tf.float32 if save_as_float32 else tf.float64
                        ).numpy()

                    model_E_out[j:j+jb] = model_E_j
                    # release sub-batch
                    del conds_j, logits_j, model_E_j
                    gc.collect()

                # ---- write this chunk ----
                out["truth_E"][pos:pos+m] = truth_E
                out["model_E"][pos:pos+m] = model_E_out
                pos += m
                out.attrs["rows_done"] = pos
                out.flush()

                # release large arrays
                del fine_logits_true, conds_chunk, coarse_chunk, truth_E, model_E_out
                gc.collect()

            print(f"[DONE] Wrote {pos:,d} rows to {out_h5}")
        finally:
            out.close()
