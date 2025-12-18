import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Union, Optional, Any
from pathlib import Path
import h5py
import pickle
import time
import inspect
import sys

# Add parent directory to path if GMetrics is in a sibling directory
if 'GMetrics' not in sys.modules:
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
import GMetrics



def compute_pvalues_table(
    metrics_config: dict,
    unique_key: str,
    observed: dict,        # observed[name]["alt_array"] in INTERNAL units (FGD×10 already applied)
    ndims: int,
    *,
    agg: str = "median",   # "median" | "mean"
    tail: str = "right",   # "right" | "left" | "two-sided"
    save_csv: str | None = None,
    print_table: bool = True,   # NEW: show a pandas table at the end
    verbose: bool = False,
):
    """
    For every metric in metrics_config[unique_key]:
      - load the NULL distribution from meta["null_file"]
      - bring NULL to INTERNAL scale (scale_func(ns_eff, ndims); FGD×10)
      - aggregate the observed replicates (median/mean)
      - compute an empirical p-value (right/left/two-sided)

    Returns (df, results_dict). If print_table is True, displays the table.
    """
    assert agg in {"median", "mean"}, "agg must be 'median' or 'mean'"
    assert tail in {"right", "left", "two-sided"}, "tail must be 'right' | 'left' | 'two-sided'"

    agg_fn = np.median if agg == "median" else np.mean

    rows, out = [], {}

    def _safe_seed(val, default=0):
        try: return int(val)
        except Exception: return int(default)

    def _probe_slices(kwargs: dict):
        for k in ("nslices", "num_slices", "num_directions", "n_projections"):
            if k in kwargs:
                try:
                    return int(kwargs[k])
                except Exception:
                    return kwargs[k]
        return None

    for name, meta in metrics_config[unique_key].items():
        key = name.lower()
        if key in {"lr", "likelihood_ratio", "likelihoodratio"}:
            if verbose: print(f"[pvals] skip {name} (requires likelihoods).")
            continue
        if name not in observed:
            if verbose: print(f"[pvals] WARNING: '{name}' not found in observed; skipping.")
            continue

        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        # minimal dummy inputs to load the stored null JSON
        try:
            dummy_inputs = GMetrics.TwoSampleTestInputs(
                dist_1_input=np.zeros((1, ndims), dtype=np.float32),
                dist_2_input=np.zeros((1, ndims), dtype=np.float32),
                niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
            )
        except NameError as e:
            raise RuntimeError("GMetrics is not imported/available in this scope.") from e

        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], dtype=float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        ns_eff_null   = nsamples_null / 2.0
        scale_func    = meta.get("scale_func", None)
        null_scale    = float(scale_func(ns_eff_null, ndims)) if callable(scale_func) else 1.0

        null_internal = null_raw * null_scale
        if key == "fgd":
            null_internal *= 10.0

        alt_internal = np.asarray(observed[name]["alt_array"], dtype=float)

        # reconcile scales if observed/meta carries a different batch_size_test
        obs_meta_cfg = observed.get(name, {}).get("meta", {}).get("test_config", {})
        if obs_meta_cfg and callable(scale_func):
            try:
                bs_alt = int(obs_meta_cfg.get("batch_size_test", nsamples_null))
                ns_eff_alt = bs_alt / 2.0
                alt_scale  = float(scale_func(ns_eff_alt, ndims))
                if not np.isclose(alt_scale, null_scale):
                    alt_internal = alt_internal * (null_scale / alt_scale)
            except Exception:
                if verbose:
                    print(f"[pvals] note: could not reconcile alt scale for {name}; assuming same scale.")

        # aggregate observed replicates -> single test statistic
        t_obs = float(agg_fn(alt_internal)) if alt_internal.size else np.nan

        n_null = len(null_internal)
        if n_null == 0 or not np.isfinite(t_obs):
            p_right = p_left = np.nan
        else:
            p_right = (np.sum(null_internal >= t_obs) + 1.0) / (n_null + 1.0)
            p_left  = (np.sum(null_internal <= t_obs) + 1.0) / (n_null + 1.0)

        if tail == "right":
            pval = p_right
        elif tail == "left":
            pval = p_left
        else:
            pval = 2.0 * min(p_right, p_left)
            pval = min(1.0, max(0.0, pval))

        # optional timing (best effort)
        time_sec = None
        try:
            rv = null_obj.Results[-1].result_value
            for k in ("time_sec", "elapsed_sec", "elapsed_time", "total_time_sec", "times_total"):
                if k in rv:
                    time_sec = rv[k]
                    break
        except Exception:
            pass

        nslices = _probe_slices(metric_kwargs)

        rows.append({
            "metric": name.upper(),
            "niter": niter_null,
            "batch_size": nsamples_null,
            "nslices": nslices,             # will be stringified to "-" if missing
            "t_obs": t_obs,
            "p_value_num": float(pval) if np.isfinite(pval) else np.nan,
            "time_sec": time_sec,
        })

        out[name] = {
            "t_obs": t_obs,
            "p_value": float(pval) if np.isfinite(pval) else np.nan,
            "niter": niter_null,
            "batch_size": nsamples_null,
            "nslices": nslices,
            "time_sec": time_sec,
        }

    # Build DataFrame (keep numeric p_value_num; format/bold via Styler)
    df = pd.DataFrame(rows)

    if not df.empty:
        # nslices: show "-" when missing/NaN
        def _fmt_nslices(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "-"
            return str(x)

        df["nslices"] = df["nslices"].apply(_fmt_nslices)

        # Arrange columns (p_value_num stays numeric; we'll display a bold formatted view)
        df = df[["metric", "niter", "batch_size", "nslices", "p_value_num", "t_obs", "time_sec"]]

    # Save CSV with numeric p_value_num
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)

    # Pretty printed pandas table (bold p-values) if requested
        # Pretty printed pandas table (bold p-values) if requested
    if print_table and not df.empty:
        try:
            from IPython.display import display
            import math

            display_df = df.copy()
            # user-facing view: keep numeric p_value_num but show as 'p_value'
            display_df["p_value"] = display_df["p_value_num"]
            display_df = display_df[["metric", "niter", "batch_size", "nslices",
                                     "p_value", "time_sec"]]

            # --- Safe formatters that handle None/NaN gracefully ---
            def fmt_p(x):
                try:
                    if x is None or (isinstance(x, float) and not np.isfinite(x)):
                        return "-"
                    return f"{float(x):.3g}"
                except Exception:
                    return "-"

            def fmt_t(x):
                try:
                    if x is None or (isinstance(x, float) and not np.isfinite(x)):
                        return "-"
                    return f"{float(x):.4g}"
                except Exception:
                    return "-"

            def fmt_time(x):
                try:
                    if x is None or (isinstance(x, float) and not np.isfinite(x)):
                        return "-"
                    return f"{float(x):.1f}"
                except Exception:
                    return "-"

            styler = (
                display_df.style
                .format({"p_value": fmt_p, "t_obs": fmt_t, "time_sec": fmt_time})
                .set_properties(subset=["p_value"], **{"font-weight": "bold"})
            )
            # hide the index (compat for different pandas versions)
            if hasattr(styler, "hide"):
                styler = styler.hide(axis="index")
            else:
                styler = styler.hide_index()

            display(styler)

        except Exception:
            # Terminal fallback (no HTML/CSS styling available)
            print(df.to_string(index=False))


    return df, out













def compute_observed_distributions(
    metrics_config: dict,
    unique_key: str,
    dist_1_num,                 # TRUTH bank [Ns, D], np.ndarray or tf.Tensor (numeric) OR NumpyDistribution
    dist_2_num,                 # MODEL  bank [Ns, D], np.ndarray or tf.Tensor (numeric) OR NumpyDistribution
    ndims: int,
    batch_size_test: int,
    niter_obs: int = 100,
    dtype=tf.float64,
    use_tf: bool = False,       # numeric banks => default False; NumPy distributions => keep False
    verbose: bool = False,
    *,
    cache_file: str | None = None,  # optional: load/save a pickle cache
    overwrite: bool = False,
    show_progress: bool = True,     # external tqdm over metrics
):
    """
    Builds observed (alternative) distributions T_obs under H1 by repeatedly
    sampling truth vs model with n=m=batch_size_test, niter_obs times.

    Now supports NumPy-side symbolic distributions (e.g. EmpiricalNumpyDistribution),
    in addition to numeric banks. Logic is unchanged for numeric inputs.

    Caching:
      - If `cache_file` is provided and exists (and overwrite=False), the cached
        results are loaded and immediately returned (no computation).
      - Otherwise, results are computed as usual and then saved to `cache_file`
        (if provided).

    Returns:
      results[name] = {
         'alt_array': np.ndarray shape [niter_obs],   # scaled statistics
         'alt_mean': float,
         'alt_std':  float
      }
    """
    import os, pickle, time, inspect
    import numpy as np

    # --- helpers: detect numeric arrays/tensors vs NumPy distributions ---
    def _is_numeric_bank(x):
        try:
            import tensorflow as _tf
            return isinstance(x, (np.ndarray, _tf.Tensor))
        except Exception:
            return isinstance(x, np.ndarray)

    def _is_numpy_distribution(x):
        # Prefer a strong isinstance check if NumpyDistribution is importable;
        # otherwise fall back to a light duck-typing check.
        try:
            from Thesis.evaluation.utils_eval import NumpyDistribution as _ND  # same module that defines EmpiricalNumpyDistribution's base
            return isinstance(x, _ND)
        except Exception:
            return (hasattr(x, "sample") and callable(getattr(x, "sample"))
                    and not hasattr(x, "shape"))  # distributions don't expose .shape like arrays/tensors

    # ---------- Fast path: load from cache if available ----------
    if cache_file and (not overwrite) and os.path.isfile(cache_file):
        if verbose or show_progress:
            print(f"[compute_observed_distributions] Loaded cached results from: {cache_file}")
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        return payload.get("results", {})

    # ---------- Sanity checks (only for fully numeric banks) ----------
    both_numeric = _is_numeric_bank(dist_1_num) and _is_numeric_bank(dist_2_num)
    if both_numeric:
        N1, N2 = int(dist_1_num.shape[0]), int(dist_2_num.shape[0])
        assert N1 >= batch_size_test and N2 >= batch_size_test, (
            f"batch_size_test={batch_size_test} exceeds available rows: truth={N1}, model={N2}."
        )
        if niter_obs * batch_size_test > min(N1, N2) and verbose:
            print("[compute_observed_distributions] Note: iterations will reuse rows across batches "
                  f"(niter×batch={niter_obs*batch_size_test} > min(Ns)={min(N1,N2)}).")
    else:
        # If one or both inputs are NumPy distributions, we cannot pre-check sizes.
        # Sampling feasibility and any adjustments are handled internally by GMetrics.
        if verbose:
            kinds = (
                "dist_1: " + ("numeric" if _is_numeric_bank(dist_1_num) else
                              "numpy-distribution" if _is_numpy_distribution(dist_1_num) else "unknown"),
                "dist_2: " + ("numeric" if _is_numeric_bank(dist_2_num) else
                              "numpy-distribution" if _is_numpy_distribution(dist_2_num) else "unknown"),
            )
            print("[compute_observed_distributions] Using generalized inputs ->", ", ".join(kinds))

    # ---------- Build observed inputs ----------
    # Pass through exactly what we received:
    # - numeric arrays/tensors are treated as numeric banks
    # - NumPy distributions (incl. EmpiricalNumpyDistribution) are treated as symbolic on the NumPy path
    ObsInputs = GMetrics.TwoSampleTestInputs(
        dist_1_input    = dist_1_num,
        dist_2_input    = dist_2_num,
        niter           = niter_obs,
        batch_size_test = batch_size_test,
        small_sample_threshold = 1,   # keep your original setting; you said you'll lower it elsewhere if needed
        dtype_input     = dtype,
        seed_input      = 43,         # banks determine pairing; no RNG coupling
        use_tf          = use_tf,     # remains False for NumPy path, including NumPy distributions
        mirror_strategy = False,
        verbose         = verbose,
    )

    # ---------- Progress iterator over metrics ----------
    try:
        from tqdm.auto import tqdm
        metric_items = list(metrics_config[unique_key].items())
        iter_metrics = tqdm(metric_items, disable=not show_progress,
                            desc=f"T_obs (niter={niter_obs}, batch={batch_size_test})")
    except Exception:
        iter_metrics = list(metrics_config[unique_key].items())

    results = {}

    # ---------- Loop metrics ----------
    for name, meta in iter_metrics:
        lname = name.lower()
        if lname in {"lr", "likelihood_ratio", "likelihoodratio"}:
            if show_progress and hasattr(iter_metrics, "write"):
                iter_metrics.write(f("[skip] {name} requires likelihoods; skipping in observed pass."))
            continue

        # Instantiate metric CLASS (keep your original logic)
        MetricClass   = eval(meta["class_name"])         # e.g. GMetrics.SKSTest
        metric_kwargs = dict(meta.get("kwargs", {}))
        obj = MetricClass(
            ObsInputs,
            progress_bar = True,      # let GMetrics show its own bar if available
            verbose      = verbose,
            **metric_kwargs
        )

        # Choose backend
        use_tf_flag = getattr(obj.Inputs, "use_tf", False)
        test_fn = getattr(obj, "Test_tf", None) if use_tf_flag else getattr(obj, "Test_np", None)
        if test_fn is None:
            raise RuntimeError(f"{name}: no test function for use_tf={use_tf_flag}")

        # Optional max_vectorize
        call_kwargs = {}
        import inspect as _inspect
        sig = _inspect.signature(test_fn)
        max_vectorize = meta.get("max_vectorize", None)
        if ("max_vectorize" in sig.parameters) and (max_vectorize is not None):
            call_kwargs["max_vectorize"] = max_vectorize

        t0 = time.time()
        if show_progress and hasattr(iter_metrics, "set_postfix_str"):
            iter_metrics.set_postfix_str(f"running {name}…")

        # Run once; GMetrics fills obj.Results[-1] with niter_obs replicates
        test_fn(**call_kwargs)

        t1 = time.time()

        # Extract replicates and apply SAME scaling as for nulls
        result_key  = meta["result_key"]   # e.g. 'metric_means', 'statistic_means', 'metric_list'
        scale_func  = meta.get("scale_func", None)
        raw_arr     = np.array(obj.Results[-1].result_value[result_key])

        # Effective ns: keep your original convention (n/2 for two-sample tests)
        ns_eff      = batch_size_test / 2.0
        scale       = scale_func(ns_eff, ndims) if callable(scale_func) else 1.0
        arr_scaled  = raw_arr * scale

        if lname == "fgd":
            arr_scaled = arr_scaled * 10.0  # keep your plotting convention

        results[name] = {
            "alt_array": arr_scaled,
            "alt_mean":  float(arr_scaled.mean()),
            "alt_std":   float(arr_scaled.std(ddof=1)) if arr_scaled.size > 1 else 0.0,
        }

        if show_progress and hasattr(iter_metrics, "write"):
            iter_metrics.write(f"[done] {name} in {t1 - t0:.2f}s  (niter={niter_obs}, batch={batch_size_test})")

    # ---------- Save cache (optional) ----------
    if cache_file:
        payload = {
            "results": results,
            "meta": {
                "unique_key": unique_key,
                "ndims": ndims,
                "batch_size_test": batch_size_test,
                "niter_obs": niter_obs,
                "dtype": str(dtype),
                "use_tf": use_tf,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": list(metrics_config[unique_key].keys()),
                # Note: inputs could be numeric or symbolic; we don't serialize them here.
            },
        }
        try:
            d = os.path.dirname(cache_file)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose or show_progress:
                print(f"[compute_observed_distributions] Cached results saved to: {cache_file}")
        except Exception as e:
            if verbose or show_progress:
                print(f"[compute_observed_distributions] WARNING: failed to save cache: {e}")

    return results















def load_full_showers_coarse(
    results_h5: str,
    *,
    n_events: int = 100_000,
    ncoarse_per_event: Optional[int] = None,  # if None, auto-detect from file attrs (V)
    dtype: str | np.dtype = "float32",
    events_chunk: int = 1024,              # assemble this many events per streaming chunk
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble coarse calorimeter showers from row-wise (coarse, fine-slice) segments.
    
    Modified to build COARSE representation by summing fine voxels:
        - Loads fine representation: [N_rows, Df_per_coarse] where N_rows = total_events * ncoarse_per_event
        - Sums along axis 1 (fine voxels): [N_rows, Df_per_coarse] -> [N_rows, 1]
        - Reshapes to coarse: [N_rows, 1] -> [n_events, ncoarse_per_event]

        Expects datasets:
            /truth_E : [N_rows, Df_per_coarse]   where N_rows = total_events * ncoarse_per_event
            /model_E : [N_rows, Df_per_coarse]

        For Mode A, Df_per_coarse = 10 and ncoarse_per_event = 648 → coarse shower has 648 voxels.

    Returns
    -------
    input_true : (n_events, ncoarse_per_event) float32  # COARSE representation
    input_gen  : (n_events, ncoarse_per_event) float32  # COARSE representation
    """
    path = Path(results_h5)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with h5py.File(path, "r") as f:
        if "/truth_E" not in f or "/model_E" not in f:
            raise KeyError("Expected '/truth_E' and '/model_E' in results file.")

        ds_true = f["/truth_E"]   # [N_rows, Df_per_coarse]
        ds_gen  = f["/model_E"]   # [N_rows, Df_per_coarse]
        N_rows, Df_per = ds_true.shape
        if ds_gen.shape != (N_rows, Df_per):
            raise ValueError("Shape mismatch between '/truth_E' and '/model_E'.")

        # --- Robust auto-detection of ncoarse_per_event ---
        if ncoarse_per_event is None:
            attr_V = f.attrs.get("V", None)  # should be nz*na*nr = 648 for Mode A
            candidate_from_rows = None
            if N_rows % n_events == 0:  # user requested n_events (default 100k)
                candidate_from_rows = N_rows // n_events
            # Build list of plausible candidates
            candidates = []
            if attr_V is not None:
                candidates.append(int(attr_V))
            if candidate_from_rows is not None:
                candidates.append(int(candidate_from_rows))
            # Common mistaken stored value: attr_V * Df_per (e.g. 648*10=6480)
            if attr_V is not None and attr_V % Df_per == 0:
                wrong_mult = int(attr_V * Df_per)
                if N_rows % wrong_mult == 0:  # treat as potential mis-specified
                    candidates.append(int(attr_V))  # keep correct one
            # If attribute absent, try to infer by factoring N_rows.
            if not candidates:
                # Heuristic: try typical geometry 648 first
                if N_rows % 648 == 0:
                    candidates.append(648)
                # fallback: scan small divisors up to 2000 and pick one near 648 if exists
                for d in (576, 600, 640, 648, 672, 720):
                    if N_rows % d == 0:
                        candidates.append(d)
                if not candidates:
                    raise ValueError(
                        f"Cannot infer ncoarse_per_event: no attrs 'V', and no typical divisor fits N_rows={N_rows}."
                    )
            # Choose candidate closest to 648 (expected) if multiple
            ncoarse_per_event = min(candidates, key=lambda c: abs(c - 648))
            if verbose:
                print(f"[load_full_showers] Inferred ncoarse_per_event={ncoarse_per_event} (candidates={candidates}).")
        # Detect common error: using 10x the correct value (6480)
        if ncoarse_per_event >= 5000 and ncoarse_per_event % 10 == 0:
            suspected = ncoarse_per_event // 10
            if N_rows % suspected == 0 and suspected in (648, 576, 640, 672, 720):
                if verbose:
                    print(f"[load_full_showers] Warning: ncoarse_per_event={ncoarse_per_event} looks like 10x too large; correcting to {suspected}.")
                ncoarse_per_event = suspected

        # Total events available in the file (must be an integer multiple of ncoarse_per_event).
        if N_rows % ncoarse_per_event != 0:
            raise ValueError(
                f"N_rows={N_rows} not divisible by inferred ncoarse_per_event={ncoarse_per_event}. Df_per={Df_per}."
            )
        total_events = N_rows // ncoarse_per_event
        if verbose:
            print(f"[load_full_showers] total_events={total_events}, COARSE dimension={ncoarse_per_event} (summed from {ncoarse_per_event*Df_per} fine voxels)")

        # Cap n_events to what's available
        n = min(n_events, total_events)
        if verbose:
            print(
                f"Assembling {n} / {total_events} events "
                f"(rows={N_rows}, ncoarse={ncoarse_per_event}, Df_per={Df_per} → COARSE D={ncoarse_per_event}) "
                f"from {path.name}"
            )

        # Output is COARSE representation: (n_events, ncoarse_per_event)
        out_true = np.empty((n, ncoarse_per_event), dtype=dtype)
        out_gen  = np.empty((n, ncoarse_per_event), dtype=dtype)

        # Stream in blocks of whole events
        ev = 0
        while ev < n:
            k = min(events_chunk, n - ev)  # events in this chunk
            # Corresponding row slice (each event contributes ncoarse_per_event rows)
            i0 = ev * ncoarse_per_event
            i1 = (ev + k) * ncoarse_per_event

            # Read contiguous rows for these k events
            block_true = ds_true[i0:i1, :]   # shape = (k * ncoarse_per_event, Df_per)
            block_gen  = ds_gen[i0:i1, :]

            # Sum along axis 1 to get coarse energies: (k * ncoarse_per_event, Df_per) -> (k * ncoarse_per_event,)
            coarse_true = block_true.sum(axis=1, dtype=dtype)  # shape = (k * ncoarse_per_event,)
            coarse_gen  = block_gen.sum(axis=1, dtype=dtype)   # shape = (k * ncoarse_per_event,)

            # Reshape to (k, ncoarse_per_event)
            coarse_true = coarse_true.reshape(k, ncoarse_per_event)
            coarse_gen  = coarse_gen.reshape(k, ncoarse_per_event)

            out_true[ev:ev+k, :] = coarse_true
            out_gen[ev:ev+k, :]  = coarse_gen

            ev += k
            if verbose:
                print(f"  -> assembled {ev}/{n} events (COARSE)")

    return out_true, out_gen















def load_full_showers(
    results_h5: str,
    *,
    n_events: int = 100_000,
    ncoarse_per_event: Optional[int] = None,  # if None, auto-detect from file attrs (V)
    dtype: str | np.dtype = "float32",
    events_chunk: int = 1024,              # assemble this many events per streaming chunk
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble full calorimeter showers from row-wise (coarse, fine-slice) segments.

        Expects datasets:
            /truth_E : [N_rows, Df_per_coarse]   where N_rows = total_events * ncoarse_per_event
            /model_E : [N_rows, Df_per_coarse]

        For Mode A, Df_per_coarse = 10 and ncoarse_per_event = 648 → full shower has 6480 voxels.

    Returns
    -------
    input_true : (n_events, ncoarse_per_event * Df_per_coarse) float32
    input_gen  : (n_events, ncoarse_per_event * Df_per_coarse) float32
    """
    path = Path(results_h5)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with h5py.File(path, "r") as f:
        if "/truth_E" not in f or "/model_E" not in f:
            raise KeyError("Expected '/truth_E' and '/model_E' in results file.")

        ds_true = f["/truth_E"]   # [N_rows, Df_per_coarse]
        ds_gen  = f["/model_E"]   # [N_rows, Df_per_coarse]
        N_rows, Df_per = ds_true.shape
        if ds_gen.shape != (N_rows, Df_per):
            raise ValueError("Shape mismatch between '/truth_E' and '/model_E'.")

        # --- Robust auto-detection of ncoarse_per_event ---
        if ncoarse_per_event is None:
            attr_V = f.attrs.get("V", None)  # should be nz*na*nr = 648 for Mode A
            candidate_from_rows = None
            if N_rows % n_events == 0:  # user requested n_events (default 100k)
                candidate_from_rows = N_rows // n_events
            # Build list of plausible candidates
            candidates = []
            if attr_V is not None:
                candidates.append(int(attr_V))
            if candidate_from_rows is not None:
                candidates.append(int(candidate_from_rows))
            # Common mistaken stored value: attr_V * Df_per (e.g. 648*10=6480)
            if attr_V is not None and attr_V % Df_per == 0:
                wrong_mult = int(attr_V * Df_per)
                if N_rows % wrong_mult == 0:  # treat as potential mis-specified
                    candidates.append(int(attr_V))  # keep correct one
            # If attribute absent, try to infer by factoring N_rows.
            if not candidates:
                # Heuristic: try typical geometry 648 first
                if N_rows % 648 == 0:
                    candidates.append(648)
                # fallback: scan small divisors up to 2000 and pick one near 648 if exists
                for d in (576, 600, 640, 648, 672, 720):
                    if N_rows % d == 0:
                        candidates.append(d)
                if not candidates:
                    raise ValueError(
                        f"Cannot infer ncoarse_per_event: no attrs 'V', and no typical divisor fits N_rows={N_rows}."
                    )
            # Choose candidate closest to 648 (expected) if multiple
            ncoarse_per_event = min(candidates, key=lambda c: abs(c - 648))
            if verbose:
                print(f"[load_full_showers] Inferred ncoarse_per_event={ncoarse_per_event} (candidates={candidates}).")
        # Detect common error: using 10x the correct value (6480)
        if ncoarse_per_event >= 5000 and ncoarse_per_event % 10 == 0:
            suspected = ncoarse_per_event // 10
            if N_rows % suspected == 0 and suspected in (648, 576, 640, 672, 720):
                if verbose:
                    print(f"[load_full_showers] Warning: ncoarse_per_event={ncoarse_per_event} looks like 10x too large; correcting to {suspected}.")
                ncoarse_per_event = suspected

        # Total events available in the file (must be an integer multiple of ncoarse_per_event).
        if N_rows % ncoarse_per_event != 0:
            raise ValueError(
                f"N_rows={N_rows} not divisible by inferred ncoarse_per_event={ncoarse_per_event}. Df_per={Df_per}."
            )
        total_events = N_rows // ncoarse_per_event
        if verbose:
            print(f"[load_full_showers] total_events={total_events}, full_dimension={ncoarse_per_event*Df_per}")

        # Cap n_events to what's available
        n = min(n_events, total_events)
        if verbose:
            print(
                f"Assembling {n} / {total_events} events "
                f"(rows={N_rows}, ncoarse={ncoarse_per_event}, Df_per={Df_per} → full D={ncoarse_per_event*Df_per}) "
                f"from {path.name}"
            )

        full_D = ncoarse_per_event * Df_per
        out_true = np.empty((n, full_D), dtype=dtype)
        out_gen  = np.empty((n, full_D), dtype=dtype)

        # Stream in blocks of whole events
        ev = 0
        while ev < n:
            k = min(events_chunk, n - ev)  # events in this chunk
            # Corresponding row slice (each event contributes ncoarse_per_event rows)
            i0 = ev * ncoarse_per_event
            i1 = (ev + k) * ncoarse_per_event

            # Read contiguous rows for these k events
            block_true = ds_true[i0:i1, :]   # shape = (k * ncoarse_per_event, Df_per)
            block_gen  = ds_gen[i0:i1, :]

            # Reshape to (k, ncoarse, Df_per) then flatten to (k, ncoarse*Df_per)
            block_true = block_true.reshape(k, ncoarse_per_event, Df_per).astype(dtype, copy=False)
            block_gen  = block_gen.reshape(k, ncoarse_per_event, Df_per).astype(dtype, copy=False)

            out_true[ev:ev+k, :] = block_true.reshape(k, full_D)
            out_gen[ev:ev+k, :]  = block_gen.reshape(k, full_D)

            ev += k
            if verbose:
                print(f"  -> assembled {ev}/{n} events")

    return out_true, out_gen






def load_layer_energies(
    out_h5="/teo_fs_fast/users/acosso/Dataset/eval_features_A.hdf5",
    N=None,                     # cap number of events to load; None = all
    to_dtype=np.float64         # promote to float64 for numerics if you want
):
    """
    Loads per-layer energy arrays from eval_features HDF5.

    Returns:
        layer_true: (N, nz) ndarray
        layer_gen:  (N, nz) ndarray
        meta: dict with keys: N_events, nz, na, nr, zf, af, rf, Df, source files
    """
    out_h5 = str(out_h5)
    if not Path(out_h5).exists():
        raise FileNotFoundError(out_h5)

    with h5py.File(out_h5, "r") as f:
        # Required datasets from make_eval_features.py
        if ("/per_event/layer_E_true" not in f) or ("/per_event/layer_E_gen" not in f):
            raise KeyError("Missing '/per_event/layer_E_true' or '/per_event/layer_E_gen' in eval_features file")

        lay_true_ds = f["/per_event/layer_E_true"]   # shape [N, nz]
        lay_gen_ds  = f["/per_event/layer_E_gen"]    # shape [N, nz]
        N_total, nz = lay_true_ds.shape
        if lay_gen_ds.shape != (N_total, nz):
            raise ValueError("layer_E_true and layer_E_gen shapes differ.")

        # Clamp N if requested
        N_use = N_total if (N is None) else min(int(N), int(N_total))

        # Load into RAM (N*45*4B ~ 18 MB if float32 for N=100k → safe; we promote if asked)
        layer_true = lay_true_ds[:N_use]
        layer_gen  = lay_gen_ds[:N_use]

        if to_dtype is not None:
            layer_true = layer_true.astype(to_dtype, copy=False)
            layer_gen  = layer_gen.astype(to_dtype,  copy=False)

        # Collect metadata (stored by make_eval_features)
        attrs = f.attrs
        meta = {
            "N_events": int(attrs.get("N_events", N_total)),
            "Df":       int(attrs.get("Df", -1)),
            "nz":       int(attrs.get("nz", nz)),
            "na":       int(attrs.get("na", -1)),
            "nr":       int(attrs.get("nr", -1)),
            "zf":       int(attrs.get("zf", -1)),
            "af":       int(attrs.get("af", -1)),
            "rf":       int(attrs.get("rf", -1)),
            "source_results": attrs.get("source_results", ""),
            "source_truth":   attrs.get("source_truth", ""),
            "loaded_N": N_use,
            "file": out_h5,
        }

    return layer_true, layer_gen, meta








import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import h5py

def _attr(f: h5py.File, key: str, default: Any = None) -> Any:
    """Safely read an HDF5 attribute and convert bytes/0-d arrays to Python types."""
    if key not in f.attrs:
        return default
    val = f.attrs[key]
    # Convert 0-d numpy arrays to scalars
    if isinstance(val, np.ndarray) and val.shape == ():
        val = val.item()
    # Decode bytes to str
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return val

def load_total_energies(
    features_h5: str,
    *,
    dtype: Optional[np.dtype] = None,     # e.g., np.float64; if None, keep on-disk dtype
    verify_attrs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load per-event total energies from a features file produced by `build_eval_features`.

    Returns
    -------
    sum_true : np.ndarray, shape (N, 1)
        Values from /per_event/sum_E_true.
    sum_gen  : np.ndarray, shape (N, 1)
        Values from /per_event/sum_E_gen.
    meta     : dict
        Basic metadata (ints where appropriate + string for storage_mode).
    """
    if not os.path.isfile(features_h5):
        raise FileNotFoundError(f"Features file not found: {features_h5}")

    with h5py.File(features_h5, "r") as f:
        # required datasets
        for name in ("/per_event/sum_E_true", "/per_event/sum_E_gen"):
            if name not in f:
                raise KeyError(f"Missing dataset '{name}' in {features_h5}")

        d_true = f["/per_event/sum_E_true"]
        d_gen  = f["/per_event/sum_E_gen"]

        # read into memory
        sum_true = d_true[...]
        sum_gen  = d_gen[...]

        # optional cast
        if dtype is not None:
            sum_true = sum_true.astype(dtype, copy=False)
            sum_gen  = sum_gen.astype(dtype, copy=False)

        # ensure (N, 1)
        if sum_true.ndim == 1:
            sum_true = sum_true.reshape(-1, 1)
        if sum_gen.ndim == 1:
            sum_gen = sum_gen.reshape(-1, 1)

        # collect attrs (correct typing)
        meta = {
            "N_events":     int(_attr(f, "N_events", -1)),
            "Df":           int(_attr(f, "Df", -1)),
            "nz":           int(_attr(f, "nz", -1)),
            "na":           int(_attr(f, "na", -1)),
            "nr":           int(_attr(f, "nr", -1)),
            "zf":           int(_attr(f, "zf", -1)),
            "af":           int(_attr(f, "af", -1)),
            "rf":           int(_attr(f, "rf", -1)),
            "storage_mode": str(_attr(f, "storage_mode", "unknown")),
        }

        if verify_attrs and meta["N_events"] > 0:
            n_file = meta["N_events"]
            n_loaded = sum_true.shape[0]
            if n_loaded != n_file:
                raise ValueError(
                    f"N_events attr ({n_file}) != loaded rows ({n_loaded}). "
                    f"File: {features_h5}"
                )

    return sum_true, sum_gen, meta





import os
from typing import Optional, Tuple
import numpy as np
import h5py

def load_centroids_raz(
    features_h5: str,
    *,
    dtype: Optional[np.dtype] = None,   # e.g. np.float64; if None, keep on-disk dtype
    verify_n: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-event centroids from a features file produced by `build_eval_features`.

    The file stores centroids as (z, a, r). This function reorders to (r, a, z).

    Returns
    -------
    cent_true_raz : np.ndarray, shape (N, 3)
    cent_gen_raz  : np.ndarray, shape (N, 3)
    """
    if not os.path.isfile(features_h5):
        raise FileNotFoundError(f"Features file not found: {features_h5}")

    def _ensure_n_by_3(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError(f"Cannot reshape 1D centroid array of size {arr.size} into (N,3).")
            arr = arr.reshape(-1, 3)
        elif arr.ndim == 2:
            if arr.shape[1] == 3:
                pass  # already (N,3)
            elif arr.shape[0] == 3:
                arr = arr.T  # (3,N) -> (N,3)
            else:
                raise ValueError(f"Centroid array has unexpected shape {arr.shape} (wanted (N,3)).")
        else:
            raise ValueError(f"Centroid array has unexpected ndim={arr.ndim}.")
        return arr

    with h5py.File(features_h5, "r") as f:
        # Required datasets (stored as (z, a, r))
        d_true = f["/per_event/centroid_true"]
        d_gen  = f["/per_event/centroid_gen"]

        ct = d_true[...]
        cg = d_gen[...]

        # Optional dtype cast
        if dtype is not None:
            ct = ct.astype(dtype, copy=False)
            cg = cg.astype(dtype, copy=False)

        # Ensure (N,3)
        ct = _ensure_n_by_3(ct)
        cg = _ensure_n_by_3(cg)

        # Reorder columns: (z, a, r) -> (r, a, z)
        order = (2, 1, 0)
        ct_raz = ct[:, order]
        cg_raz = cg[:, order]

        # Optional sanity check with N_events attribute
        if verify_n and "N_events" in f.attrs:
            try:
                n_attr = int(np.array(f.attrs["N_events"]).item())
                if ct_raz.shape[0] != n_attr:
                    raise ValueError(
                        f"N_events attr ({n_attr}) != loaded rows ({ct_raz.shape[0]})."
                    )
            except Exception:
                # If attribute is missing/oddly typed, just skip strict checking
                pass

    return ct_raz, cg_raz





import os
from typing import Optional, Tuple
import numpy as np
import h5py

def load_rms_long_lat_ang(
    features_h5: str,
    *,
    dtype: Optional[np.dtype] = None,   # e.g., np.float64; if None, keep on-disk dtype
    verify_n: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-event RMS values (longitudinal, lateral, angular) from a features file
    produced by `build_eval_features`.

    Returns
    -------
    rms_true : np.ndarray, shape (N, 3)
        Columns: (longitudinal, lateral, angular) == (rms_z, rms_lateral, rms_alpha).
    rms_gen  : np.ndarray, shape (N, 3)
        Same order as above.
    """
    if not os.path.isfile(features_h5):
        raise FileNotFoundError(f"Features file not found: {features_h5}")

    def _ensure_1d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and 1 in arr.shape:
            return arr.reshape(-1)
        raise ValueError(f"Expected 1D array (or Nx1), got shape {arr.shape}")

    with h5py.File(features_h5, "r") as f:
        required = [
            "/per_event/rms_longitudinal_true",
            "/per_event/rms_lateral_true",
            "/per_event/rms_a_true",
            "/per_event/rms_longitudinal_gen",
            "/per_event/rms_lateral_gen",
            "/per_event/rms_a_gen",
        ]
        missing = [name for name in required if name not in f]
        if missing:
            raise KeyError(f"Missing datasets in {features_h5}: {missing}")

        # Read true arrays
        long_t = _ensure_1d(f["/per_event/rms_longitudinal_true"][...])
        lat_t  = _ensure_1d(f["/per_event/rms_lateral_true"][...])
        ang_t  = _ensure_1d(f["/per_event/rms_a_true"][...])

        # Read gen arrays
        long_g = _ensure_1d(f["/per_event/rms_longitudinal_gen"][...])
        lat_g  = _ensure_1d(f["/per_event/rms_lateral_gen"][...])
        ang_g  = _ensure_1d(f["/per_event/rms_a_gen"][...])

        # Optional cast
        if dtype is not None:
            long_t = long_t.astype(dtype, copy=False)
            lat_t  = lat_t.astype(dtype,  copy=False)
            ang_t  = ang_t.astype(dtype,  copy=False)
            long_g = long_g.astype(dtype, copy=False)
            lat_g  = lat_g.astype(dtype,  copy=False)
            ang_g  = ang_g.astype(dtype,  copy=False)

        # Stack columns -> (N, 3) in order (longitudinal, lateral, angular)
        rms_true = np.column_stack([long_t, lat_t, ang_t])
        rms_gen  = np.column_stack([long_g, lat_g, ang_g])

        # Optional sanity check vs N_events attribute
        if verify_n and "N_events" in f.attrs:
            try:
                n_attr = int(np.array(f.attrs["N_events"]).item())
                if rms_true.shape[0] != n_attr:
                    raise ValueError(
                        f"N_events attr ({n_attr}) != loaded rows ({rms_true.shape[0]})."
                    )
            except Exception:
                pass

    return rms_true, rms_gen











