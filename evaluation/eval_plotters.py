
import numpy as np
import tensorflow as tf
import GMetrics
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches


'''
def plot_null_vs_alt_pdf(
    metrics_config: dict,
    unique_key: str,
    observed: dict,            # INTERNAL scale; FGD×10 already applied inside 'observed'
    ndims: int,
    save_dir: str,
    *,
    fill_null: bool = True,    # fill area under null to make it pop
    null_alpha: float = 0.25,
    num_bins: int = 25,

    # Legend + vertical padding
    ypad_frac: float = 0.04,           # add 4% headroom on y-axis (deterministic)
    show_span_legend: bool = True,     # show CL shaded patches in legend
    show_null_fill_legend: bool = True,# show null-fill patch in legend

    # Per-metric axis flags (linear/symlog + linthresh)
    # FGD
    fgd_x_scale: str = "linear", fgd_x_linthresh: float | None = None,
    fgd_y_scale: str = "linear", fgd_y_linthresh: float | None = None,
    # MMD
    mmd_x_scale: str = "linear", mmd_x_linthresh: float | None = None,
    mmd_y_scale: str = "linear", mmd_y_linthresh: float | None = None,
    # KS
    ks_x_scale:  str = "linear", ks_x_linthresh:  float | None = None,
    ks_y_scale:  str = "linear", ks_y_linthresh:  float | None = None,
    # SKS
    sks_x_scale: str = "linear", sks_x_linthresh: float | None = None,
    sks_y_scale: str = "linear", sks_y_linthresh: float | None = None,
    # SWD
    swd_x_scale: str = "linear", swd_x_linthresh: float | None = None,
    swd_y_scale: str = "linear", swd_y_linthresh: float | None = None,

    # === NEW: per-metric DISPLAY scaling factors (applied to plotted values) ===
    # Example defaults: FGD ×1e-7, MMD ×1e5, others ×1
    fgd_disp_scale: float = 1e-7,
    mmd_disp_scale: float = 1e5,
    ks_disp_scale:  float = 1.0,
    sks_disp_scale: float = 1.0,
    swd_disp_scale: float = 1.0,
    show_scale_in_label: bool = True,  # append "(×10^{k})" to xlabel if scale ≠ 1
):
    """
    PDF-only overlay of H0 (null) vs H1 (alt) with:
      • CL thresholds from null (internal units → display units)
      • Common binning, closed step lines, optional null fill
      • Per-metric axis scaling (linear/symlog) with linthresh on both axes
      • Deterministic Y padding using histogram max (no autoscale surprises)
      • Legend patches reflect actual opacities
      • NEW: user-controlled per-metric DISPLAY scaling (FGD, MMD, KS, SKS, SWD)
    """
    import os, numpy as np, matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(save_dir, exist_ok=True)

    def _safe_seed(val, default=0):
        try: s = int(val)
        except Exception: s = int(default)
        return max(0, min(s, 2**32 - 1))

    def _density_hist(x, bins, rng):
        counts, edges = np.histogram(x, bins=bins, range=rng, density=True)
        return counts, edges

    def _format_scaled_label(base_label: str, scale: float) -> str:
        if not show_scale_in_label or scale == 1.0:
            return base_label
        # prefer nice 10^k if scale is (close to) a power of ten
        k = int(np.round(np.log10(scale))) if scale > 0 else 0
        if np.isfinite(k) and np.isclose(scale, 10.0**k, rtol=1e-12, atol=0.0):
            return rf"{base_label}\,(\times 10^{{{k}}})"
        else:
            return rf"{base_label}\,(\times {scale:g})"

    per_metric_axes = {
        "fgd": dict(x_scale=fgd_x_scale, x_linthresh=fgd_x_linthresh,
                    y_scale=fgd_y_scale, y_linthresh=fgd_y_linthresh),
        "mmd": dict(x_scale=mmd_x_scale, x_linthresh=mmd_x_linthresh,
                    y_scale=mmd_y_scale, y_linthresh=mmd_y_linthresh),
        "ks":  dict(x_scale=ks_x_scale,  x_linthresh=ks_x_linthresh,
                    y_scale=ks_y_scale,  y_linthresh=ks_y_linthresh),
        "sks": dict(x_scale=sks_x_scale, x_linthresh=sks_x_linthresh,
                    y_scale=sks_y_scale, y_linthresh=sks_y_linthresh),
        "swd": dict(x_scale=swd_x_scale, x_linthresh=swd_x_linthresh,
                    y_scale=swd_y_scale, y_linthresh=swd_y_linthresh),
    }

    # NEW: map metric → display scale
    disp_scale_map = {
        "fgd": float(fgd_disp_scale),
        "mmd": float(mmd_disp_scale),
        "ks":  float(ks_disp_scale),
        "sks": float(sks_disp_scale),
        "swd": float(swd_disp_scale),
    }

    for name, meta in metrics_config[unique_key].items():
        if name.lower() in {"lr", "likelihood_ratio", "likelihoodratio"}:
            continue

        # ---- instantiate metric to access config
        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        dummy_inputs = GMetrics.TwoSampleTestInputs(
            dist_1_input=np.zeros((1, ndims), dtype=np.float32),
            dist_2_input=np.zeros((1, ndims), dtype=np.float32),
            niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
        )
        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        # ---- INTERNAL scaling
        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], dtype=float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        nd            = int(ndims)
        ns_eff        = nsamples_null / 2.0  # n=m

        scale_func = meta.get("scale_func", None)
        internal_scale = scale_func(ns_eff, nd) if callable(scale_func) else 1.0

        null_scaled_internal = null_raw * float(internal_scale)
        # FGD special internal factor to match 'observed' convention
        if name.lower() == "fgd":
            null_scaled_internal *= 10.0

        alt_scaled_internal = np.asarray(observed[name]["alt_array"], dtype=float)
        
        # FIX: Re-scale alternative to match null's scaling convention
        # Alt was cached with batch_size from 'observed', but may use different scale_func
        if "batch_size_test" in observed[name].get("meta", {}).get("test_config", {}):
            batch_alt = int(observed[name]["meta"]["test_config"]["batch_size_test"])
            ns_eff_alt = batch_alt / 2.0
            scale_alt = float(scale_func(ns_eff_alt, nd) if callable(scale_func) else 1.0)
            scale_null = float(internal_scale)
            
            # If scales differ, convert alt from its scale to null's scale
            if scale_alt != 0 and abs(scale_alt - scale_null) / max(abs(scale_alt), abs(scale_null)) > 1e-6:
                alt_scaled_internal = alt_scaled_internal * (scale_null / scale_alt)
                if name.lower() == "fgd":
                    alt_scaled_internal *= 10.0  # Apply FGD factor if not already applied

        # ---- DISPLAY scaling (applied to values and thresholds)
        key = name.lower()
        display_scale = disp_scale_map.get(key, 1.0)

        null_plot = null_scaled_internal * display_scale
        alt_plot  = alt_scaled_internal  * display_scale

        # ---- axis label (optionally annotate with multiplier)
        latex_label = meta["latex"]
        latex_label = _format_scaled_label(latex_label, display_scale)

        # ---- CL thresholds (right-tailed)
        cls_from_meta = [float(cl) for (cl, _, _) in meta.get("thresholds", [])] if meta.get("thresholds") else []
        cl_list = sorted(set(cls_from_meta if cls_from_meta else [0.68, 0.95, 0.99]))
        thr_internal = {int(cl*100): float(np.quantile(null_scaled_internal, cl)) for cl in cl_list}
        thresholds_disp = {cl: thr_internal[cl] * display_scale for cl in thr_internal.keys()}

        # ---- binning
        combined   = np.concatenate([null_plot, alt_plot])
        x_min_data = float(np.min(combined)) if combined.size else 0.0
        x_max_data = float(np.max(combined)) if combined.size else 1.0
        max_thr    = max(thresholds_disp.values()) if len(thresholds_disp) else x_max_data
        x_min      = x_min_data
        x_max      = max(x_max_data, max_thr)
        xpad       = 0.04 * (x_max - x_min) if np.isfinite(x_max - x_min) else 0.0
        x_left, x_right = x_min - xpad, x_max + xpad
        hist_range = (x_min, x_max)

        null_counts, bin_edges = _density_hist(null_plot, num_bins, hist_range)
        alt_counts,  _         = _density_hist(alt_plot,  num_bins, hist_range)

        # ---- plotting
        fig, ax = plt.subplots(figsize=(9.5, 6))
        ax.margins(x=0, y=0)

        # NULL: step + optional fill
        null_line = ax.step(bin_edges[:-1], null_counts, where='post', color="tomato",
                            linewidth=1.6, label="Null (truth–truth)")[0]
        null_style = dict(color=null_line.get_color(),
                          linewidth=null_line.get_linewidth(),
                          linestyle=null_line.get_linestyle(),
                          alpha=null_line.get_alpha())
        if fill_null and len(null_counts) > 0:
            x_stairs = np.repeat(bin_edges, 2)[1:-1]
            y_stairs = np.repeat(null_counts, 2)
            ax.fill_between(x_stairs, y_stairs, step='post',
                            alpha=null_alpha, color=null_line.get_color())

        # ALT: step-only
        alt_line = ax.step(bin_edges[:-1], alt_counts, where='post', color="slateblue",
                           linewidth=2.2, label="Alt (truth–model)")[0]
        alt_style = dict(color=alt_line.get_color(),
                         linewidth=alt_line.get_linewidth(),
                         linestyle=alt_line.get_linestyle(),
                         alpha=alt_line.get_alpha())

        # Bin-closure visuals
        if len(null_counts) > 0:
            ax.hlines(null_counts[0],  bin_edges[0],  bin_edges[1],  **null_style)
            ax.hlines(null_counts[-1], bin_edges[-2], bin_edges[-1], **null_style)
            ax.vlines(bin_edges[0],    0.0,           null_counts[0],  **null_style)
            ax.vlines(bin_edges[-1],   null_counts[-1], 0.0,           **null_style)
        if len(alt_counts) > 0:
            ax.hlines(alt_counts[0],   bin_edges[0],  bin_edges[1],  **alt_style)
            ax.hlines(alt_counts[-1],  bin_edges[-2], bin_edges[-1], **alt_style)
            ax.vlines(bin_edges[0],    0.0,           alt_counts[0],  **alt_style)
            ax.vlines(bin_edges[-1],   alt_counts[-1], 0.0,           **alt_style)

        # thresholds shading/markers
        dash_styles = ['-', '--', '-.']
        colors = ["royalblue", "darkorange", "mediumseagreen"]
        span_alpha = 0.08
        cl_patches = []
        for i, cl in enumerate(sorted(thresholds_disp.keys())):
            thr = thresholds_disp[cl]
            ax.axvspan(thr, x_right, color=colors[i % len(colors)], alpha=span_alpha)
            ax.axvline(thr, linestyle=dash_styles[i % len(dash_styles)],
                       color=colors[i % len(colors)], linewidth=1.2)
            if show_span_legend:
                cl_patches.append(
                    mpatches.Patch(facecolor=colors[i % len(colors)],
                                   edgecolor='none', alpha=span_alpha, label=f"{cl}% CL")
                )

        # X limits fixed early
        ax.set_xlim(x_left, x_right)

        # ---- Apply per-metric axis scaling
        params = per_metric_axes.get(key, dict(x_scale="linear", x_linthresh=None,
                                               y_scale="linear", y_linthresh=None))
        # X axis
        if params["x_scale"] == "linear":
            ax.set_xscale("linear")
        elif params["x_scale"] == "symlog":
            x_lt = params["x_linthresh"] if params["x_linthresh"] is not None else max(1e-12, 0.01 * (x_right - x_left))
            ax.set_xscale("symlog", linthresh=x_lt)
        else:
            raise ValueError(f"Unknown x scale '{params['x_scale']}' for metric '{key}'.")
        # Y axis (set scale only; limits are set deterministically below)
        if params["y_scale"] == "linear":
            ax.set_yscale("linear")
        elif params["y_scale"] == "symlog":
            tmp_ymax = max(float(np.nanmax(null_counts)) if len(null_counts) else 0.0,
                           float(np.nanmax(alt_counts))  if len(alt_counts)  else 0.0)
            y_lt = params["y_linthresh"] if params["y_linthresh"] is not None else max(1e-12, 0.01 * max(1.0, tmp_ymax))
            ax.set_yscale("symlog", linthresh=y_lt)
        else:
            raise ValueError(f"Unknown y scale '{params['y_scale']}' for metric '{key}'.")

        # ---- Legend with opacity-true patches
        handles = [null_line, alt_line]
        if fill_null and show_null_fill_legend:
            handles.append(mpatches.Patch(facecolor=null_line.get_color(), edgecolor='none',
                                          alpha=null_alpha, label="Null area"))
        if show_span_legend and cl_patches:
            handles.extend(cl_patches)

        #legend1 = ax[1].legend(loc='upper right', fontsize=15)
        #for handle in legend1.legend_handles[-3:]:
        #    handle.set_alpha(0.5)
        legend = ax.legend(handles=handles, loc='upper right', fontsize=13)
        for handle in legend.legend_handles[-3:]:
           handle.set_alpha(0.5)

        # ---- DETERMINISTIC Y-LIMITS WITH PADDING (no autoscale)
        ymax_null = float(np.nanmax(null_counts)) if len(null_counts) else 0.0
        ymax_alt  = float(np.nanmax(alt_counts))  if len(alt_counts)  else 0.0
        ymax = max(ymax_null, ymax_alt)
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0  # fallback to avoid singular ylim

        y_bottom = 0.0 if params["y_scale"] == "linear" else 1e-12
        y_top    = ymax * (1.0 + float(ypad_frac))

        # Freeze autoscale and apply limits LAST
        ax.autoscale(enable=False, axis='both')
        ax.set_ylim(y_bottom, y_top)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel(latex_label, fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(rf"{name.upper()} — $d={nd}$, $n=m={nsamples_null//1000}$K, $n_{{\mathrm{{iter}}}}={niter_null}$",
                     fontsize=15)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        plt.tight_layout()
        out_pdf = os.path.join(save_dir, f"{name.upper()}_null_vs_alt_PDF.pdf")
        plt.savefig(out_pdf)
        plt.show()
        plt.close()
        print(f"[{name.upper()}] saved {out_pdf}")
'''

def plot_null_vs_alt_pdf(
    metrics_config: dict,
    unique_key: str,
    observed: dict,            # INTERNAL scale; FGD×10 already applied inside 'observed'
    ndims: int,
    save_dir: str | None,      # <-- now optional; if None => don't save
    *,
    fill_null: bool = True,    # fill area under null to make it pop
    null_alpha: float = 0.25,
    num_bins: int = 25,

    # Legend + vertical padding
    ypad_frac: float = 0.04,           # add 4% headroom on y-axis (deterministic)
    show_span_legend: bool = True,     # show CL shaded patches in legend
    show_null_fill_legend: bool = True,# show null-fill patch in legend

    # Per-metric axis flags (linear/symlog + linthresh)
    # FGD
    fgd_x_scale: str = "linear", fgd_x_linthresh: float | None = None,
    fgd_y_scale: str = "linear", fgd_y_linthresh: float | None = None,
    # MMD
    mmd_x_scale: str = "linear", mmd_x_linthresh: float | None = None,
    mmd_y_scale: str = "linear", mmd_y_linthresh: float | None = None,
    # KS
    ks_x_scale:  str = "linear", ks_x_linthresh:  float | None = None,
    ks_y_scale:  str = "linear", ks_y_linthresh:  float | None = None,
    # SKS
    sks_x_scale: str = "linear", sks_x_linthresh: float | None = None,
    sks_y_scale: str = "linear", sks_y_linthresh: float | None = None,
    # SWD
    swd_x_scale: str = "linear", swd_x_linthresh: float | None = None,
    swd_y_scale: str = "linear", swd_y_linthresh: float | None = None,

    # per-metric DISPLAY scaling factors (applied to plotted values)
    fgd_disp_scale: float = 1e-7,
    mmd_disp_scale: float = 1e5,
    ks_disp_scale:  float = 1.0,
    sks_disp_scale: float = 1.0,
    swd_disp_scale: float = 1.0,
    show_scale_in_label: bool = True,
    batch_size_test: int = 1000, 

    # layout controls
    plotting_mode: str = "single",     # "single" or "multiple"
    expected_num_metrics: int | None = None,
    ncols_multiple: int = 3,           # fixed 3 per your request (rows of 3)
):
    """
    If save_dir is None or empty, figures are *not* saved and only shown.
    """

    import os, numpy as np, matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import math

    # Optional save control
    save_enabled = isinstance(save_dir, str) and bool(save_dir.strip())
    if save_enabled:
        os.makedirs(save_dir, exist_ok=True)

    # External deps (as in your previous version)
    try:
        import tensorflow as tf
    except Exception:
        pass
    try:
        import GMetrics
    except Exception:
        raise ImportError("GMetrics is required but not importable in this scope.")

    def _safe_seed(val, default=0):
        try: s = int(val)
        except Exception: s = int(default)
        return max(0, min(s, 2**32 - 1))

    def _density_hist(x, bins, rng):
        counts, edges = np.histogram(x, bins=bins, range=rng, density=True)
        return counts, edges

    def _format_scaled_label(base_label: str, scale: float) -> str:
        if not show_scale_in_label or scale == 1.0:
            return base_label
        k = int(np.round(np.log10(scale))) if scale > 0 else 0
        if np.isfinite(k) and np.isclose(scale, 10.0**k, rtol=1e-12, atol=0.0):
            return rf"{base_label}\,(\times 10^{{{k}}})"
        else:
            return rf"{base_label}\,(\times {scale:g})"

    per_metric_axes = {
        "fgd": dict(x_scale=fgd_x_scale, x_linthresh=fgd_x_linthresh,
                    y_scale=fgd_y_scale, y_linthresh=fgd_y_linthresh),
        "mmd": dict(x_scale=mmd_x_scale, x_linthresh=mmd_x_linthresh,
                    y_scale=mmd_y_scale, y_linthresh=mmd_y_linthresh),
        "ks":  dict(x_scale=ks_x_scale,  x_linthresh=ks_x_linthresh,
                    y_scale=ks_y_scale,  y_linthresh=ks_y_linthresh),
        "sks": dict(x_scale=sks_x_scale, x_linthresh=sks_x_linthresh,
                    y_scale=sks_y_scale, y_linthresh=sks_y_linthresh),
        "swd": dict(x_scale=swd_x_scale, x_linthresh=swd_x_linthresh,
                    y_scale=swd_y_scale, y_linthresh=swd_y_linthresh),
    }

    disp_scale_map = {
        "fgd": float(fgd_disp_scale),
        "mmd": float(mmd_disp_scale),
        "ks":  float(ks_disp_scale),
        "sks": float(sks_disp_scale),
        "swd": float(swd_disp_scale),
    }

    # Collect metrics to plot
    items_all = list(metrics_config[unique_key].items())
    metrics_to_plot = []
    for name, meta in items_all:
        if name.lower() in {"lr", "likelihood_ratio", "likelihoodratio"}:
            continue
        metrics_to_plot.append((name, meta))

    n_metrics_found = len(metrics_to_plot)
    if expected_num_metrics is not None:
        if n_metrics_found > expected_num_metrics:
            raise ValueError(
                f"Found {n_metrics_found} metrics to plot, which exceeds expected_num_metrics={expected_num_metrics}."
            )
        elif n_metrics_found < expected_num_metrics:
            print(f"[warn] Found {n_metrics_found} metrics, fewer than expected_num_metrics={expected_num_metrics}.")

    def _plot_one_metric(ax, name: str, meta: dict):
        key = name.lower()

        # Instantiate metric + load null
        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        dummy_inputs = GMetrics.TwoSampleTestInputs(
            dist_1_input=np.zeros((1, ndims), dtype=np.float32),
            dist_2_input=np.zeros((1, ndims), dtype=np.float32),
            niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
        )
        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], dtype=float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        nd            = int(ndims)
        ns_eff        = nsamples_null / 2.0  # n=m

        scale_func = meta.get("scale_func", None)
        internal_scale = scale_func(ns_eff, nd) if callable(scale_func) else 1.0

        null_scaled_internal = null_raw * float(internal_scale)
        if key == "fgd":
            null_scaled_internal *= 10.0  # match your 'observed' convention

        alt_scaled_internal = np.asarray(observed[name]["alt_array"], dtype=float)

        # Re-scale alternative to null's convention if needed
        if "batch_size_test" in observed[name].get("meta", {}).get("test_config", {}):
            batch_alt = int(observed[name]["meta"]["test_config"]["batch_size_test"])
            ns_eff_alt = batch_alt / 2.0
            scale_alt = float(scale_func(ns_eff_alt, nd) if callable(scale_func) else 1.0)
            scale_null = float(internal_scale)
            if scale_alt != 0 and abs(scale_alt - scale_null) / max(abs(scale_alt), abs(scale_null)) > 1e-6:
                alt_scaled_internal = alt_scaled_internal * (scale_null / scale_alt)
                if key == "fgd":
                    alt_scaled_internal *= 10.0

        # DISPLAY scaling
        display_scale = disp_scale_map.get(key, 1.0)
        null_plot = null_scaled_internal * display_scale
        alt_plot  = alt_scaled_internal  * display_scale

        # Labels
        latex_label = _format_scaled_label(meta["latex"], display_scale)

        # CL thresholds (right-tailed)
        cls_from_meta = [float(cl) for (cl, _, _) in meta.get("thresholds", [])] if meta.get("thresholds") else []
        cl_list = sorted(set(cls_from_meta if cls_from_meta else [0.68, 0.95, 0.99]))
        thr_internal = {int(cl*100): float(np.quantile(null_scaled_internal, cl)) for cl in cl_list}
        thresholds_disp = {cl: thr_internal[cl] * display_scale for cl in thr_internal.keys()}

        # Binning (common)
        combined   = np.concatenate([null_plot, alt_plot])
        x_min_data = float(np.min(combined)) if combined.size else 0.0
        x_max_data = float(np.max(combined)) if combined.size else 1.0
        max_thr    = max(thresholds_disp.values()) if len(thresholds_disp) else x_max_data
        x_min      = x_min_data
        x_max      = max(x_max_data, max_thr)
        xpad       = 0.04 * (x_max - x_min) if np.isfinite(x_max - x_min) else 0.0
        x_left, x_right = x_min - xpad, x_max + xpad
        hist_range = (x_min, x_max)

        null_counts, bin_edges = _density_hist(null_plot, num_bins, hist_range)
        alt_counts,  _         = _density_hist(alt_plot,  num_bins, hist_range)

        # Plot
        ax.margins(x=0, y=0)
        null_line = ax.step(bin_edges[:-1], null_counts, where='post', color="tomato",
                            linewidth=1.6, label="Null (truth–truth)")[0]
        if fill_null and len(null_counts) > 0:
            x_stairs = np.repeat(bin_edges, 2)[1:-1]
            y_stairs = np.repeat(null_counts, 2)
            ax.fill_between(x_stairs, y_stairs, step='post',
                            alpha=null_alpha, color=null_line.get_color())

        alt_line = ax.step(bin_edges[:-1], alt_counts, where='post', color="slateblue",
                           linewidth=2.2, label="Alt (truth–model)")[0]

        # Bin-closure visuals
        null_style = dict(color=null_line.get_color(), linewidth=null_line.get_linewidth(),
                          linestyle=null_line.get_linestyle(), alpha=null_line.get_alpha())
        alt_style  = dict(color=alt_line.get_color(),  linewidth=alt_line.get_linewidth(),
                          linestyle=alt_line.get_linestyle(),  alpha=alt_line.get_alpha())
        if len(null_counts) > 0:
            ax.hlines(null_counts[0],  bin_edges[0],  bin_edges[1],  **null_style)
            ax.hlines(null_counts[-1], bin_edges[-2], bin_edges[-1], **null_style)
            ax.vlines(bin_edges[0],    0.0,           null_counts[0],  **null_style)
            ax.vlines(bin_edges[-1],   null_counts[-1], 0.0,           **null_style)
        if len(alt_counts) > 0:
            ax.hlines(alt_counts[0],   bin_edges[0],  bin_edges[1],  **alt_style)
            ax.hlines(alt_counts[-1],  bin_edges[-2], bin_edges[-1], **alt_style)
            ax.vlines(bin_edges[0],    0.0,           alt_counts[0],  **alt_style)
            ax.vlines(bin_edges[-1],   alt_counts[-1], 0.0,           **alt_style)

        # Thresholds
        dash_styles = ['-', '--', '-.']
        colors = ["royalblue", "darkorange", "mediumseagreen"]
        span_alpha = 0.08
        cl_patches = []
        for i, cl in enumerate(sorted(thresholds_disp.keys())):
            thr = thresholds_disp[cl]
            ax.axvspan(thr, x_right, color=colors[i % len(colors)], alpha=span_alpha)
            ax.axvline(thr, linestyle=dash_styles[i % len(dash_styles)],
                       color=colors[i % len(colors)], linewidth=1.2)
            if show_span_legend:
                cl_patches.append(
                    mpatches.Patch(facecolor=colors[i % len(colors)],
                                   edgecolor='none', alpha=span_alpha, label=f"{cl}% CL")
                )

        # Limits & scales
        ax.set_xlim(x_left, x_right)
        params = per_metric_axes.get(key, dict(x_scale="linear", x_linthresh=None,
                                               y_scale="linear", y_linthresh=None))
        if params["x_scale"] == "linear":
            ax.set_xscale("linear")
        elif params["x_scale"] == "symlog":
            x_lt = params["x_linthresh"] if params["x_linthresh"] is not None else max(1e-12, 0.01 * (x_right - x_left))
            ax.set_xscale("symlog", linthresh=x_lt)
        else:
            raise ValueError(f"Unknown x scale '{params['x_scale']}' for metric '{key}'.")

        if params["y_scale"] == "linear":
            ax.set_yscale("linear")
        elif params["y_scale"] == "symlog":
            tmp_ymax = max(float(np.nanmax(null_counts)) if len(null_counts) else 0.0,
                           float(np.nanmax(alt_counts))  if len(alt_counts)  else 0.0)
            y_lt = params["y_linthresh"] if params["y_linthresh"] is not None else max(1e-12, 0.01 * max(1.0, tmp_ymax))
            ax.set_yscale("symlog", linthresh=y_lt)
        else:
            raise ValueError(f"Unknown y scale '{params['y_scale']}' for metric '{key}'.")

        # Legend
        handles = [null_line, alt_line]
        if fill_null and show_null_fill_legend:
            handles.append(mpatches.Patch(facecolor=null_line.get_color(), edgecolor='none',
                                          alpha=null_alpha, label="Null area"))
        if show_span_legend and cl_patches:
            handles.extend(cl_patches)
        #ax.legend(handles=handles, loc='upper right', fontsize=11)
        legend = ax.legend(handles=handles, loc='upper right', fontsize=15)
        for handle in legend.legend_handles[-3:]:
           handle.set_alpha(0.5)

        # Deterministic y-limits
        ymax_null = float(np.nanmax(null_counts)) if len(null_counts) else 0.0
        ymax_alt  = float(np.nanmax(alt_counts))  if len(alt_counts)  else 0.0
        ymax = max(ymax_null, ymax_alt) or 1.0
        y_bottom = 0.0 if params["y_scale"] == "linear" else 1e-12
        y_top    = ymax * (1.0 + float(ypad_frac))
        ax.autoscale(enable=False, axis='both')
        ax.set_ylim(y_bottom, y_top)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel(latex_label, fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        ax.set_title(rf"{name.upper()}",
                     fontsize=15)

        return niter_null

    niter_null = None
    # ---- plotting modes ----
    if plotting_mode.lower() == "single":
        for name, meta in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(9.5, 6))
            niter_null = _plot_one_metric(ax, name, meta)
            plt.tight_layout()

            if save_enabled:
                out_pdf = os.path.join(save_dir, f"{name.upper()}_null_vs_alt_PDF.pdf")
                plt.savefig(out_pdf)
                print(f"[{name.upper()}] saved {out_pdf}")
            else:
                # no saving when save_dir is None/empty
                pass

            plt.show()
            plt.close(fig)

    elif plotting_mode.lower() == "multiple":
        n = n_metrics_found
        if n == 0:
            raise ValueError("No metrics to plot in 'multiple' mode.")
        ncols = int(ncols_multiple) if ncols_multiple and ncols_multiple > 0 else 3
        nrows = math.ceil(n / ncols)
        figsize = (6.0 * ncols, 6.0 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, dpi=300)
        axes_flat = axes.ravel()

        for i, (name, meta) in enumerate(metrics_to_plot):
            _plot_one_metric(axes_flat[i], name, meta)

        # hide unused axes
        for j in range(len(metrics_to_plot), len(axes_flat)):
            axes_flat[j].axis('off')

        plt.suptitle(fr"Null vs Alternative Hypothesis Distributions — $d={ndims}$, $\text{{batch size}} = {batch_size_test}$", fontsize=18)
        plt.tight_layout()

        if save_enabled:
            out_pdf = os.path.join(save_dir, f"{unique_key}_null_vs_alt_PDF_grid_{n}.pdf")
            plt.savefig(out_pdf)
            print(f"[GRID] saved {out_pdf}")
        else:
            # not saving
            pass

        plt.show()
        plt.close(fig)

    else:
        raise ValueError("plotting_mode must be 'single' or 'multiple'.")





def plot_null_vs_alt_ecdf(
    metrics_config: dict,
    unique_key: str,
    observed: dict,            # output of compute_observed_distributions (INTERNAL scale; FGD×10 already applied)
    ndims: int,
    save_dir: str,
    *,
    show_power: bool = False,   # if True, shows fraction of Alt above each CL threshold
    power_text_loc: tuple = (0.99, 1.02),  # (x,y) in axes fraction
):
    """
    eCDF overlay of H0 (null) vs H1 (alt), with:
      • CL thresholds computed as quantile(null_internal, CL) and shaded right tails
      • Same display scaling as your PDF plots (FGD ×1e-7 label; MMD ×1e5 label)
      • Step closures at left (first jump) and right (last x)
      • Optional on-plot power annotation: fraction of Alt above each CL threshold
    """

    os.makedirs(save_dir, exist_ok=True)

    def _safe_seed(val, default=0):
        try: s = int(val)
        except Exception: s = int(default)
        return max(0, min(s, 2**32 - 1))

    # eCDF utility: returns sorted x and F(x) in [0,1], with where='post' interpretation
    def _ecdf(x: np.ndarray):
        xs = np.sort(np.asarray(x, dtype=float))
        n  = xs.size
        if n == 0:
            return np.array([0.0]), np.array([0.0])
        ys = np.arange(1, n+1, dtype=float)/float(n)
        return xs, ys

    for name, meta in metrics_config[unique_key].items():
        if name.lower() in {"lr", "likelihood_ratio", "likelihoodratio"}:
            continue

        # --- restore/configure metric instance to read null Results and metadata
        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        dummy_inputs = GMetrics.TwoSampleTestInputs(
            dist_1_input=np.zeros((1, ndims), dtype=np.float32),
            dist_2_input=np.zeros((1, ndims), dtype=np.float32),
            niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
        )
        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        # --- INTERNAL scaling
        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], dtype=float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        nd            = int(ndims)
        ns_eff        = nsamples_null / 2.0  # n=m

        scale_func = meta.get("scale_func", None)
        internal_scale = scale_func(ns_eff, nd) if callable(scale_func) else 1.0

        null_scaled_internal = null_raw * float(internal_scale)
        if name.lower() == "fgd":
            null_scaled_internal *= 10.0

        alt_scaled_internal = np.asarray(observed[name]["alt_array"], dtype=float)

        # --- DISPLAY scaling + label
        latex_label = meta["latex"]
        display_scale = 1.0
        if "FGD" in latex_label:
            display_scale = 1e-7
            latex_label   = r"$t_{\mathrm{FGD}}\,(\times 10^{-7})$"
        elif "MMD" in latex_label:
            display_scale = 1e5
            latex_label   = r"$t_{\mathrm{MMD}}\,(\times 10^{5})$"

        null_plot = null_scaled_internal * display_scale
        alt_plot  = alt_scaled_internal  * display_scale

        # --- CL thresholds (quantile at CL) in DISPLAY units, with defaults if none stored
        cls_from_meta = [float(cl) for (cl, _, _) in meta.get("thresholds", [])] if meta.get("thresholds") else []
        cl_list = sorted(set(cls_from_meta if cls_from_meta else [0.68, 0.95, 0.99]))
        thr_internal = {int(cl*100): float(np.quantile(null_scaled_internal, cl)) for cl in cl_list}
        thresholds = {cl: thr_internal[cl] * display_scale for cl in thr_internal.keys()}

        # --- eCDFs
        x0, F0 = _ecdf(null_plot)
        x1, F1 = _ecdf(alt_plot)

        # x-limits (pad + include thresholds)
        x_min = float(min(x0[0], x1[0])) if len(x0) and len(x1) else (x0[0] if len(x0) else (x1[0] if len(x1) else 0.0))
        x_max = float(max(x0[-1], x1[-1])) if len(x0) and len(x1) else (x0[-1] if len(x0) else (x1[-1] if len(x1) else 1.0))
        max_thr = max(thresholds.values()) if len(thresholds) else x_max
        x_max   = max(x_max, max_thr)
        xpad    = 0.04 * (x_max - x_min) if np.isfinite(x_max - x_min) else 0.0
        x_left, x_right = x_min - xpad, x_max + xpad

        # --- plotting
        fig, ax = plt.subplots(figsize=(9.5, 6))

        # Null eCDF (tomato) + closures
        null_line = ax.step(x0, F0, where='post', color="tomato", linewidth=1.8, label="Null (eCDF)")[0]
        null_style = dict(color=null_line.get_color(),
                          linewidth=null_line.get_linewidth(),
                          linestyle=null_line.get_linestyle(),
                          alpha=null_line.get_alpha())
        if len(F0):
            ax.vlines(x0[0],   0.0, F0[0], **null_style, zorder=3)     # left closure
            ax.vlines(x0[-1],  0.0, F0[-1], **null_style, zorder=3)    # right closure to exact last x

        # Alt eCDF (slateblue) + closures
        alt_line = ax.step(x1, F1, where='post', color="slateblue", linewidth=2.2, label="Alt (eCDF)")[0]
        alt_style = dict(color=alt_line.get_color(),
                         linewidth=alt_line.get_linewidth(),
                         linestyle=alt_line.get_linestyle(),
                         alpha=alt_line.get_alpha())
        if len(F1):
            ax.vlines(x1[0],   0.0, F1[0], **alt_style, zorder=3)
            ax.vlines(x1[-1],  0.0, F1[-1], **alt_style, zorder=3)

        # Threshold shading/markers (DISPLAY units, right tail)
        dash_styles = ['-', '--', '-.']
        colors = ["royalblue", "darkorange", "mediumseagreen"]
        for i, cl in enumerate(sorted(thresholds.keys())):
            thr = thresholds[cl]
            ax.axvspan(thr, x_right, color=colors[i % len(colors)], alpha=0.08, label=f"{cl}% CL")
            ax.axvline(thr, linestyle=dash_styles[i % len(dash_styles)],
                       color=colors[i % len(colors)], linewidth=1.2)

        # Axes + labels
        #if name.lower() == "fgd":
        #    ax.set_xscale('symlog', linthresh=1)
        ax.set_ylim(0.0, 1.0 + 0.04)   # top padding
        ax.set_xlim(x_left, x_right)
        ax.set_xlabel(latex_label, fontsize=16)
        ax.set_ylabel("eCDF", fontsize=16)
        ax.set_title(rf"{name.upper()} — $d={nd}$, $n=m={nsamples_null//1000}$K, $n_{{\mathrm{{iter}}}}={niter_null}$",
                     fontsize=15)
        leg = ax.legend(loc='lower right', fontsize=13, frameon=False)
        for h in leg.legend_handles[-min(3, len(thresholds)):]:
            try: h.set_alpha(0.5)
            except Exception: pass

        # Optional: power annotation — fraction of Alt beyond each CL threshold
        if show_power and len(x1):
            txt = "Alt > CL: " + ", ".join([f"{cl}%: {(alt_plot > thresholds[cl]).mean()*100:.1f}%"
                                            for cl in sorted(thresholds)])
            ax.text(power_text_loc[0], power_text_loc[1], txt, ha='right', va='bottom',
                    transform=ax.transAxes, fontsize=12)

        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_pdf = os.path.join(save_dir, f"{name.upper()}_null_vs_alt_eCDF.pdf")
        plt.savefig(out_pdf)
        plt.show()
        plt.close()
        print(f"[{name.upper()}] saved {out_pdf}")



def plot_power_curves_all(
    metrics_config: dict,
    unique_key: str,
    observed: dict,            # output of compute_observed_distributions (INTERNAL scale; FGD×10 already applied)
    ndims: int,
    save_dir: str,
    *,
    alpha_grid=None,           # scalar or iterable in (0,1); defaults provided below
    logx: bool = True,         # log-scale for alpha (recommended)
    alpha_mark: float = 0.05,  # draw a marker at this alpha
):
    """
    Plot ROC-like power curves for ALL metrics on a single figure.

    Definitions (RIGHT-TAILED):
      • t_α = Quantile_H0(1 - α)
      • π(α) = mean_{H1}[ T_alt > t_α ]

    Conventions:
      • INTERNAL scaling for thresholds & alt (scale_func; FGD×10).
      • DISPLAY scaling is irrelevant here (we plot π vs α only).
      • Skips LR by default (tailing may differ).

    Returns:
      results: { metric_name: { 'alpha': np.ndarray, 'power': np.ndarray, 'power_at_mark': float } }
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---- alpha grid handling (accept scalar or iterable) ----
    if alpha_grid is None:
        # Good default: dense log sweep from 1e-3 to 1e-1 (inclusive), plus a couple of linear points up to 0.2
        alpha_grid = np.concatenate([
            np.logspace(-3, -1, 24),
            np.linspace(0.11, 0.20, 10)
        ])
    elif np.isscalar(alpha_grid):
        alpha_grid = np.array([float(alpha_grid)], dtype=float)
    else:
        alpha_grid = np.asarray(alpha_grid, dtype=float)

    # Clean grid: keep (0,1); sort unique; remove exact duplicates
    alpha_grid = alpha_grid[(alpha_grid > 0.0) & (alpha_grid < 1.0)]
    alpha_grid = np.unique(np.sort(alpha_grid))
    if alpha_grid.size == 0:
        raise ValueError("alpha_grid must contain values in (0,1).")

    # Utility
    def _safe_seed(val, default=0):
        try: s = int(val)
        except Exception: s = int(default)
        return max(0, min(s, 2**32 - 1))

    results = {}
    # For a unified title, read nsamples/niter from the first metric encountered
    title_nsamples = None
    title_niter    = None

    # Prepare figure once
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Nice distinct colors for multiple curves
    color_cycle = ["slateblue", "tomato", "mediumpurple", "seagreen", "goldenrod",  "teal", "firebrick", "darkorange"]
    dash_styles = ['-', '--', '-.', ':']

    # Iterate metrics in a stable, readable order
    for idx, (name, meta) in enumerate(sorted(metrics_config[unique_key].items(), key=lambda kv: kv[0].lower())):
        if name.lower() in {"lr", "likelihood_ratio", "likelihoodratio"}:
            continue

        # --- load metric + null results (to read config & null array)
        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        dummy_inputs = GMetrics.TwoSampleTestInputs(
            dist_1_input=np.zeros((1, ndims), dtype=np.float32),
            dist_2_input=np.zeros((1, ndims), dtype=np.float32),
            niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
        )
        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        # --- INTERNAL scaling (used for thresholds & power calculation)
        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], dtype=float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        nd            = int(ndims)
        ns_eff        = nsamples_null / 2.0  # n=m

        if title_nsamples is None:  # fill title info from first metric
            title_nsamples = nsamples_null
            title_niter    = niter_null

        scale_func = meta.get("scale_func", None)
        internal_scale = scale_func(ns_eff, nd) if callable(scale_func) else 1.0

        null_scaled_internal = null_raw * float(internal_scale)
        if name.lower() == "fgd":
            null_scaled_internal *= 10.0

        # Alt (already INTERNAL-scaled in 'observed')
        alt_scaled_internal = np.asarray(observed[name]["alt_array"], dtype=float)
        if alt_scaled_internal.size == 0 or null_scaled_internal.size == 0:
            # Nothing to compute; skip gracefully
            continue

        # --- thresholds & power (RIGHT-TAILED)
        # For vectorized power on a grid: thresholds shape [K], alt shape [N]
        thresholds = np.quantile(null_scaled_internal, 1.0 - alpha_grid, method="linear")
        power = (alt_scaled_internal.reshape(1, -1) > thresholds.reshape(-1, 1)).mean(axis=1)

        # Also compute power exactly at alpha_mark (even if not in the grid)
        alpha_mark = float(alpha_mark)
        if not (0.0 < alpha_mark < 1.0):
            alpha_mark = 0.05  # fallback
        t_mark = float(np.quantile(null_scaled_internal, 1.0 - alpha_mark, method="linear"))
        power_mark = float((alt_scaled_internal > t_mark).mean())

        results[name] = {
            "alpha": alpha_grid.copy(),
            "power": power.copy(),
            "power_at_mark": power_mark,
            "alpha_mark": alpha_mark
        }

        # --- Plot this curve
        color = color_cycle[idx % len(color_cycle)]
        style = dash_styles[idx % len(dash_styles)]
        label = name.upper()

        ax.plot(alpha_grid, power, color=color, linestyle=style, linewidth=2.4, label=label)
        # Mark α = alpha_mark
        ax.scatter([alpha_mark], [power_mark], s=48, color=color, zorder=3)

    # Cosmetics / axes
    #ax.set_ylim(0.0, 1.0)
    ax.set_yscale('symlog', linthresh=4e-2)

    if logx:
        ax.set_xscale('symlog', linthresh=2e-1)
        xmin, xmax = alpha_grid.min(), alpha_grid.max()
        if xmin == xmax:
            ax.set_xlim(xmin / 1.5, xmax * 1.5)
        else:
            ax.set_xlim(xmin * 0.9, xmax * 1.1)
        ax.set_xlabel(r"Significance $\alpha$ (log scale)", fontsize=16)
    else:
        xmin, xmax = alpha_grid.min(), alpha_grid.max()
        if xmin == xmax:
            pad = 0.01 if xmin < 0.5 else 0.05
            ax.set_xlim(xmin - pad, xmax + pad)
        else:
            ax.set_xlim(xmin, xmax)
        ax.set_xlabel(r"Significance $\alpha$", fontsize=16)

    ax.set_ylabel(r"Power  $\pi(\alpha)$", fontsize=16)

    # Title (from first metric's config, assuming consistent across metrics)
    if title_nsamples is None:
        title_nsamples = 0
    if title_niter is None:
        title_niter = 0
    ax.set_title(rf"Power Curves — $d={ndims}$, $n=m={title_nsamples//1000}$K, $n_{{\mathrm{{iter}}}}={title_niter}$",
                 fontsize=15)

    ax.legend(loc='upper left', fontsize=12, frameon=False, ncol=1)
    ax.grid(True, which='both', axis='both', alpha=0.3)

    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    out_pdf = os.path.join(save_dir, "ALL_metrics_power_curves.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.show()
    plt.close()
    print("[ALL] saved ")

    return results






'''
def compute_and_plot_alt_pvalues(
    metrics_config: dict,
    unique_key: str,
    observed: dict,
    ndims: int,
    save_dir: str,
    *,
    tail: str = "right",
    num_bins: int = 40,
    top_pad_ecdf: float = 0.04,
    plot_kind: str = "both",   # "both" | "pdf" | "cdf"
):

    os.makedirs(save_dir, exist_ok=True)

    # normalize flag
    plot_kind = (plot_kind or "both").lower()
    if plot_kind not in {"both", "pdf", "cdf"}:
        plot_kind = "both"
    want_pdf  = plot_kind in {"both", "pdf"}
    want_ecdf = plot_kind in {"both", "cdf"}

    def _safe_seed(val, default=0):
        try:
            s = int(val)
        except Exception:
            s = int(default)
        return max(0, min(s, 2**32 - 1))

    def _compute_pvalues(null_samples, alt_samples, mode="right"):
        x = np.asarray(null_samples, float)
        t = np.asarray(alt_samples, float)
        if x.size == 0 or t.size == 0:
            return np.empty_like(t)
        x = x[np.isfinite(x)]
        t = t[np.isfinite(t)]
        if x.size == 0 or t.size == 0:
            return np.empty_like(t)
        x_sorted = np.sort(x)
        N = x_sorted.size
        r_idx = np.searchsorted(x_sorted, t, side="right")
        p_right = (N - r_idx) / N
        l_idx = np.searchsorted(x_sorted, t, side="left")
        p_left = l_idx / N
        mode = (mode or "right").lower()
        if mode == "right":
            p = p_right
        elif mode == "left":
            p = p_left
        elif mode in {"two-sided", "two_sided", "2s", "two"}:
            p = 2.0 * np.minimum(p_left, p_right)
            p = np.clip(p, 0.0, 1.0)
        else:
            p = p_right
        eps = 1.0 / (N + 1.0)
        return np.clip(p, eps, 1.0)

    def _stairs_hist(ax, data, bins, rng, color, label, linewidth=2.0):
        counts, edges = np.histogram(data, bins=bins, range=rng, density=True)
        line = ax.step(edges[:-1], counts, where='post', color=color, linewidth=linewidth, label=label)[0]
        style = dict(color=line.get_color(), linewidth=line.get_linewidth(),
                     linestyle=line.get_linestyle(), alpha=line.get_alpha())
        if len(counts) > 0:
            ax.hlines(counts[0],  edges[0],  edges[1],  **style)
            ax.hlines(counts[-1], edges[-2], edges[-1], **style)
            ax.vlines(edges[0],   0.0,       counts[0], **style)
            ax.vlines(edges[-1],  counts[-1], 0.0,      **style)
        return counts, edges

    def _plot_ecdf(ax, pvals, color="slateblue", label="ALT p-values", linewidth=2.2, top_pad=0.04):
        if pvals.size == 0:
            return
        x = np.sort(pvals)
        y = np.arange(1, x.size + 1, dtype=float) / x.size
        line = ax.step(x, y, where="post", color=color, linewidth=linewidth, label=label)[0]
        style = dict(color=line.get_color(), linewidth=line.get_linewidth(),
                     linestyle=line.get_linestyle(), alpha=line.get_alpha())
        ax.hlines(y[-1], x[-1], 1.0, **style)
        ax.vlines(0.0, 0.0, y[0], **style)
        ax.vlines(1.0, y[-1], 0.0, **style)
        ax.plot([0, 1], [0, 1], color="tomato", linewidth=1.6, linestyle="--", label="Uniform under H0")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.0 + top_pad)

    out = {}

    for name, meta in metrics_config[unique_key].items():
        if name.lower() in {"lr", "likelihood_ratio", "likelihoodratio"}:
            continue

        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        dummy_inputs = GMetrics.TwoSampleTestInputs(
            dist_1_input=np.zeros((1, ndims), dtype=np.float32),
            dist_2_input=np.zeros((1, ndims), dtype=np.float32),
            niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
        )
        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        nd            = int(ndims)
        ns_eff        = nsamples_null / 2.0

        scale_func = meta.get("scale_func", None)
        internal_scale = scale_func(ns_eff, nd) if callable(scale_func) else 1.0
        null_internal = null_raw * float(internal_scale)
        if name.lower() == "fgd":
            null_internal *= 10.0

        if name not in observed or "alt_array" not in observed[name]:
            print(f"[{name.upper()}] No ALT data found in observed — skipping.")
            continue
        alt_internal = np.asarray(observed[name]["alt_array"], float)

        pvals = _compute_pvalues(null_internal, alt_internal, mode=tail)
        out[name] = pvals

        alpha_lines = [(0.32, "royalblue", "-"),
                       (0.05, "darkorange", "--"),
                       (0.01, "mediumseagreen", "-.")]

        # ------------------- PDF -------------------
        if want_pdf:
            fig, ax = plt.subplots(figsize=(9.5, 6))
            _ = _stairs_hist(ax, pvals, bins=num_bins, rng=(0.0, 1.0),
                             color="slateblue", label="ALT p-values", linewidth=2.2)
            for alpha, c, ls in alpha_lines:
                ax.axvline(alpha, color=c, linestyle=ls, linewidth=1.3,
                           label=f"α={alpha:.2g}")
            # background grid
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("p-value", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.set_title(rf"{name.upper()} p-values — tail='{tail}', $d={nd}$, $n=m={nsamples_null//1000}$K, $n_{{\mathrm{{iter}}}}={niter_null}$",
                         fontsize=15)
            ax.legend(loc='upper right', fontsize=13, frameon=False)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xlim(-0.01, 1.01)
            plt.tight_layout()
            out_pdf = os.path.join(save_dir, f"{name.upper()}_ALT_pvalues_hist.pdf")
            plt.savefig(out_pdf); plt.show(); plt.close()
            print(f"[{name.upper()}] saved {out_pdf}")

        # ------------------- ECDF -------------------
        if want_ecdf:
            fig, ax = plt.subplots(figsize=(9.5, 6))
            _plot_ecdf(ax, pvals, color="slateblue", label="ALT p-values", top_pad=top_pad_ecdf)
            for alpha, c, ls in alpha_lines:
                ax.axvline(alpha, color=c, linestyle=ls, linewidth=1.3, label=f"α={alpha:.2g}")
            # background grid
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("p-value", fontsize=16)
            ax.set_ylabel("eCDF", fontsize=16)
            ax.set_title(rf"{name.upper()} p-values eCDF — tail='{tail}', $d={nd}$, $n=m={nsamples_null//1000}$K, $n_{{\mathrm{{iter}}}}={niter_null}$",
                         fontsize=15)
            ax.legend(loc='lower right', fontsize=13, frameon=False)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            plt.tight_layout()
            out_pdf = os.path.join(save_dir, f"{name.upper()}_ALT_pvalues_eCDF.pdf")
            plt.savefig(out_pdf); plt.show(); plt.close()
            print(f"[{name.upper()}] saved {out_pdf}")

    return out
'''

def compute_and_plot_alt_pvalues(
    metrics_config: dict,
    unique_key: str,
    observed: dict,
    ndims: int,
    save_dir: str | None,
    *,
    tail: str = "right",
    num_bins: int = 40,
    top_pad_ecdf: float = 0.04,
    plot_kind: str = "both",   # "both" | "pdf" | "cdf"
    batch_size_test: int = 1000,
):
    """
    Computes ALT p-values for each metric and:
      • (optional) plots per-metric PDF histograms (one figure per metric)
      • plots ALL eCDFs on ONE figure (rows not needed; curves overlayed)
    If save_dir is None/empty, figures are shown but not saved.
    Colors/dash styles follow plot_power_curves_all for metric curves.
    """
    import os, numpy as np, matplotlib.pyplot as plt

    # Optional save control
    save_enabled = isinstance(save_dir, str) and bool(save_dir.strip())
    if save_enabled:
        os.makedirs(save_dir, exist_ok=True)

    # normalize flags
    plot_kind = (plot_kind or "both").lower()
    if plot_kind not in {"both", "pdf", "cdf"}:
        plot_kind = "both"
    want_pdf  = plot_kind in {"both", "pdf"}
    want_ecdf = plot_kind in {"both", "cdf"}

    # Utilities
    def _safe_seed(val, default=0):
        try:
            s = int(val)
        except Exception:
            s = int(default)
        return max(0, min(s, 2**32 - 1))

    def _compute_pvalues(null_samples, alt_samples, mode="right"):
        x = np.asarray(null_samples, float)
        t = np.asarray(alt_samples, float)
        if x.size == 0 or t.size == 0: return np.empty_like(t)
        x = x[np.isfinite(x)]; t = t[np.isfinite(t)]
        if x.size == 0 or t.size == 0: return np.empty_like(t)

        x_sorted = np.sort(x); N = x_sorted.size
        r_idx = np.searchsorted(x_sorted, t, side="right")
        p_right = (N - r_idx) / N
        l_idx = np.searchsorted(x_sorted, t, side="left")
        p_left = l_idx / N

        mode = (mode or "right").lower()
        if mode == "right":
            p = p_right
        elif mode == "left":
            p = p_left
        elif mode in {"two-sided", "two_sided", "2s", "two"}:
            p = 2.0 * np.minimum(p_left, p_right)
            p = np.clip(p, 0.0, 1.0)
        else:
            p = p_right

        eps = 1.0 / (N + 1.0)
        return np.clip(p, eps, 1.0)

    def _stairs_hist(ax, data, bins, rng, color, label, linewidth=2.2):
        counts, edges = np.histogram(data, bins=bins, range=rng, density=True)
        line = ax.step(edges[:-1], counts, where='post', color=color, linewidth=linewidth, label=label)[0]
        style = dict(color=line.get_color(), linewidth=line.get_linewidth(),
                     linestyle=line.get_linestyle(), alpha=line.get_alpha())
        if len(counts) > 0:
            ax.hlines(counts[0],  edges[0],  edges[1],  **style)
            ax.hlines(counts[-1], edges[-2], edges[-1], **style)
            ax.vlines(edges[0],   0.0,       counts[0], **style)
            ax.vlines(edges[-1],  counts[-1], 0.0,      **style)
        return counts, edges

    def _ecdf_xy(vals: np.ndarray):
        x = np.sort(vals)
        y = np.arange(1, x.size + 1, dtype=float) / max(1, x.size)
        return x, y

    # storage
    out = {}
    ecdf_cache = {}  # metric -> (x,y)

    # Alpha markers (like your previous p-value plots)
    alpha_lines = [(0.32, "royalblue", "-"),
                   (0.05, "darkorange", "--"),
                   (0.01, "mediumseagreen", "-.")]

    # Read modules used by metrics (kept local to avoid global deps)
    import numpy as np
    try:
        import tensorflow as tf
    except Exception:
        pass
    try:
        import GMetrics
    except Exception:
        raise ImportError("GMetrics is required but not importable in this scope.")

    # For a unified title on the ECDF plot, capture nsamples/niter from 1st metric
    title_nsamples = None
    title_niter    = None

    # Iterate metrics (stable order)
    for name, meta in sorted(metrics_config[unique_key].items(), key=lambda kv: kv[0].lower()):
        if name.lower() in {"lr", "likelihood_ratio", "likelihoodratio"}:
            continue

        MetricClass   = eval(meta["class_name"])
        metric_kwargs = dict(meta.get("kwargs", {}))
        if "seed_slicing" in metric_kwargs:
            metric_kwargs["seed_slicing"] = _safe_seed(metric_kwargs["seed_slicing"], default=0)

        dummy_inputs = GMetrics.TwoSampleTestInputs(
            dist_1_input=np.zeros((1, ndims), dtype=np.float32),
            dist_2_input=np.zeros((1, ndims), dtype=np.float32),
            niter=1, batch_size_test=1, dtype_input=tf.float32, use_tf=True
        )
        null_obj = MetricClass(dummy_inputs, progress_bar=False, verbose=False, **metric_kwargs)
        null_obj.Results.load_from_json(meta["null_file"])

        result_key = meta["result_key"]
        null_raw   = np.asarray(null_obj.Results[-1].result_value[result_key], float)

        nsamples_null = int(meta["test_config"]["batch_size_test"])
        niter_null    = int(meta["test_config"]["niter"])
        nd            = int(ndims)
        ns_eff        = nsamples_null / 2.0

        if title_nsamples is None:
            title_nsamples = nsamples_null
            title_niter    = niter_null

        scale_func = meta.get("scale_func", None)
        internal_scale = scale_func(ns_eff, nd) if callable(scale_func) else 1.0
        null_internal = null_raw * float(internal_scale)
        if name.lower() == "fgd":
            null_internal *= 10.0

        if name not in observed or "alt_array" not in observed[name]:
            print(f"[{name.upper()}] No ALT data found in observed — skipping.")
            continue
        alt_internal = np.asarray(observed[name]["alt_array"], float)

        # p-values for this metric
        pvals = _compute_pvalues(null_internal, alt_internal, mode=tail)
        out[name] = pvals

        # cache eCDF curve
        x_ecdf, y_ecdf = _ecdf_xy(pvals)
        ecdf_cache[name] = (x_ecdf, y_ecdf)

        # --- per-metric PDF (optional, keeps your previous behavior) ---
        if want_pdf:
            fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
            _ = _stairs_hist(ax, pvals, bins=num_bins, rng=(0.0, 1.0),
                             color="slateblue", label="ALT p-values", linewidth=2.2)
            for alpha, c, ls in alpha_lines:
                ax.axvline(alpha, color=c, linestyle=ls, linewidth=1.3, label=f"α={alpha:.2g}")

            ax.grid(True, alpha=0.3)
            ax.set_xlabel("p-value", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.set_title(rf"{name.upper()} p-values — tail='{tail}', $d={nd}$, $n=m={nsamples_null//1000}$K, $n_{{\mathrm{{iter}}}}={niter_null}$",
                         fontsize=15)
            ax.legend(loc='upper right', fontsize=13, frameon=False)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xlim(-0.01, 1.01)
            plt.tight_layout()

            if save_enabled:
                out_pdf = os.path.join(save_dir, f"{name.upper()}_ALT_pvalues_hist.pdf")
                plt.savefig(out_pdf)
            plt.show(); plt.close()

    # ----------------- ONE CANVAS: eCDF for ALL metrics -----------------
    if want_ecdf and len(ecdf_cache) > 0:
        fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)

        # Same palette & style logic as plot_power_curves_all
        color_cycle = ["slateblue", "tomato", "mediumpurple", "seagreen", "goldenrod",
                       "teal", "firebrick", "darkorange"]
        dash_styles = ['-', '--', '-.', ':']

        # Diagonal reference (Uniform under H0)
        ax.plot([0, 1], [0, 1], color="tomato", linewidth=1.6, linestyle="--", label="Uniform under H0")

        # Plot each metric eCDF
        for idx, (name, (x, y)) in enumerate(sorted(ecdf_cache.items(), key=lambda kv: kv[0].lower())):
            color = color_cycle[idx % len(color_cycle)]
            style = dash_styles[idx % len(dash_styles)]
            label = name.upper()
            ax.step(x, y, where="post", color=color, linestyle=style, linewidth=2.4, label=label)

        # α reference lines (keep same colors as before)
        for alpha, c, ls in alpha_lines:
            ax.axvline(alpha, color=c, linestyle=ls, linewidth=1.3, label=f"α={alpha:.2g}")

        # Axes/limits/grid
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.0 + float(top_pad_ecdf))
        ax.set_xlabel("p-value", fontsize=16)
        ax.set_ylabel("eCDF", fontsize=16)

        # Title (consistent with power-curves style)
        if title_nsamples is None: title_nsamples = 0
        if title_niter is None:    title_niter = 0
        ax.set_title(rf"p-values eCDF — tail='{tail}', $d={ndims}$, $\text{{batch size}}={batch_size_test//1000}$K, $n_{{\mathrm{{iter}}}}={title_niter}$",
                     fontsize=15)

        ax.legend(loc='lower right', fontsize=12, frameon=False, ncol=1)
        ax.grid(True, which='both', axis='both', alpha=0.3)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

        plt.tight_layout()
        if save_enabled:
            out_pdf = os.path.join(save_dir, "ALL_metrics_ALT_pvalues_eCDF.pdf")
            plt.savefig(out_pdf)
        plt.show(); plt.close()

    return out

    





