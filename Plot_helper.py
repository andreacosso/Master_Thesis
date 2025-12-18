
import h5py
import numpy as np
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from typing import List, Tuple, Dict, Union, Optional, Any

# ─── Modified get_original_shower ─────────────────────────────────────────────

def get_original_shower(cache_file: str, event_id: int) -> tuple[tf.Tensor, float]:
    """
    Open the HDF5 cache_file, read 'showers_raw' and 'incident_energies',
    grab the event_id’th entry, and return:
      1) original_shower: tf.Tensor shape [1, 6480], raw fine‐voxel energies in MeV
      2) incident_energy: float, true Geant4 incident energy for that event
    """
    with h5py.File(cache_file, "r") as f:
        all_showers      = f["showers_raw"][:]             # shape [N, 45, 16, 9]
        all_inc_energies = f["incident_energies"][:]       # shape [N]

    # 1) Extract the single event’s 45×16×9 array:
    single = all_showers[event_id]   # shape (45, 16, 9)

    # 2) Flatten row‐major → shape (6480,)
    flat = single.reshape(-1)

    # 3) Add batch dim → (1, 6480)
    flat_batched = flat[np.newaxis, :]

    # 4) Convert to tf.Tensor float32
    shower_tf = tf.convert_to_tensor(flat_batched, dtype=tf.float32)

    # 5) Extract the true incident energy as a Python float
    incident_energy = all_inc_energies[event_id].item()

    return shower_tf, incident_energy


# ─── Utility to ensure we have a NumPy array for HLF ─────────────────────────

def to_numpy_array(x):
    """
    If x is a tf.Tensor, return x.numpy().
    If x is already a NumPy array, return x directly.
    """
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        try:
            return np.asarray(x)
        except Exception:
            raise ValueError("Input must be a tf.Tensor or np.ndarray.")


# ─── Visualization Functions (handling both tf.Tensor and np.ndarray) ───────

def plot_two_showers(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    hlf: object,
    save_prefix: str = None
) -> None:
    """
    Side-by-side polar plots of the flow-generated shower vs. the true Geant4 shower.

    Args:
      - full_calo_generated: tf.Tensor or np.ndarray of shape [1, 6480] (MeV)
      - cache_file:          path to HDF5 cache
      - event_id:            which event to compare
      - hlf:                 instance of HighLevelFeatures (already initialized)
      - save_prefix:         optional string prefix for saving figures
    """
    # 1) Extract original shower + true incident energy
    original_shower_tf, true_inc_E = get_original_shower(cache_file, event_id)

    # 2) Convert both to NumPy arrays before feeding into HLF:
    full_calo_np       = to_numpy_array(full_calo_generated)   # shape [1, 6480]
    original_shower_np = to_numpy_array(original_shower_tf)     # shape [1, 6480]

    # 3) Plot generated shower
    hlf.CalculateFeatures(full_calo_np)
    title_gen = f"Flow-sampled shower (event {event_id}), E_inc ≃ {true_inc_E:.1f} MeV"
    hlf.DrawSingleShower(
        full_calo_np[0],
        filename=(f"{save_prefix}gen_event{event_id}.png" if save_prefix else None),
        title=title_gen
    )

    # 4) Plot original Geant4 shower
    hlf.CalculateFeatures(original_shower_np)
    title_true = f"True Geant4 shower (event {event_id}), E_inc = {true_inc_E:.1f} MeV"
    hlf.DrawSingleShower(
        original_shower_np[0],
        filename=(f"{save_prefix}true_event{event_id}.png" if save_prefix else None),
        title=title_true
    )


def plot_residual(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    hlf: object,
    save_prefix: str = None
) -> None:
    """
    Compute and plot the per-voxel residual (generated − true) in a polar-layer heatmap.

    Args:
      - full_calo_generated: tf.Tensor or np.ndarray of shape [1, 6480] (MeV)
      - cache_file:          path to HDF5 cache
      - event_id:            which event to compare
      - hlf:                 instance of HighLevelFeatures
      - save_prefix:         optional string prefix for saving figures
    """
    original_shower_tf, _ = get_original_shower(cache_file, event_id)

    # 1) Convert to NumPy arrays
    gen_np  = to_numpy_array(full_calo_generated)[0]   # shape (6480,)
    true_np = to_numpy_array(original_shower_tf)[0]    # shape (6480,)

    # 2) Compute residual (gen − true)
    residual = gen_np - true_np                        # shape (6480,)

    # 3) Prepare [1, 6480] array for HLF
    residual_np = residual[np.newaxis, :]              # shape (1, 6480)

    # 4) Plot residual
    hlf.CalculateFeatures(residual_np)
    title_res = f"Residual (gen − true), event {event_id}"
    hlf.DrawSingleShower(
        residual_np[0],
        filename=(f"{save_prefix}residual_event{event_id}.png" if save_prefix else None),
        title=title_res
    )

'''
def plot_scalar_quantities(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    hlf: object,
    save_prefix: str = None
) -> None:
    """
    Plot layer-by-layer energy, centroids, and widths for the generated vs. true shower.

    Creates a 2×2 grid of subplots:
      1) Layer total energy comparison
      2) Eta-centroid vs. layer
      3) Phi-centroid vs. layer
      4) Widths (eta & phi) vs. layer

    Args:
      - full_calo_generated: tf.Tensor or np.ndarray of shape [1, 6480] (MeV)
      - cache_file:          path to HDF5 cache
      - event_id:            which event to compare
      - hlf:                 instance of HighLevelFeatures
      - save_prefix:         optional string prefix for saving figures
    """
    original_shower_tf, true_inc_E = get_original_shower(cache_file, event_id)

    # Convert both to NumPy before calling HLF
    original_shower_np = to_numpy_array(original_shower_tf)
    full_calo_np       = to_numpy_array(full_calo_generated)

    # (A) True
    hlf.CalculateFeatures(original_shower_np)
    E_layers_true   = hlf.E_layers
    EC_etas_true    = hlf.EC_etas
    EC_phis_true    = hlf.EC_phis
    width_etas_true = hlf.width_etas
    width_phis_true = hlf.width_phis

    # (B) Generated
    hlf.CalculateFeatures(full_calo_np)
    E_layers_gen    = hlf.E_layers
    EC_etas_gen     = hlf.EC_etas
    EC_phis_gen     = hlf.EC_phis
    width_etas_gen  = hlf.width_etas
    width_phis_gen  = hlf.width_phis

    layers = sorted(E_layers_true.keys())

    energies_true   = [E_layers_true[l][0] for l in layers]
    energies_gen    = [E_layers_gen[l][0]  for l in layers]

    eta_true        = [EC_etas_true[l][0] for l in layers]
    eta_gen         = [EC_etas_gen[l][0]  for l in layers]

    phi_true        = [EC_phis_true[l][0] for l in layers]
    phi_gen         = [EC_phis_gen[l][0]  for l in layers]

    width_eta_true  = [width_etas_true[l][0] for l in layers]
    width_eta_gen   = [width_etas_gen[l][0]  for l in layers]

    width_phi_true  = [width_phis_true[l][0] for l in layers]
    width_phi_gen   = [width_phis_gen[l][0]  for l in layers]

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_e      = axes[0, 0]
    ax_eta    = axes[0, 1]
    ax_phi    = axes[1, 0]
    ax_w      = axes[1, 1]

    # (1) Layer energies
    ax_e.plot(layers, energies_true, label="True", marker='o')
    ax_e.plot(layers, energies_gen,  label="Generated", marker='x')
    ax_e.set_title("Layer-by-layer Energy")
    ax_e.set_xlabel("Layer index")
    ax_e.set_ylabel("Total Energy (MeV)")
    ax_e.legend()

    # (2) Eta centroids
    ax_eta.plot(layers, eta_true, label="True", marker='o')
    ax_eta.plot(layers, eta_gen,  label="Generated", marker='x')
    ax_eta.set_title("Eta Centroid vs. Layer")
    ax_eta.set_xlabel("Layer")
    ax_eta.set_ylabel(r"$\eta_{EC}$")
    ax_eta.legend()

    # (3) Phi centroids
    ax_phi.plot(layers, phi_true, label="True", marker='o')
    ax_phi.plot(layers, phi_gen,  label="Generated", marker='x')
    ax_phi.set_title("Phi Centroid vs. Layer")
    ax_phi.set_xlabel("Layer")
    ax_phi.set_ylabel(r"$\phi_{EC}$")
    ax_phi.legend()

    # (4) Widths
    ax_w.plot(layers, width_eta_true,  label=r"True $\sigma_\eta$", marker='o')
    ax_w.plot(layers, width_eta_gen,   label=r"Gen $\sigma_\eta$", marker='x')
    ax_w.plot(layers, width_phi_true,  label=r"True $\sigma_\phi$", marker='s', linestyle='--')
    ax_w.plot(layers, width_phi_gen,   label=r"Gen $\sigma_\phi$", marker='d', linestyle='--')
    ax_w.set_title("Widths vs. Layer")
    ax_w.set_xlabel("Layer")
    ax_w.set_ylabel("Width")
    ax_w.legend()

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}scalar_comparison_event{event_id}.png")
    plt.show()
    plt.close(fig)
'''

def plot_scalar_quantities(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    hlf: object,
    save_prefix: str = None,
):
    """
    Compare scalar shower quantities (energy, η/φ centroids, widths)
    between truth and generated, using a *single* HighLevelFeatures object.

    Parameters
    ----------
    full_calo_generated : np.ndarray | tf.Tensor  shape [1, 6480]
    cache_file          : str   – path to the HDF5 cache
    event_id            : int   – event index to load as truth
    hlf                 : HighLevelFeatures – pre-constructed instance
    save_prefix         : str or None – if provided, png is written as
                         f"{save_prefix}scalar_comparison_event{event_id}.png"
    """

    # 0. true vs generated showers -------------------------------------------
    true_calo_tf, _ = get_original_shower(cache_file, event_id)
    true_calo_np    = to_numpy_array(true_calo_tf)
    gen_calo_np     = to_numpy_array(full_calo_generated)

    # 1. ---- TRUE shower features -------------------------------------------
    hlf.CalculateFeatures(true_calo_np)

    # deep-copy the dictionaries so they survive the next call
    true_E_layers = copy.deepcopy(hlf.GetElayers())
    true_EC_etas  = copy.deepcopy(hlf.GetECEtas())
    true_EC_phis  = copy.deepcopy(hlf.GetECPhis())
    true_w_eta    = copy.deepcopy(hlf.GetWidthEtas())
    true_w_phi    = copy.deepcopy(hlf.GetWidthPhis())

    # 2. ---- GENERATED shower features --------------------------------------
    hlf.CalculateFeatures(gen_calo_np)

    gen_E_layers = hlf.GetElayers()
    gen_EC_etas  = hlf.GetECEtas()
    gen_EC_phis  = hlf.GetECPhis()
    gen_w_eta    = hlf.GetWidthEtas()
    gen_w_phi    = hlf.GetWidthPhis()

    # 3. ---- build ordered lists --------------------------------------------
    layers = sorted(true_E_layers.keys())

    energies_true = [true_E_layers[l][0] for l in layers]
    energies_gen  = [gen_E_layers [l][0] for l in layers]

    eta_true = [true_EC_etas[l][0] for l in layers]
    eta_gen  = [gen_EC_etas [l][0] for l in layers]

    phi_true = [true_EC_phis[l][0] for l in layers]
    phi_gen  = [gen_EC_phis [l][0] for l in layers]

    w_eta_true = [true_w_eta[l][0] for l in layers]
    w_eta_gen  = [gen_w_eta [l][0] for l in layers]

    w_phi_true = [true_w_phi[l][0] for l in layers]
    w_phi_gen  = [gen_w_phi [l][0] for l in layers]

    # 4. ---- plotting --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_E, ax_eta = axes[0]
    ax_phi, ax_w = axes[1]

    # (a) Layer energies
    ax_E.plot(layers, energies_true, "o-", label="True")
    ax_E.plot(layers, energies_gen , "x-", label="Generated")
    ax_E.set_title("Layer-by-layer Energy")
    ax_E.set_xlabel("Layer")
    ax_E.set_ylabel("Energy (MeV)")
    ax_E.legend()

    # (b) η centroid
    ax_eta.plot(layers, eta_true, "o-", label="True")
    ax_eta.plot(layers, eta_gen , "x-", label="Generated")
    ax_eta.set_title(r"$\eta_{EC}$ vs. Layer")
    ax_eta.set_xlabel("Layer")
    ax_eta.set_ylabel(r"$\eta_{EC}$")
    ax_eta.legend()

    # (c) φ centroid
    ax_phi.plot(layers, phi_true, "o-", label="True")
    ax_phi.plot(layers, phi_gen , "x-", label="Generated")
    ax_phi.set_title(r"$\phi_{EC}$ vs. Layer")
    ax_phi.set_xlabel("Layer")
    ax_phi.set_ylabel(r"$\phi_{EC}$")
    ax_phi.legend()

    # (d) Widths
    ax_w.plot(layers, w_eta_true, "o-",          label=r"True σ$_η$")
    ax_w.plot(layers, w_eta_gen , "x-",          label=r"Gen  σ$_η$")
    ax_w.plot(layers, w_phi_true, "s--",         label=r"True σ$_φ$")
    ax_w.plot(layers, w_phi_gen , "d--",         label=r"Gen  σ$_φ$")
    ax_w.set_title("Widths vs. Layer")
    ax_w.set_xlabel("Layer")
    ax_w.set_ylabel("Width")
    ax_w.legend(ncol=2)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}scalar_comparison_event{event_id}.png",
                    dpi=200, facecolor="white")
    plt.show()

def plot_global_summary_metrics(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    hlf: object,
    save_prefix: str = None
) -> None:
    """
    Compute and report (and optionally plot) global summary metrics between generated
    and true shower for a single event. Metrics include:
      - Total energy difference
      - Layer-wise MSE
      - Centroid shifts (η, φ)
      - Kolmogorov–Smirnov D statistic on the full 6480-voxel distribution

    Args:
      - full_calo_generated: tf.Tensor or np.ndarray of shape [1, 6480] (MeV)
      - cache_file:          path to HDF5 cache
      - event_id:            which event to compare
      - hlf:                 instance of HighLevelFeatures
      - save_prefix:         optional prefix for saving plots (if any)
    """
    original_shower_tf, true_inc_E = get_original_shower(cache_file, event_id)

    gen_np   = to_numpy_array(full_calo_generated)[0]   # shape (6480,)
    true_np  = to_numpy_array(original_shower_tf)[0]    # shape (6480,)

    # (A) Total energy difference
    E_tot_true = true_np.sum()
    E_tot_gen  = gen_np.sum()
    ΔE_tot     = E_tot_gen - E_tot_true

    # (B) Layer-wise MSE (45 layers, 16×9 voxels each)
    n_layers = 45
    voxels_per_layer = 16 * 9
    layer_mse = []
    for i in range(n_layers):
        start = i * voxels_per_layer
        end   = start + voxels_per_layer
        mse_i = np.mean((gen_np[start:end] - true_np[start:end])**2)
        layer_mse.append(mse_i)

    # (C) Centroid shift per layer
    original_shower_np = to_numpy_array(original_shower_tf)
    full_calo_np       = to_numpy_array(full_calo_generated)

    # Compute centroids for “true”
    hlf.CalculateFeatures(original_shower_np)
    eta_true = np.array([hlf.EC_etas[l][0] for l in sorted(hlf.EC_etas.keys())])
    phi_true = np.array([hlf.EC_phis[l][0] for l in sorted(hlf.EC_phis.keys())])

    # Compute centroids for “generated”
    hlf.CalculateFeatures(full_calo_np)
    eta_gen = np.array([hlf.EC_etas[l][0] for l in sorted(hlf.EC_etas.keys())])
    phi_gen = np.array([hlf.EC_phis[l][0] for l in sorted(hlf.EC_phis.keys())])

    centroid_shifts = np.sqrt((eta_gen - eta_true)**2 + (phi_gen - phi_true)**2)

    # (D) Kolmogorov–Smirnov on full 6480-voxel distributions
    try:
        ks_stat, ks_pval = ks_2samp(true_np, gen_np)
    except Exception:
        ks_stat, ks_pval = np.nan, np.nan

    # --- Print summary
    print(f"Event {event_id} Summary Metrics:")
    print(f"  • True E_tot       = {E_tot_true:.3f} MeV")
    print(f"  • Generated E_tot  = {E_tot_gen:.3f} MeV")
    print(f"  • ΔE_tot           = {ΔE_tot:.3f} MeV\n")

    print("  • Layer-wise MSE:")
    for i, mse_i in enumerate(layer_mse):
        pass
        #print(f"      Layer {i:2d}: MSE = {mse_i:.4e}")

    print("\n  • Centroid shifts (η, φ) per layer:")
    for i, shift_i in enumerate(centroid_shifts):
        pass
        #print(f"      Layer {i:2d}: |Δcentroid| = {shift_i:.4e}")

    print(f"\n  • KS statistic on full voxel distribution: D = {ks_stat:.4f}, p-value = {ks_pval:.4f}")

    # --- Optional plots for MSE and centroid shifts
    layers = np.arange(n_layers)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(layers, layer_mse, marker='o', color='C0', label="Layer MSE")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("MSE (MeV²)", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_title(f"Global Metrics, event {event_id}")

    ax2 = ax1.twinx()
    ax2.plot(layers, centroid_shifts, marker='x', color='C1', label="Centroid shift")
    ax2.set_ylabel(r"Centroid shift $\sqrt{Δη²+Δφ²}$", color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}global_metrics_event{event_id}.png")
    plt.show()
    plt.close(fig)

    

def _flat_to_LAR(vec_6480: np.ndarray, L=45, A=16, R=9) -> np.ndarray:
    """
    vec_6480: shape (6480,) or (1,6480). Returns (L, A, R) with R fastest.
    """
    v = vec_6480.reshape(-1)
    assert v.size == L*A*R, f"Expected 6480 elements, got {v.size}"
    return v.reshape(L, A, R)

def _angular_profile_per_layer(LAR: np.ndarray) -> np.ndarray:
    """
    Sum over radius to get an angular profile per layer.
    Returns P with shape (L, A), P[l, a] = sum_r E[l, a, r].
    """
    return LAR.sum(axis=2)

def _circular_diff(a: float, b: float) -> float:
    """Minimal signed angular difference a-b in [-pi, pi]."""
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return d

def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """
    Weighted circular mean and resultant length.
    angles: shape (A,), in radians
    weights: shape (A,), nonnegative
    Returns (mu, R) where mu in [-pi, pi], R in [0,1].
    """
    wsum = weights.sum()
    if wsum <= 0:
        return np.nan, np.nan
    z = np.sum(weights * np.exp(1j * angles))
    mu = np.angle(z)
    R = np.abs(z) / wsum
    return mu, R

def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns np.nan if degenerate."""
    x0 = x - x.mean()
    y0 = y - y.mean()
    nx = np.linalg.norm(x0)
    ny = np.linalg.norm(y0)
    if nx == 0 or ny == 0:
        return np.nan
    return float(np.dot(x0, y0) / (nx * ny))

def _best_circular_shift_corr(p: np.ndarray, q: np.ndarray) -> tuple[float, int]:
    """
    Max Pearson correlation between p and q over all circular shifts of q.
    Returns (rho_max, shift), where q is rolled by -shift to align with p.
    """
    A = p.size
    best_rho, best_s = -np.inf, 0
    for s in range(A):
        rho = _pearson_corr(p, np.roll(q, -s))
        if np.isnan(rho):
            continue
        if rho > best_rho:
            best_rho, best_s = rho, s
    if best_rho == -np.inf:
        best_rho = np.nan
    return best_rho, best_s

def compute_angular_layer_metrics(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    normalize_profiles: bool = True
) -> dict:
    """
    Compute angular continuity metrics within the generated event and (optionally)
    compare to the true event from cache_file for the same event_id.

    Returns a dict with keys:
      'gen': {
         'mu': (L,), radians
         'mu_deg': (L,), degrees
         'R': (L,), resultant length (concentration)
         'E_layer': (L,), layer energies
         'dphi_adj_abs': (L-1,), |Δφ| between adjacent layers (radians)
         'dphi_adj_abs_deg': (L-1,), degrees
         'flip_fraction_90': scalar, fraction with |Δφ| > 90°
         'profile_corr_max': (L-1,), max corr of P_l vs P_{l+1} over circular shifts
         'profile_shift_bins': (L-1,), shift (in bins) that maximizes corr
         'dispersion_delta': (L-1,), |ΔR| between adjacent layers
         'summary': {...}  # means (optionally energy-weighted)
      }
      'true': {...}  # same keys for the Geant4 event
      'gen_vs_true': {
         'mu_error_abs': (L,), |φ_gen - φ_true| (radians, circular)
         'mu_error_abs_deg': (L,), degrees
         'profile_corr_same_layer': (L,), max corr (over circular shifts) of gen vs true
         'summary': {...}
      }
    """
    # --- Load vectors, reshape to (L,A,R), build angular profiles P[l,a]
    true_tf, _ = get_original_shower(cache_file, event_id)
    gen_np  = to_numpy_array(full_calo_generated)[0]
    true_np = to_numpy_array(true_tf)[0]

    L, A, R = 45, 16, 9
    ang = 2*np.pi * np.arange(A) / A

    gen_LAR  = _flat_to_LAR(gen_np, L, A, R)
    true_LAR = _flat_to_LAR(true_np, L, A, R)

    Pgen  = _angular_profile_per_layer(gen_LAR)   # (L,A)
    Ptrue = _angular_profile_per_layer(true_LAR)  # (L,A)

    Egen  = Pgen.sum(axis=1)  # (L,)
    Etrue = Ptrue.sum(axis=1) # (L,)

    if normalize_profiles:
        Pg = Pgen / np.maximum(Egen[:,None], 1e-12)
        Pt = Ptrue / np.maximum(Etrue[:,None], 1e-12)
    else:
        Pg, Pt = Pgen.copy(), Ptrue.copy()

    # --- Per-layer circular stats
    mu_g, R_g = [], []
    mu_t, R_t = [], []
    for l in range(L):
        mu, Rr = _circular_mean(ang, Pg[l])
        mu_g.append(mu); R_g.append(Rr)
        mu, Rr = _circular_mean(ang, Pt[l])
        mu_t.append(mu); R_t.append(Rr)
    mu_g = np.array(mu_g); R_g = np.array(R_g)
    mu_t = np.array(mu_t); R_t = np.array(R_t)

    # --- Adjacent-layer continuity (generated & true)
    def continuity(mu, R, P):
        dphi = np.array([abs(_circular_diff(mu[l+1], mu[l])) for l in range(L-1)])
        flip_frac_90 = float(np.mean(dphi > (np.pi/2)))
        # Profile correlation across adjacent layers (search best circular shift)
        corr = np.zeros(L-1)
        shft = np.zeros(L-1, dtype=int)
        for l in range(L-1):
            corr[l], shft[l] = _best_circular_shift_corr(P[l], P[l+1])
        dR = np.abs(np.diff(R))
        # Summaries (both unweighted and energy-weighted)
        # Use layer energy of the *first* layer in the pair for a simple weighting choice
        w = np.ones(L-1)
        return {
            'dphi_adj_abs': dphi,
            'dphi_adj_abs_deg': np.degrees(dphi),
            'flip_fraction_90': flip_frac_90,
            'profile_corr_max': corr,
            'profile_shift_bins': shft,
            'dispersion_delta': dR,
            'summary': {
                'mean_|Δφ|_deg': float(np.nanmean(np.degrees(dphi))),
                'mean_profile_corr_max': float(np.nanmean(corr)),
                'mean_|ΔR|': float(np.nanmean(dR)),
                'flip_fraction_90': flip_frac_90
            }
        }

    cont_g = continuity(mu_g, R_g, Pg)
    cont_t = continuity(mu_t, R_t, Pt)

    # --- Per-layer gen vs true comparison
    mu_err = np.array([abs(_circular_diff(mu_g[l], mu_t[l])) for l in range(L)])
    corr_same = np.zeros(L)
    for l in range(L):
        corr_same[l], _ = _best_circular_shift_corr(Pt[l], Pg[l])

    gen_vs_true = {
        'mu_error_abs': mu_err,
        'mu_error_abs_deg': np.degrees(mu_err),
        'profile_corr_same_layer': corr_same,
        'summary': {
            'mean_|μ_gen-μ_true|_deg': float(np.nanmean(np.degrees(mu_err))),
            'mean_profile_corr_same_layer': float(np.nanmean(corr_same))
        }
    }

    result = {
        'gen': {
            'mu': mu_g, 'mu_deg': np.degrees(mu_g), 'R': R_g, 'E_layer': Egen,
            **cont_g
        },
        'true': {
            'mu': mu_t, 'mu_deg': np.degrees(mu_t), 'R': R_t, 'E_layer': Etrue,
            **cont_t
        },
        'gen_vs_true': gen_vs_true
    }
    return result

def plot_angular_continuity_metrics(
    full_calo_generated,
    cache_file: str,
    event_id: int,
    normalize_profiles: bool = True,
    save_prefix: str | None = None
) -> dict:
    """
    Wrapper that computes angular metrics and produces a compact 2x2 diagnostic plot:
      (1) φ centroid vs layer (true vs gen)
      (2) |Δφ| between adjacent layers (deg) for true vs gen
      (3) Max profile correlation between adjacent layers for true vs gen
      (4) |μ_gen - μ_true| per layer (deg)

    Returns the metrics dict from compute_angular_layer_metrics.
    """
    metrics = compute_angular_layer_metrics(
        full_calo_generated, cache_file, event_id, normalize_profiles
    )

    L = 45
    x = np.arange(L)
    xt = np.arange(L-1)

    mu_t = metrics['true']['mu_deg']
    mu_g = metrics['gen']['mu_deg']
    dphi_t = metrics['true']['dphi_adj_abs_deg']
    dphi_g = metrics['gen']['dphi_adj_abs_deg']
    corr_t = metrics['true']['profile_corr_max']
    corr_g = metrics['gen']['profile_corr_max']
    mu_err_deg = metrics['gen_vs_true']['mu_error_abs_deg']

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    # (1) φ centroid vs layer
    axs[0,0].plot(x, mu_t, marker='o', label='True φ̄ (deg)')
    axs[0,0].plot(x, mu_g, marker='x', label='Gen φ̄ (deg)')
    axs[0,0].set_title("Angular centroid per layer")
    axs[0,0].set_xlabel("Layer")
    axs[0,0].set_ylabel("φ̄ (deg)")
    axs[0,0].legend()

    # (2) |Δφ| adjacent layers
    axs[0,1].plot(xt, dphi_t, marker='o', label='True |Δφ| (deg)')
    axs[0,1].plot(xt, dphi_g, marker='x', label='Gen |Δφ| (deg)')
    axs[0,1].axhline(90, linestyle='--', label='90° threshold')
    axs[0,1].set_title("|Δφ| between adjacent layers")
    axs[0,1].set_xlabel("Layer to layer+1")
    axs[0,1].set_ylabel("|Δφ| (deg)")
    axs[0,1].legend()

    # (3) Max profile corr(adjacent)
    axs[1,0].plot(xt, corr_t, marker='o', label='True corr_max')
    axs[1,0].plot(xt, corr_g, marker='x', label='Gen corr_max')
    axs[1,0].set_title("Max correlation of angular profiles (adjacent layers)")
    axs[1,0].set_xlabel("Layer to layer+1")
    axs[1,0].set_ylabel("corr (Pearson)")
    axs[1,0].set_ylim(-0.2, 1.05)
    axs[1,0].legend()

    # (4) Per-layer gen vs true error in φ centroid
    axs[1,1].plot(x, mu_err_deg, marker='s', label='|μ_gen - μ_true| (deg)')
    axs[1,1].set_title("Per-layer centroid error (gen vs true)")
    axs[1,1].set_xlabel("Layer")
    axs[1,1].set_ylabel("deg")
    axs[1,1].legend()

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}angular_continuity_event{event_id}.png", dpi=200)
    plt.show()

    # Also print concise summaries
    print("\nAngular continuity (Generated):",
          metrics['gen']['summary'])
    print("Angular continuity (True):     ",
          metrics['true']['summary'])
    print("Gen vs True (per-layer):      ",
          metrics['gen_vs_true']['summary'])

    return metrics
