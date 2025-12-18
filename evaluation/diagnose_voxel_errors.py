"""
Diagnostic to find which specific voxels have errors.

This will show you WHERE your model is making mistakes in the 648-dimensional
coarse calorimeter representation.
"""

import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from pathlib import Path

# Try to load your actual data
print("="*70)
print("PER-VOXEL ERROR ANALYSIS")
print("="*70)

try:
    import h5py
    
    print("\nLoading data from eval_results_A.hdf5...")
    results_file = "/teo_fs_fast/users/acosso/Dataset/eval_results_A.hdf5"
    
    with h5py.File(results_file, "r") as f:
        # Load just enough data for diagnostics
        n_events = 10000
        n_rows_per_event = 648  # coarse voxels per event
        n_rows = n_events * n_rows_per_event
        
        truth_fine = f["/truth_E"][:n_rows, :]  # [n_rows, 10]
        model_fine = f["/model_E"][:n_rows, :]  # [n_rows, 10]
        
        # Sum to coarse representation
        truth_coarse_flat = truth_fine.sum(axis=1)  # [n_rows]
        model_coarse_flat = model_fine.sum(axis=1)  # [n_rows]
        
        # Reshape to [n_events, 648]
        truth_coarse = truth_coarse_flat.reshape(n_events, 648)
        model_coarse = model_coarse_flat.reshape(n_events, 648)
    
    print(f"✓ Loaded {n_events} events")
    print(f"  Shape: {truth_coarse.shape}")
    print(f"  Truth range: [{truth_coarse.min():.4f}, {truth_coarse.max():.4f}]")
    print(f"  Model range: [{model_coarse.min():.4f}, {model_coarse.max():.4f}]")
    
    # Compute per-voxel KS tests
    print("\nRunning per-voxel KS tests (648 tests)...")
    ks_stats = []
    ks_pvals = []
    mean_diffs = []
    std_ratios = []
    
    for d in range(648):
        stat, pval = ks_2samp(truth_coarse[:, d], model_coarse[:, d])
        ks_stats.append(stat)
        ks_pvals.append(pval)
        
        # Also compute basic statistics
        truth_mean = truth_coarse[:, d].mean()
        model_mean = model_coarse[:, d].mean()
        truth_std = truth_coarse[:, d].std()
        model_std = model_coarse[:, d].std()
        
        mean_diffs.append(model_mean - truth_mean)
        std_ratios.append(model_std / truth_std if truth_std > 0 else 1.0)
    
    ks_stats = np.array(ks_stats)
    ks_pvals = np.array(ks_pvals)
    mean_diffs = np.array(mean_diffs)
    std_ratios = np.array(std_ratios)
    
    print("✓ Done")
    
    # Analyze results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    
    significant_01 = (ks_pvals < 0.01).sum()
    significant_05 = (ks_pvals < 0.05).sum()
    
    print(f"\nStatistically significant differences:")
    print(f"  p < 0.01: {significant_01}/{648} voxels ({100*significant_01/648:.1f}%)")
    print(f"  p < 0.05: {significant_05}/{648} voxels ({100*significant_05/648:.1f}%)")
    
    print(f"\nMean differences (model - truth):")
    print(f"  Max positive: +{mean_diffs.max():.6f}")
    print(f"  Max negative: {mean_diffs.min():.6f}")
    print(f"  Mean |diff|:  {np.abs(mean_diffs).mean():.6f}")
    print(f"  Median |diff|: {np.median(np.abs(mean_diffs)):.6f}")
    
    print(f"\nStd ratio (model / truth):")
    print(f"  Max:    {std_ratios.max():.4f}")
    print(f"  Min:    {std_ratios.min():.4f}")
    print(f"  Mean:   {std_ratios.mean():.4f}")
    print(f"  Median: {np.median(std_ratios):.4f}")
    
    # Find worst voxels
    print("\n" + "-"*70)
    print("TOP 10 WORST VOXELS (by KS statistic)")
    print("-"*70)
    
    worst_indices = np.argsort(ks_stats)[::-1][:10]
    
    for i, idx in enumerate(worst_indices):
        print(f"\n{i+1}. Voxel {idx}:")
        print(f"   KS stat:    {ks_stats[idx]:.6f}")
        print(f"   p-value:    {ks_pvals[idx]:.2e}")
        print(f"   Mean diff:  {mean_diffs[idx]:+.6f}")
        print(f"   Std ratio:  {std_ratios[idx]:.4f}")
        print(f"   Truth mean: {truth_coarse[:, idx].mean():.6f}")
        print(f"   Model mean: {model_coarse[:, idx].mean():.6f}")
    
    # Create visualization
    print("\n" + "-"*70)
    print("CREATING VISUALIZATIONS")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. P-values across voxels
    ax = axes[0, 0]
    ax.bar(range(648), -np.log10(ks_pvals + 1e-100), width=1.0, edgecolor='none')
    ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', linewidth=2, label='p=0.05')
    ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=2, label='p=0.01')
    ax.set_xlabel('Voxel index', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title('Per-Voxel KS Test Significance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mean differences
    ax = axes[0, 1]
    colors = ['red' if p < 0.01 else 'gray' for p in ks_pvals]
    ax.bar(range(648), mean_diffs, width=1.0, color=colors, edgecolor='none', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Voxel index', fontsize=12)
    ax.set_ylabel('Mean difference (model - truth)', fontsize=12)
    ax.set_title('Per-Voxel Mean Differences (red = p<0.01)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. KS statistic distribution
    ax = axes[1, 0]
    ax.hist(ks_stats, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=ks_stats.mean(), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {ks_stats.mean():.4f}')
    ax.set_xlabel('KS statistic', fontsize=12)
    ax.set_ylabel('Number of voxels', fontsize=12)
    ax.set_title('Distribution of KS Statistics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Scatter: mean diff vs KS stat
    ax = axes[1, 1]
    scatter = ax.scatter(mean_diffs, ks_stats, c=-np.log10(ks_pvals + 1e-100), 
                        cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Mean difference (model - truth)', fontsize=12)
    ax.set_ylabel('KS statistic', fontsize=12)
    ax.set_title('KS Statistic vs Mean Difference', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log₁₀(p-value)', fontsize=10)
    
    plt.tight_layout()
    
    output_dir = Path("./53")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "voxel_error_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    
    # Create histograms of worst voxels
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(worst_indices):
        ax = axes[i]
        ax.hist(truth_coarse[:, idx], bins=50, alpha=0.5, label='Truth', 
               density=True, edgecolor='black')
        ax.hist(model_coarse[:, idx], bins=50, alpha=0.5, label='Model', 
               density=True, edgecolor='black')
        ax.set_title(f'Voxel {idx} (p={ks_pvals[idx]:.2e})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Energy', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = output_dir / "worst_voxels_distributions.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved worst voxels to: {output_file2}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"""
Your results show that {significant_01} out of 648 voxels ({100*significant_01/648:.1f}%) 
have statistically significant differences (p < 0.01).

This explains why:
  ✓ KS Test works:  Tests each voxel independently → sees all {significant_01} errors
  ✗ SKS/SWD fails:  200 random projections dilute these {significant_01} errors 
                    among all 648 voxels

WHY THIS HAPPENS:
━━━━━━━━━━━━━━━━━
Each random projection in SKS/SWD is:
  projection = w₁×voxel₁ + w₂×voxel₂ + ... + w₆₄₈×voxel₆₄₈

Where weights wᵢ ≈ 1/√648 ≈ 0.04

With ~{significant_01} error voxels out of 648:
  - Error contribution:  {significant_01} × 0.04 × (error) ≈ {significant_01*0.04:.1f} × (error)
  - Correct contribution: {648-significant_01} × 0.04 × (correct) ≈ {(648-significant_01)*0.04:.1f} × (correct)

The errors get DILUTED by the correct voxels!

SOLUTIONS:
━━━━━━━━━━
1. Increase nslices from 200 to 1000-5000
   → More projections = higher chance of aligning with error modes

2. Accept that KS is simply better for sparse errors
   → This is the RIGHT result! Your model has localized issues.

3. Investigate WHY these specific voxels have errors
   → Detector geometry? Energy range? Physics-based explanation?
    """)
    
except Exception as e:
    print(f"\n⚠️ Could not load data: {e}")
    print("\nThis diagnostic requires:")
    print("  - eval_results_A.hdf5 at /teo_fs_fast/users/acosso/Dataset/")
    print("  - scipy, matplotlib installed")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DONE")
print("="*70)
