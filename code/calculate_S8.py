#!/usr/bin/env python3
"""
DMTDE S₈ Tension Analysis
Corrected version with proper CLASS parameters
"""

from classy import Class
import numpy as np
import matplotlib.pyplot as plt

# Planck 2018 best-fit parameters (Table 1 of Planck 2018 VI)
PLANCK_PARAMS = {
    'output': 'mPk',
    'h': 0.6736,                    # Hubble parameter
    'omega_b': 0.02237,             # Baryon density (physical)
    'omega_cdm': 0.1200,            # CDM density (physical)
    'A_s': 2.1e-9,                  # Scalar amplitude
    'n_s': 0.9649,                  # Spectral index
    'tau_reio': 0.0544,             # Optical depth
    'P_k_max_h/Mpc': 10.0,
    'z_max_pk': 10.0
}

# Observational constraints
PLANCK_S8 = 0.834
PLANCK_S8_ERR = 0.016
KIDS_S8 = 0.766
KIDS_S8_ERR = 0.017
DES_S8 = 0.776
DES_S8_ERR = 0.017

print("="*70)
print("DMTDE S₈ TENSION ANALYSIS")
print("="*70)
print("\nRunning CLASS with DMTDE modifications...")
print(f"Fixed parameters: zc = 1.5, σ = 0.2")
print("-"*70)

# Scan fd0 values
fd0_values = np.array([0.00, 0.05, 0.10, 0.15, 0.20, -0.05, -0.10, -0.15])
results = []

for fd0 in sorted(fd0_values):
    print(f"\nComputing: fd₀ = {fd0:+.2f}")
    
    # Setup parameters
    params = PLANCK_PARAMS.copy()
    
    if fd0 != 0.0:
        params['use_dmtde'] = 1
        params['dmtde_fd0'] = fd0
        params['dmtde_zc'] = 1.5
        params['dmtde_dlnac'] = 0.2
    else:
        params['use_dmtde'] = 0
    
    try:
        # Initialize CLASS
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        
        # Extract cosmological parameters
        sigma8 = cosmo.sigma8()
        h = cosmo.h()
        Omega_m = cosmo.Omega_m()
        Omega_b = cosmo.Omega_b()
        Omega_cdm = cosmo.Omega0_cdm()
        
        # Compute S₈ = σ₈ √(Ωₘ/0.3)
        S8 = sigma8 * np.sqrt(Omega_m / 0.3)
        
        # Growth rate at z=0
        f_z0 = cosmo.scale_independent_growth_factor_f(0.0)
        fsigma8_z0 = f_z0 * sigma8
        
        # Store results
        results.append({
            'fd0': fd0,
            'sigma8': sigma8,
            'Omega_m': Omega_m,
            'Omega_b': Omega_b,
            'Omega_cdm': Omega_cdm,
            'S8': S8,
            'fsigma8_z0': fsigma8_z0,
            'h': h
        })
        
        print(f"  σ₈ = {sigma8:.6f}")
        print(f"  S₈ = {S8:.6f}")
        print(f"  Ωₘ = {Omega_m:.6f}")
        print(f"  fσ₈(z=0) = {fsigma8_z0:.6f}")
        
        # Clean up
        cosmo.struct_cleanup()
        cosmo.empty()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Convert to arrays
fd0_arr = np.array([r['fd0'] for r in results])
sigma8_arr = np.array([r['sigma8'] for r in results])
S8_arr = np.array([r['S8'] for r in results])
Omega_m_arr = np.array([r['Omega_m'] for r in results])

# Print table
print(f"\n{'fd₀':<10} {'σ₈':<12} {'Ωₘ':<12} {'S₈':<12} {'Δσ₈/σ₈ (%)':<15} {'ΔS₈/S₈ (%)':<15}")
print("-"*70)

# Get ΛCDM reference (fd0 = 0)
idx_lcdm = np.where(fd0_arr == 0.0)[0][0]
sigma8_lcdm = sigma8_arr[idx_lcdm]
S8_lcdm = S8_arr[idx_lcdm]

for i, r in enumerate(results):
    delta_sigma8_pct = (r['sigma8'] - sigma8_lcdm) / sigma8_lcdm * 100
    delta_S8_pct = (r['S8'] - S8_lcdm) / S8_lcdm * 100
    
    print(f"{r['fd0']:<10.2f} {r['sigma8']:<12.6f} {r['Omega_m']:<12.6f} "
          f"{r['S8']:<12.6f} {delta_sigma8_pct:<15.3f} {delta_S8_pct:<15.3f}")

print("="*70)

# Observational comparison
print("\nOBSERVATIONAL CONSTRAINTS:")
print(f"  Planck 2018:  S₈ = {PLANCK_S8:.3f} ± {PLANCK_S8_ERR:.3f}")
print(f"  KiDS+VIKING:  S₈ = {KIDS_S8:.3f} ± {KIDS_S8_ERR:.3f}")
print(f"  DES Year 3:   S₈ = {DES_S8:.3f} ± {DES_S8_ERR:.3f}")
print(f"  Tension:      ΔS₈ = {PLANCK_S8 - KIDS_S8:.3f} ({(PLANCK_S8-KIDS_S8)/KIDS_S8*100:.1f}%)")

# Find best fits
idx_planck = np.argmin(np.abs(S8_arr - PLANCK_S8))
idx_kids = np.argmin(np.abs(S8_arr - KIDS_S8))

print("\nBEST FITS:")
print(f"  Planck S₈: fd₀ = {fd0_arr[idx_planck]:+.2f}, "
      f"S₈ = {S8_arr[idx_planck]:.6f} (Δ = {(S8_arr[idx_planck]-PLANCK_S8)*1000:+.2f}×10⁻³)")
print(f"  KiDS S₈:   fd₀ = {fd0_arr[idx_kids]:+.2f}, "
      f"S₈ = {S8_arr[idx_kids]:.6f} (Δ = {(S8_arr[idx_kids]-KIDS_S8)*1000:+.2f}×10⁻³)")

# Check if tension can be resolved
S8_range = S8_arr.max() - S8_arr.min()
tension_gap = PLANCK_S8 - KIDS_S8
resolution_pct = (S8_range / tension_gap) * 100

print(f"\nTENSION RESOLUTION:")
print(f"  DMTDE range:  ΔS₈ = {S8_range:.4f}")
print(f"  Required gap: ΔS₈ = {tension_gap:.4f}")
print(f"  Achievable:   {resolution_pct:.1f}%")

if resolution_pct < 100:
    print(f"\n  ⚠️  DMTDE CANNOT RESOLVE THE TENSION")
    print(f"     (Can only achieve {resolution_pct:.1f}% of required shift)")
else:
    print(f"\n  ✓  DMTDE CAN POTENTIALLY RESOLVE THE TENSION")

print("="*70)

# ==================== PLOTTING ====================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: σ₈ vs fd₀
ax = axes[0, 0]
ax.plot(fd0_arr, sigma8_arr, 'o-', linewidth=2, markersize=8, label='DMTDE')
ax.axhline(sigma8_lcdm, color='gray', linestyle=':', label='ΛCDM')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('$f_{d0}$ (DM-DE coupling)', fontsize=12)
ax.set_ylabel('$\\sigma_8$', fontsize=12)
ax.set_title('Linear Growth Amplitude', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: S₈ vs fd₀
ax = axes[0, 1]
ax.plot(fd0_arr, S8_arr, 'o-', linewidth=2, markersize=8, label='DMTDE', color='blue')
ax.axhline(PLANCK_S8, color='red', linestyle='--', label='Planck')
ax.fill_between([-0.2, 0.25], 
                 PLANCK_S8 - PLANCK_S8_ERR,
                 PLANCK_S8 + PLANCK_S8_ERR,
                 color='red', alpha=0.2)
ax.axhline(KIDS_S8, color='green', linestyle='--', label='KiDS')
ax.fill_between([-0.2, 0.25], 
                 KIDS_S8 - KIDS_S8_ERR,
                 KIDS_S8 + KIDS_S8_ERR,
                 color='green', alpha=0.2)
ax.axhline(DES_S8, color='orange', linestyle='--', label='DES', alpha=0.7)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('$f_{d0}$ (DM-DE coupling)', fontsize=12)
ax.set_ylabel('$S_8 = \\sigma_8 \\sqrt{\\Omega_m/0.3}$', fontsize=12)
ax.set_title('$S_8$ Tension', fontsize=13, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_xlim(-0.2, 0.25)

# Plot 3: Fractional change
ax = axes[1, 0]
delta_sigma8_pct = (sigma8_arr - sigma8_lcdm) / sigma8_lcdm * 100
ax.plot(fd0_arr, delta_sigma8_pct, 'o-', linewidth=2, markersize=8, color='purple')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.fill_between([-0.2, 0.25], -3, 3, color='yellow', alpha=0.2, 
                label='No-go bound: |Δσ₈/σ₈| ≲ 3%')
ax.set_xlabel('$f_{d0}$', fontsize=12)
ax.set_ylabel('$\\Delta\\sigma_8 / \\sigma_8$ (%)', fontsize=12)
ax.set_title('Fractional Change in $\\sigma_8$', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(-0.2, 0.25)

# Plot 4: Linear fit
ax = axes[1, 1]
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(fd0_arr, sigma8_arr)

ax.scatter(fd0_arr, sigma8_arr, s=100, c='blue', zorder=3, label='Data')
fd0_fit = np.linspace(-0.2, 0.25, 100)
sigma8_fit = intercept + slope * fd0_fit
ax.plot(fd0_fit, sigma8_fit, 'r--', lw=2, 
        label=f'Fit: $\\sigma_8 = {intercept:.4f} + {slope:.3f}f_{{d0}}$')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('$f_{d0}$', fontsize=12)
ax.set_ylabel('$\\sigma_8$', fontsize=12)
ax.set_title(f'Lock Constant: $R^2 = {r_value**2:.6f}$', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Slope: {slope:.4f}\nIntercept: {intercept:.4f}', 
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('S8_tension_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: S8_tension_analysis.png")

# Save results to file
np.savez('S8_tension_results.npz',
         fd0=fd0_arr,
         sigma8=sigma8_arr,
         S8=S8_arr,
         Omega_m=Omega_m_arr,
         planck_S8=PLANCK_S8,
         kids_S8=KIDS_S8)
print("✓ Results saved: S8_tension_results.npz")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)