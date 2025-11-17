#!/usr/bin/env python3
"""
Plot results from zc scan
"""

import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('zc_scan_results.npz')

zc_arr = np.array(data['zc'])
fd0_arr = np.array(data['fd0'])
sigma8_arr = np.array(data['sigma8'])
S8_arr = np.array(data['S8'])

# Observational constraints
planck_sigma8 = 0.8120
planck_sigma8_err = 0.0073
planck_S8 = 0.834
planck_S8_err = 0.016

kids_S8 = 0.766
kids_S8_err = 0.017

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: σ₈ vs zc (fixed fd0 = 0.10)
ax = axes[0, 0]
mask = np.abs(fd0_arr - 0.10) < 0.01
zc_plot = zc_arr[mask]
sigma8_plot = sigma8_arr[mask]

ax.plot(zc_plot, sigma8_plot, 'o-', linewidth=2, markersize=8, label='DMTDE')
ax.axhline(planck_sigma8, color='red', linestyle='--', label='Planck')
ax.fill_between([0, 3.5], 
                 planck_sigma8 - planck_sigma8_err,
                 planck_sigma8 + planck_sigma8_err,
                 color='red', alpha=0.2)
ax.set_xlabel('Transition Redshift $z_c$', fontsize=12)
ax.set_ylabel('$\\sigma_8$', fontsize=12)
ax.set_title('$\\sigma_8$ vs $z_c$ (fixed $f_{d0}=0.10$)', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: S₈ vs zc (fixed fd0 = 0.10)
ax = axes[0, 1]
S8_plot = S8_arr[mask]

ax.plot(zc_plot, S8_plot, 'o-', linewidth=2, markersize=8, label='DMTDE')
ax.axhline(planck_S8, color='red', linestyle='--', label='Planck')
ax.fill_between([0, 3.5], 
                 planck_S8 - planck_S8_err,
                 planck_S8 + planck_S8_err,
                 color='red', alpha=0.2)
ax.axhline(kids_S8, color='blue', linestyle='--', label='KiDS')
ax.fill_between([0, 3.5], 
                 kids_S8 - kids_S8_err,
                 kids_S8 + kids_S8_err,
                 color='blue', alpha=0.2)
ax.set_xlabel('Transition Redshift $z_c$', fontsize=12)
ax.set_ylabel('$S_8$', fontsize=12)
ax.set_title('$S_8$ vs $z_c$ (fixed $f_{d0}=0.10$)', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: σ₈ vs fd0 (fixed zc = 1.5)
ax = axes[1, 0]
mask = np.abs(zc_arr - 1.5) < 0.01
fd0_plot = fd0_arr[mask]
sigma8_plot = sigma8_arr[mask]

# Sort by fd0
sort_idx = np.argsort(fd0_plot)
fd0_plot = fd0_plot[sort_idx]
sigma8_plot = sigma8_plot[sort_idx]

ax.plot(fd0_plot, sigma8_plot, 'o-', linewidth=2, markersize=8, label='DMTDE')
ax.axhline(planck_sigma8, color='red', linestyle='--', label='Planck')
ax.fill_between([-0.2, 0.2], 
                 planck_sigma8 - planck_sigma8_err,
                 planck_sigma8 + planck_sigma8_err,
                 color='red', alpha=0.2)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('$f_{d0}$ (DM-DE coupling)', fontsize=12)
ax.set_ylabel('$\\sigma_8$', fontsize=12)
ax.set_title('$\\sigma_8$ vs $f_{d0}$ (fixed $z_c=1.5$)', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: S₈ vs fd0 (fixed zc = 1.5)
ax = axes[1, 1]
S8_plot = S8_arr[mask][sort_idx]

ax.plot(fd0_plot, S8_plot, 'o-', linewidth=2, markersize=8, label='DMTDE')
ax.axhline(planck_S8, color='red', linestyle='--', label='Planck')
ax.fill_between([-0.2, 0.2], 
                 planck_S8 - planck_S8_err,
                 planck_S8 + planck_S8_err,
                 color='red', alpha=0.2)
ax.axhline(kids_S8, color='blue', linestyle='--', label='KiDS')
ax.fill_between([-0.2, 0.2], 
                 kids_S8 - kids_S8_err,
                 kids_S8 + kids_S8_err,
                 color='blue', alpha=0.2)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('$f_{d0}$ (DM-DE coupling)', fontsize=12)
ax.set_ylabel('$S_8$', fontsize=12)
ax.set_title('$S_8$ vs $f_{d0}$ (fixed $z_c=1.5$)', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('zc_scan_results.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: zc_scan_results.png")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nTest 1: zc scan (fd0 = 0.10)")
mask = np.abs(fd0_arr - 0.10) < 0.01
for i in np.where(mask)[0]:
    delta_sigma8 = (sigma8_arr[i] - planck_sigma8) / planck_sigma8 * 100
    delta_S8 = (S8_arr[i] - planck_S8) / planck_S8 * 100
    print(f"  zc = {zc_arr[i]:.1f}: σ₈ = {sigma8_arr[i]:.4f} ({delta_sigma8:+.2f}%), "
          f"S₈ = {S8_arr[i]:.4f} ({delta_S8:+.2f}%)")

print("\nTest 2: fd0 scan (zc = 1.5)")
mask = np.abs(zc_arr - 1.5) < 0.01
sort_idx = np.argsort(fd0_arr[mask])
for i in np.where(mask)[0][sort_idx]:
    delta_sigma8 = (sigma8_arr[i] - planck_sigma8) / planck_sigma8 * 100
    delta_S8 = (S8_arr[i] - planck_S8) / planck_S8 * 100
    print(f"  fd0 = {fd0_arr[i]:+.2f}: σ₈ = {sigma8_arr[i]:.4f} ({delta_sigma8:+.2f}%), "
          f"S₈ = {S8_arr[i]:.4f} ({delta_S8:+.2f}%)")

print("\nBest fit to Planck σ₈:")
idx_best = np.argmin(np.abs(sigma8_arr - planck_sigma8))
print(f"  zc = {zc_arr[idx_best]:.1f}, fd0 = {fd0_arr[idx_best]:+.2f}")
print(f"  σ₈ = {sigma8_arr[idx_best]:.4f}, S₈ = {S8_arr[idx_best]:.4f}")

print("\nBest fit to KiDS S₈:")
idx_best = np.argmin(np.abs(S8_arr - kids_S8))
print(f"  zc = {zc_arr[idx_best]:.1f}, fd0 = {fd0_arr[idx_best]:+.2f}")
print(f"  σ₈ = {sigma8_arr[idx_best]:.4f}, S₈ = {S8_arr[idx_best]:.4f}")

# Analyze zc dependence
print("\n" + "="*60)
print("ZC DEPENDENCE ANALYSIS")
print("="*60)

mask = np.abs(fd0_arr - 0.10) < 0.01
zc_scan = zc_arr[mask]
sigma8_scan = sigma8_arr[mask]

# Linear fit
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(zc_scan, sigma8_scan)

print(f"\nLinear fit: σ₈ = {intercept:.4f} + {slope:.4f} × zc")
print(f"R² = {r_value**2:.6f}")
print(f"p-value = {p_value:.2e}")

# Compare with fd0 sensitivity
mask_fd0 = np.abs(zc_arr - 1.5) < 0.01
fd0_scan = fd0_arr[mask_fd0]
sigma8_fd0_scan = sigma8_arr[mask_fd0]
sort_idx = np.argsort(fd0_scan)
fd0_scan = fd0_scan[sort_idx]
sigma8_fd0_scan = sigma8_fd0_scan[sort_idx]

slope_fd0, intercept_fd0, r_fd0, p_fd0, std_fd0 = linregress(fd0_scan, sigma8_fd0_scan)

print(f"\nFor comparison, fd0 sensitivity:")
print(f"Linear fit: σ₈ = {intercept_fd0:.4f} + {slope_fd0:.4f} × fd0")
print(f"R² = {r_fd0**2:.6f}")

print(f"\nSensitivity ratio: |∂σ₈/∂fd0| / |∂σ₈/∂zc| = {abs(slope_fd0/slope):.1f}:1")

print("\n" + "="*60)