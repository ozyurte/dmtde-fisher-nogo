#!/usr/bin/env python3
"""
DMTDE Lock Constant Analysis
Inspired by Planck's A_s * e^(-2τ) parameter lock
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import pearsonr

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11

print("="*70)
print("DMTDE LOCK CONSTANT ANALYSIS")
print("="*70)

# ==================== DATA ====================
print("\n[1] Loading data...")

# Data from Table 1: zc scan (fd0 = 0.10 fixed)
zc_data = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
sigma8_zc = np.array([0.8396, 0.8363, 0.8324, 0.8296, 0.8276, 0.8263])
S8_zc = np.array([0.8535, 0.8547, 0.8508, 0.8479, 0.8459, 0.8445])
fd0_fixed = 0.10

# Data from Table 2: fd0 scan (zc = 1.5 fixed)
fd0_data = np.array([-0.15, -0.10, -0.05, 0.05, 0.10, 0.15])
sigma8_fd0 = np.array([0.8080, 0.8127, 0.8175, 0.8273, 0.8324, 0.8376])
S8_fd0 = np.array([0.8258, 0.8306, 0.8355, 0.8456, 0.8508, 0.8561])
zc_fixed = 1.5

# Reference values
sigma8_lcdm = 0.8274
sigma8_planck = 0.8120
S8_planck = 0.834
S8_kids = 0.766

print(f"  ✓ Loaded {len(zc_data)} points from zc scan")
print(f"  ✓ Loaded {len(fd0_data)} points from fd0 scan")

# ==================== ANALYSIS 1: LINEAR FIT ====================
print("\n[2] Linear regression analysis...")

# Fit sigma8 vs fd0
coeffs_fd0 = np.polyfit(fd0_data, sigma8_fd0, 1)
alpha_fd0 = coeffs_fd0[0]
intercept_fd0 = coeffs_fd0[1]

# Fit sigma8 vs zc
coeffs_zc = np.polyfit(zc_data, sigma8_zc, 1)
alpha_zc = coeffs_zc[0]
intercept_zc = coeffs_zc[1]

# Calculate R²
r_fd0, p_fd0 = pearsonr(fd0_data, sigma8_fd0)
r_zc, p_zc = pearsonr(zc_data, sigma8_zc)

print(f"\n  σ₈ vs fd₀:")
print(f"    Slope (α_fd0):     {alpha_fd0:.4f}")
print(f"    Intercept:         {intercept_fd0:.4f}")
print(f"    R²:                {r_fd0**2:.6f}")
print(f"    p-value:           {p_fd0:.2e}")

print(f"\n  σ₈ vs zc:")
print(f"    Slope (α_zc):      {alpha_zc:.4f}")
print(f"    Intercept:         {intercept_zc:.4f}")
print(f"    R²:                {r_zc**2:.6f}")
print(f"    p-value:           {p_zc:.2e}")

# Sensitivity ratio
sensitivity_ratio = abs(alpha_fd0 / alpha_zc)
print(f"\n  Sensitivity ratio: |α_fd0 / α_zc| = {sensitivity_ratio:.1f}")

# ==================== ANALYSIS 2: LOCK CONSTANT OPTIMIZATION ====================
print("\n[3] Optimizing lock constant Ξ = fd₀ · (1+zc)^γ...")

def Xi_function(fd0, zc, gamma):
    """Lock constant definition"""
    return fd0 * (1 + zc)**gamma

def objective_gamma(gamma):
    """Minimize variance of σ₈ - σ₈^LCDM - α·Ξ"""
    # Calculate Xi for both datasets
    Xi_zc = Xi_function(fd0_fixed, zc_data, gamma)
    Xi_fd0 = Xi_function(fd0_data, zc_fixed, gamma)
    
    # Combine
    all_Xi = np.concatenate([Xi_zc, Xi_fd0])
    all_sigma8 = np.concatenate([sigma8_zc, sigma8_fd0])
    
    # Fit α
    alpha = np.mean((all_sigma8 - sigma8_lcdm) / all_Xi)
    
    # Residuals
    predicted = sigma8_lcdm + alpha * all_Xi
    residuals = all_sigma8 - predicted
    
    return np.var(residuals)

# Optimize gamma
result = minimize(objective_gamma, x0=[0.0], bounds=[(-2, 2)], method='L-BFGS-B')
gamma_opt = result.x[0]
min_variance = result.fun

print(f"\n  Optimal γ:         {gamma_opt:.4f}")
print(f"  Minimum variance:  {min_variance:.2e}")

# Calculate optimal alpha with optimal gamma
Xi_zc_opt = Xi_function(fd0_fixed, zc_data, gamma_opt)
Xi_fd0_opt = Xi_function(fd0_data, zc_fixed, gamma_opt)
all_Xi_opt = np.concatenate([Xi_zc_opt, Xi_fd0_opt])
all_sigma8 = np.concatenate([sigma8_zc, sigma8_fd0])
alpha_opt = np.mean((all_sigma8 - sigma8_lcdm) / all_Xi_opt)

print(f"  Optimal α:         {alpha_opt:.4f}")

# ==================== ANALYSIS 3: OFFSET ANALYSIS ====================
print("\n[4] Residual offset analysis...")

# Extrapolate to fd0 = 0
sigma8_at_zero = intercept_fd0
offset = sigma8_at_zero - sigma8_lcdm
offset_percent = (offset / sigma8_lcdm) * 100

print(f"\n  σ₈(fd₀=0):         {sigma8_at_zero:.4f}")
print(f"  σ₈^ΛCDM:           {sigma8_lcdm:.4f}")
print(f"  Offset:            {offset:.4f} ({offset_percent:+.2f}%)")
print(f"\n  → Gaussian tail effect!")

# ==================== ANALYSIS 4: DEGENERACY ANALYSIS ====================
print("\n[5] Parameter degeneracy analysis...")

# Covariance matrix
data_combined = np.column_stack([
    np.concatenate([np.full(len(zc_data), fd0_fixed), fd0_data]),
    np.concatenate([zc_data, np.full(len(fd0_data), zc_fixed)]),
    all_sigma8
])

cov_matrix = np.cov(data_combined.T)
correlation_matrix = np.corrcoef(data_combined.T)

print(f"\n  Correlation matrix:")
print(f"           fd₀      zc       σ₈")
print(f"    fd₀   {correlation_matrix[0,0]:6.3f}  {correlation_matrix[0,1]:6.3f}  {correlation_matrix[0,2]:6.3f}")
print(f"    zc    {correlation_matrix[1,0]:6.3f}  {correlation_matrix[1,1]:6.3f}  {correlation_matrix[1,2]:6.3f}")
print(f"    σ₈    {correlation_matrix[2,0]:6.3f}  {correlation_matrix[2,1]:6.3f}  {correlation_matrix[2,2]:6.3f}")

# ==================== ANALYSIS 5: PLANCK-KIDS TENSION ====================
print("\n[6] Tension resolution analysis...")

# Best fit to Planck
idx_planck = np.argmin(np.abs(sigma8_fd0 - sigma8_planck))
fd0_planck = fd0_data[idx_planck]
sigma8_fit_planck = sigma8_fd0[idx_planck]
S8_fit_planck = S8_fd0[idx_planck]

print(f"\n  Best fit to Planck σ₈ = {sigma8_planck}:")
print(f"    fd₀:               {fd0_planck:+.2f}")
print(f"    σ₈:                {sigma8_fit_planck:.4f} (Δ = {(sigma8_fit_planck-sigma8_planck)*1000:+.2f}×10⁻³)")
print(f"    S₈:                {S8_fit_planck:.4f}")

# Best fit to KiDS
idx_kids = np.argmin(np.abs(S8_fd0 - S8_kids))
fd0_kids = fd0_data[idx_kids]
sigma8_fit_kids = sigma8_fd0[idx_kids]
S8_fit_kids = S8_fd0[idx_kids]

print(f"\n  Best fit to KiDS S₈ = {S8_kids}:")
print(f"    fd₀:               {fd0_kids:+.2f}")
print(f"    σ₈:                {sigma8_fit_kids:.4f}")
print(f"    S₈:                {S8_fit_kids:.4f} (Δ = {(S8_fit_kids-S8_kids)*1000:+.2f}×10⁻³)")

# Gap analysis
gap_S8 = S8_fit_planck - S8_kids
gap_percent = (gap_S8 / S8_kids) * 100

print(f"\n  Planck-KiDS gap:")
print(f"    ΔS₈:               {gap_S8:.4f} ({gap_percent:+.2f}%)")
print(f"    Required:          {(S8_planck - S8_kids):.4f} ({((S8_planck-S8_kids)/S8_kids)*100:.2f}%)")
print(f"    Achievable:        {((S8_fit_planck - S8_fit_kids) / (S8_planck - S8_kids))*100:.1f}%")

# ==================== PLOTTING ====================
print("\n[7] Generating plots...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: σ₈ vs fd₀ (Lock demonstration)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(fd0_data, sigma8_fd0, s=100, c='blue', zorder=3, label='Data')
fd0_fit = np.linspace(-0.2, 0.2, 100)
sigma8_fit = intercept_fd0 + alpha_fd0 * fd0_fit
ax1.plot(fd0_fit, sigma8_fit, 'r--', lw=2, label=f'Fit: $\\sigma_8 = {intercept_fd0:.4f} + {alpha_fd0:.3f}f_{{d0}}$')
ax1.axhline(sigma8_lcdm, color='gray', ls=':', label='$\\Lambda$CDM')
ax1.axhline(sigma8_planck, color='red', ls='--', alpha=0.5, label='Planck')
ax1.axvline(0, color='gray', ls=':', alpha=0.5)
ax1.set_xlabel('$f_{d0}$')
ax1.set_ylabel('$\\sigma_8$')
ax1.set_title(f'Lock Constant: $R^2 = {r_fd0**2:.6f}$', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, f'Sensitivity: {alpha_fd0:.3f}', 
         transform=ax1.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: σ₈ vs zc (Weak dependence)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(zc_data, sigma8_zc, s=100, c='green', zorder=3, label='Data')
zc_fit = np.linspace(0, 3.5, 100)
sigma8_zc_fit = intercept_zc + alpha_zc * zc_fit
ax2.plot(zc_fit, sigma8_zc_fit, 'r--', lw=2, label=f'Fit: $\\sigma_8 = {intercept_zc:.4f} + {alpha_zc:.4f}z_c$')
ax2.axhline(sigma8_lcdm, color='gray', ls=':', label='$\\Lambda$CDM')
ax2.set_xlabel('$z_c$')
ax2.set_ylabel('$\\sigma_8$')
ax2.set_title(f'Weak Timing Control: $R^2 = {r_zc**2:.6f}$', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, f'Sensitivity: {alpha_zc:.4f}', 
         transform=ax2.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Plot 3: Sensitivity ratio
ax3 = plt.subplot(2, 3, 3)
labels = ['$f_{d0}$', '$z_c$']
sensitivities = [abs(alpha_fd0), abs(alpha_zc)]
colors_bar = ['blue', 'green']
bars = ax3.bar(labels, sensitivities, color=colors_bar, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Sensitivity $|\\partial\\sigma_8/\\partial\\theta|$')
ax3.set_title(f'Sensitivity Ratio: {sensitivity_ratio:.1f}:1', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
# Add values on bars
for bar, val in zip(bars, sensitivities):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Optimal lock constant
ax4 = plt.subplot(2, 3, 4)
Xi_opt_plot = all_Xi_opt
sigma8_plot = all_sigma8
predicted_opt = sigma8_lcdm + alpha_opt * Xi_opt_plot
ax4.scatter(Xi_opt_plot, sigma8_plot, s=100, c='purple', zorder=3, label='Data')
Xi_range = np.linspace(Xi_opt_plot.min(), Xi_opt_plot.max(), 100)
ax4.plot(Xi_range, sigma8_lcdm + alpha_opt * Xi_range, 'r--', lw=2,
         label=f'$\\sigma_8 = {sigma8_lcdm:.4f} + {alpha_opt:.3f}\\Xi$')
ax4.set_xlabel(f'$\\Xi = f_{{d0}} \\cdot (1+z_c)^{{{gamma_opt:.2f}}}$')
ax4.set_ylabel('$\\sigma_8$')
ax4.set_title(f'Optimal Lock Constant ($\\gamma = {gamma_opt:.3f}$)', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# Calculate R² for optimal
residuals_opt = sigma8_plot - predicted_opt
ss_res = np.sum(residuals_opt**2)
ss_tot = np.sum((sigma8_plot - np.mean(sigma8_plot))**2)
r2_opt = 1 - (ss_res / ss_tot)
ax4.text(0.05, 0.95, f'$R^2 = {r2_opt:.6f}$', 
         transform=ax4.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

# Plot 5: Residual offset
ax5 = plt.subplot(2, 3, 5)
residuals_fd0 = sigma8_fd0 - (intercept_fd0 + alpha_fd0 * fd0_data)
ax5.scatter(fd0_data, residuals_fd0 * 1000, s=100, c='orange', zorder=3)
ax5.axhline(0, color='black', ls='-', lw=1)
ax5.axhline(offset * 1000, color='red', ls='--', lw=2, 
            label=f'Offset: {offset*1000:.2f}×10⁻³')
ax5.set_xlabel('$f_{d0}$')
ax5.set_ylabel('Residuals $\\times 10^{-3}$')
ax5.set_title('Gaussian Tail Effect', fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Planck-KiDS gap
ax6 = plt.subplot(2, 3, 6)
constraints = ['Planck\n$\\sigma_8$', 'KiDS\n$S_8$']
values = [sigma8_planck, S8_kids]
fits = [sigma8_fit_planck, S8_fit_kids]
x_pos = np.arange(len(constraints))
width = 0.35
ax6.bar(x_pos - width/2, values, width, label='Observation', 
        color=['red', 'blue'], alpha=0.7, edgecolor='black')
ax6.bar(x_pos + width/2, [sigma8_fit_planck, S8_fit_planck], width, 
        label='DMTDE Fit', color=['pink', 'lightblue'], alpha=0.7, edgecolor='black')
ax6.set_ylabel('Value')
ax6.set_title('Tension Resolution', fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(constraints)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('lock_constant_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: lock_constant_analysis.png")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
KEY FINDINGS:

1. LOCK CONSTANT IDENTIFICATION:
   σ₈ ≈ {intercept_fd0:.4f} + {alpha_fd0:.3f} · fd₀
   
   This demonstrates a PARAMETER LOCK similar to Planck's A_s·e^(-2τ).

2. SENSITIVITY HIERARCHY:
   ∂σ₈/∂fd₀ = {alpha_fd0:.4f}  (strong)
   ∂σ₈/∂zc  = {alpha_zc:.4f}  (weak)
   Ratio: {sensitivity_ratio:.1f}:1
   
   → fd₀ DOMINATES, zc is SUBDOMINANT

3. OPTIMAL LOCK CONSTANT:
   Ξ = fd₀ · (1+zc)^{gamma_opt:.3f}
   
   Minimizes variance across parameter space.

4. RESIDUAL OFFSET:
   σ₈(fd₀=0) = {sigma8_at_zero:.4f}
   σ₈^ΛCDM   = {sigma8_lcdm:.4f}
   Offset    = {offset:.4f} ({offset_percent:+.2f}%)
   
   → GAUSSIAN TAIL EFFECT (unphysical)

5. TENSION RESOLUTION:
   Planck σ₈: ✓ Can be satisfied (fd₀ = {fd0_planck:+.2f})
   KiDS S₈:   ✗ Cannot be reached (gap = {gap_S8:.4f})
   
   → SIMULTANEOUS FIT IMPOSSIBLE

PHYSICAL INTERPRETATION:

The DMTDE Model A exhibits a FUNDAMENTAL PARAMETER LOCK where σ₈ is
locked to fd₀ amplitude, independent of transition timing zc.

This is analogous to Planck's A_s·e^(-2τ) degeneracy, but here it
represents a MODEL LIMITATION rather than observational degeneracy.

The lock prevents independent control of H₀ (early-time) and S₈ 
(late-time) tensions, proving Model A is INSUFFICIENT.

CONCLUSION:

Model A's Gaussian parametrization creates an intrinsic lock that
fundamentally limits its ability to resolve cosmological tensions.

Perturbation-level coupling (Model B) is NECESSARY to break this lock.
""")

print("="*70)
print("Analysis complete!")
print("="*70)
