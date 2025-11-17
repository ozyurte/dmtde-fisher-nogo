#!/usr/bin/env python3
"""
Test growth rate evolution for different zc values
"""

import numpy as np
import matplotlib.pyplot as plt
from classy import Class

# Redshift points
z_array = np.linspace(0, 3, 50)

# Test cases (label'ı CLASS'a göndermeyeceğiz)
test_cases = [
    {'name': 'ΛCDM', 'params': {'use_dmtde': 0}},
    {'name': 'zc=0.5, fd0=+0.10', 'params': {'use_dmtde': 1, 'dmtde_zc': 0.5, 'dmtde_fd0': 0.10, 'dmtde_dlnac': 0.2}},
    {'name': 'zc=1.5, fd0=+0.10', 'params': {'use_dmtde': 1, 'dmtde_zc': 1.5, 'dmtde_fd0': 0.10, 'dmtde_dlnac': 0.2}},
    {'name': 'zc=3.0, fd0=+0.10', 'params': {'use_dmtde': 1, 'dmtde_zc': 3.0, 'dmtde_fd0': 0.10, 'dmtde_dlnac': 0.2}},
    {'name': 'zc=1.5, fd0=-0.10', 'params': {'use_dmtde': 1, 'dmtde_zc': 1.5, 'dmtde_fd0': -0.10, 'dmtde_dlnac': 0.2}},
]

# Base parameters
base_params = {
    'output': 'mPk',
    'h': 0.674,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.054,
    'P_k_max_h/Mpc': 1.0,
    'z_max_pk': 10.0
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

print("Computing growth rate evolution...")

# Plot 1: fσ₈(z)
ax = axes[0]
for case in test_cases:
    print(f"  Running: {case['name']}")
    
    params = base_params.copy()
    params.update(case['params'])
    
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    fsigma8_array = []
    for z in z_array:
        f = cosmo.scale_independent_growth_factor_f(z)
        sigma8_z = cosmo.sigma8() * cosmo.scale_independent_growth_factor(z)
        fsigma8_array.append(f * sigma8_z)
    
    ax.plot(z_array, fsigma8_array, linewidth=2, label=case['name'])
    
    cosmo.struct_cleanup()
    cosmo.empty()

ax.set_xlabel('Redshift $z$', fontsize=13)
ax.set_ylabel('$f\\sigma_8(z)$', fontsize=13)
ax.set_title('Growth Rate Evolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(0, 3)

# Plot 2: Fractional difference from ΛCDM
ax = axes[1]

print("  Computing ΛCDM reference...")
# Get ΛCDM reference
params = base_params.copy()
params['use_dmtde'] = 0
cosmo_lcdm = Class()
cosmo_lcdm.set(params)
cosmo_lcdm.compute()

fsigma8_lcdm = []
for z in z_array:
    f = cosmo_lcdm.scale_independent_growth_factor_f(z)
    sigma8_z = cosmo_lcdm.sigma8() * cosmo_lcdm.scale_independent_growth_factor(z)
    fsigma8_lcdm.append(f * sigma8_z)
fsigma8_lcdm = np.array(fsigma8_lcdm)

cosmo_lcdm.struct_cleanup()
cosmo_lcdm.empty()

# Plot differences
for case in test_cases[1:]:  # Skip ΛCDM
    print(f"  Computing difference: {case['name']}")
    
    params = base_params.copy()
    params.update(case['params'])
    
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    fsigma8_array = []
    for z in z_array:
        f = cosmo.scale_independent_growth_factor_f(z)
        sigma8_z = cosmo.sigma8() * cosmo.scale_independent_growth_factor(z)
        fsigma8_array.append(f * sigma8_z)
    
    fsigma8_array = np.array(fsigma8_array)
    diff_percent = (fsigma8_array - fsigma8_lcdm) / fsigma8_lcdm * 100
    
    ax.plot(z_array, diff_percent, linewidth=2, label=case['name'])
    
    # Mark transition redshift
    if 'dmtde_zc' in case['params']:
        ax.axvline(case['params']['dmtde_zc'], color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(case['params']['dmtde_zc'], ax.get_ylim()[1]*0.95, 
                f"$z_c$={case['params']['dmtde_zc']}", 
                rotation=90, va='top', fontsize=9, alpha=0.7)
    
    cosmo.struct_cleanup()
    cosmo.empty()

ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Redshift $z$', fontsize=13)
ax.set_ylabel('$\\Delta f\\sigma_8 / f\\sigma_8^{\\Lambda CDM}$ (%)', fontsize=13)
ax.set_title('Fractional Difference from ΛCDM', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)
ax.set_xlim(0, 3)

plt.tight_layout()
plt.savefig('growth_evolution.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved: growth_evolution.png")
plt.show()
