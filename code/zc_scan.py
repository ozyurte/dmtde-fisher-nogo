#!/usr/bin/env python3
"""
DMTDE zc Parameter Scan
Test how transition redshift affects σ₈ and S₈
"""

import numpy as np
import matplotlib.pyplot as plt
from classy import Class

# Test parameters
zc_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
fd0_values = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]

# Base cosmology (Planck 2018)
base_params = {
    'output': 'mPk',
    'h': 0.6736,                    # UPDATED: Consistent with calculate_S8.py
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,                  # UPDATED: Planck 2018 best-fit
    'tau_reio': 0.0544,             # UPDATED: Planck 2018 best-fit
    'P_k_max_h/Mpc': 10.0,
    'z_max_pk': 10.0
}

# Storage for results
results = {
    'zc': [],
    'fd0': [],
    'sigma8': [],
    'S8': [],
    'Omega_m': [],
    'fsigma8_z0': []
}

print("="*70)
print("DMTDE zc PARAMETER SCAN")
print("="*70)

# Test 1: Scan zc for fixed fd0
print("\n[TEST 1] Scanning zc with fd0 = 0.10")
print("-"*70)

fd0_fixed = 0.10
for zc in zc_values:
    print(f"\nRunning: zc = {zc:.1f}, fd0 = {fd0_fixed:.2f}")
    
    # Setup CLASS parameters
    params = base_params.copy()
    params['use_dmtde'] = 1
    params['dmtde_fd0'] = fd0_fixed
    params['dmtde_zc'] = zc
    params['dmtde_dlnac'] = 0.2
    
    try:
        # Initialize CLASS
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        
        # Get results
        sigma8 = cosmo.sigma8()
        h = cosmo.h()
        Omega_m = cosmo.Omega_m()
        S8 = sigma8 * np.sqrt(Omega_m / 0.3)
        
        # Growth rate at z=0
        fsigma8_z0 = cosmo.scale_independent_growth_factor_f(0.0) * sigma8
        
        # Store results
        results['zc'].append(zc)
        results['fd0'].append(fd0_fixed)
        results['sigma8'].append(sigma8)
        results['S8'].append(S8)
        results['Omega_m'].append(Omega_m)
        results['fsigma8_z0'].append(fsigma8_z0)
        
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

# Test 2: Scan fd0 for fixed zc
print("\n\n[TEST 2] Scanning fd0 with zc = 1.5")
print("-"*70)

zc_fixed = 1.5
for fd0 in fd0_values:
    print(f"\nRunning: zc = {zc_fixed:.1f}, fd0 = {fd0:+.2f}")
    
    params = base_params.copy()
    
    if fd0 != 0.0:
        params['use_dmtde'] = 1
        params['dmtde_fd0'] = fd0
        params['dmtde_zc'] = zc_fixed
        params['dmtde_dlnac'] = 0.2
    else:
        params['use_dmtde'] = 0  # ΛCDM reference
    
    try:
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        
        sigma8 = cosmo.sigma8()
        h = cosmo.h()
        Omega_m = cosmo.Omega_m()
        S8 = sigma8 * np.sqrt(Omega_m / 0.3)
        fsigma8_z0 = cosmo.scale_independent_growth_factor_f(0.0) * sigma8
        
        results['zc'].append(zc_fixed)
        results['fd0'].append(fd0)
        results['sigma8'].append(sigma8)
        results['S8'].append(S8)
        results['Omega_m'].append(Omega_m)
        results['fsigma8_z0'].append(fsigma8_z0)
        
        print(f"  σ₈ = {sigma8:.6f}")
        print(f"  S₈ = {S8:.6f}")
        print(f"  Ωₘ = {Omega_m:.6f}")
        
        cosmo.struct_cleanup()
        cosmo.empty()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

# Save results
np.savez('zc_scan_results.npz', **results)
print("\n" + "="*70)
print("Results saved to: zc_scan_results.npz")
print("="*70)

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

zc_arr = np.array(results['zc'])
fd0_arr = np.array(results['fd0'])
sigma8_arr = np.array(results['sigma8'])
S8_arr = np.array(results['S8'])

print("\nTest 1: zc scan (fd0 = 0.10)")
mask = np.abs(fd0_arr - 0.10) < 0.01
for i in np.where(mask)[0]:
    print(f"  zc = {zc_arr[i]:.1f}: σ₈ = {sigma8_arr[i]:.6f}, S₈ = {S8_arr[i]:.6f}")

print("\nTest 2: fd0 scan (zc = 1.5)")
mask = np.abs(zc_arr - 1.5) < 0.01
sort_idx = np.argsort(fd0_arr[mask])
for i in np.where(mask)[0][sort_idx]:
    print(f"  fd0 = {fd0_arr[i]:+.2f}: σ₈ = {sigma8_arr[i]:.6f}, S₈ = {S8_arr[i]:.6f}")

print("\n" + "="*70)