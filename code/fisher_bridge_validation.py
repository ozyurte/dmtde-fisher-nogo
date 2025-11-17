#!/usr/bin/env python3
"""
Fisher Bridge Validation
Computes Fisher information matrix from S(z; f_d0, z_c) field
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Kozmolojik parametreler (Planck 2018)
Omega_m_0 = 0.315
Omega_Lambda_0 = 0.685
sigma_8_0 = 0.8274

def Omega_m(z):
    """Madde yoğunluk parametresi"""
    return Omega_m_0 * (1+z)**3 / (Omega_m_0*(1+z)**3 + Omega_Lambda_0)

def Omega_Lambda(z):
    """Karanlık enerji yoğunluk parametresi"""
    return Omega_Lambda_0 / (Omega_m_0*(1+z)**3 + Omega_Lambda_0)

def growth_factor_approx(z):
    """Lineer büyüme faktörü D(z) - Carroll, Press & Turner (1992)"""
    Om = Omega_m(z)
    OL = Omega_Lambda(z)
    
    g = 5*Om/2 * (Om**(4/7) - OL + (1 + Om/2)*(1 + OL/70))**(-1)
    g0 = 5*Omega_m_0/2 * (Omega_m_0**(4/7) - Omega_Lambda_0 + 
                          (1 + Omega_m_0/2)*(1 + Omega_Lambda_0/70))**(-1)
    
    D = g / g0 / (1+z)
    return D

def sigma_8_LCDM(z):
    """ΛCDM'de σ_8(z)"""
    return sigma_8_0 * growth_factor_approx(z)

def gaussian_transition(z, z_c, sigma=0.2):
    """Gaussian geçiş fonksiyonu"""
    return np.exp(-(z - z_c)**2 / (2*sigma**2))

def sigma_8_DMTDE(z, f_d0, z_c, sigma=0.2):
    """DMTDE modelinde σ_8"""
    return sigma_8_LCDM(z) * (1 + f_d0 * gaussian_transition(z, z_c, sigma))

# Grid tanımları
z_grid = np.linspace(0, 3, 61)
f_d0_grid = np.linspace(-0.15, 0.15, 31)
z_c_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
sigma = 0.2

def compute_S_field(z, f_d0, z_c, sigma=0.2):
    """S(z; f_d0, z_c) = -χ²(z)/2"""
    sigma_8_model = sigma_8_DMTDE(z, f_d0, z_c, sigma)
    sigma_8_obs = sigma_8_LCDM(z)
    sigma_obs_error = 0.01 * sigma_8_obs
    
    chi_squared = ((sigma_8_model - sigma_8_obs) / sigma_obs_error)**2
    S = -0.5 * chi_squared
    
    return S

print("=" * 70)
print("ADIM 1: S(z; f_d0, z_c) ALANI HESAPLANIYOR")
print("=" * 70)

S_field = np.zeros((len(z_grid), len(f_d0_grid), len(z_c_grid)))

for i, z in enumerate(z_grid):
    for j, f_d0 in enumerate(f_d0_grid):
        for k, z_c in enumerate(z_c_grid):
            S_field[i, j, k] = compute_S_field(z, f_d0, z_c, sigma)

print(f"✓ S alanı hesaplandı")
print(f"  Shape: {S_field.shape}")
print(f"  Min: {S_field.min():.2e}, Max: {S_field.max():.2e}")

print("\n" + "=" * 70)
print("ADIM 2: GRADYANLAR HESAPLANIYOR")
print("=" * 70)

def compute_gradients(S_field, f_d0_grid, z_c_grid):
    """∂S/∂f_d0 ve ∂S/∂z_c gradyanlarını hesapla"""
    nz, nf, nzc = S_field.shape
    
    grad_S_f = np.zeros_like(S_field)
    grad_S_zc = np.zeros_like(S_field)
    
    df = f_d0_grid[1] - f_d0_grid[0]
    for j in range(1, nf-1):
        grad_S_f[:, j, :] = (S_field[:, j+1, :] - S_field[:, j-1, :]) / (2*df)
    
    grad_S_f[:, 0, :] = (S_field[:, 1, :] - S_field[:, 0, :]) / df
    grad_S_f[:, -1, :] = (S_field[:, -1, :] - S_field[:, -2, :]) / df
    
    dzc = z_c_grid[1] - z_c_grid[0]
    for k in range(1, nzc-1):
        grad_S_zc[:, :, k] = (S_field[:, :, k+1] - S_field[:, :, k-1]) / (2*dzc)
    
    grad_S_zc[:, :, 0] = (S_field[:, :, 1] - S_field[:, :, 0]) / dzc
    grad_S_zc[:, :, -1] = (S_field[:, :, -1] - S_field[:, :, -2]) / dzc
    
    return grad_S_f, grad_S_zc

grad_S_f, grad_S_zc = compute_gradients(S_field, f_d0_grid, z_c_grid)

print(f"✓ Gradyanlar hesaplandı")
print(f"  ∂S/∂f_d0 range: [{grad_S_f.min():.2e}, {grad_S_f.max():.2e}]")
print(f"  ∂S/∂z_c range: [{grad_S_zc.min():.2e}, {grad_S_zc.max():.2e}]")

print("\n" + "=" * 70)
print("ADIM 3: TEORİK FISHER MATRİSİ")
print("=" * 70)

def compute_Fisher_theory(z_grid, grad_S_f, grad_S_zc, f_d0_idx, z_c_idx):
    """Teorik Fisher matrisini hesapla"""
    dS_df = grad_S_f[:, f_d0_idx, z_c_idx]
    dS_dzc = grad_S_zc[:, f_d0_idx, z_c_idx]
    
    F_11 = simpson(dS_df**2, x=z_grid)
    F_22 = simpson(dS_dzc**2, x=z_grid)
    F_12 = simpson(dS_df * dS_dzc, x=z_grid)
    
    F_theory = np.array([[F_11, F_12], [F_12, F_22]])
    
    return F_theory

f_d0_test = 0.1
z_c_test = 1.5

f_idx = np.argmin(np.abs(f_d0_grid - f_d0_test))
zc_idx = np.argmin(np.abs(z_c_grid - z_c_test))

F_theory = compute_Fisher_theory(z_grid, grad_S_f, grad_S_zc, f_idx, zc_idx)

print(f"\nParametre noktası: f_d0={f_d0_test}, z_c={z_c_test}")
print("\n✓ Teorik Fisher Matrisi:")
print(f"  [[{F_theory[0,0]:.3e}, {F_theory[0,1]:.3e}]")
print(f"   [{F_theory[1,0]:.3e}, {F_theory[1,1]:.3e}]]")

print(f"\nDiyagonal elemanlar:")
print(f"  F_11 (f_d0-f_d0): {F_theory[0,0]:.3e}")
print(f"  F_22 (z_c-z_c):   {F_theory[1,1]:.3e}")
print(f"  F_12 (cross):     {F_theory[0,1]:.3e}")

eigenvalues, eigenvectors = np.linalg.eigh(F_theory)
condition_number = eigenvalues[1] / eigenvalues[0]

print(f"\nÖzdeğer Analizi:")
print(f"  λ_1 (küçük): {eigenvalues[0]:.3e}")
print(f"  λ_2 (büyük): {eigenvalues[1]:.3e}")
print(f"  Koşul sayısı (κ = λ_2/λ_1): {condition_number:.2f}")

print(f"\nÖzvektörler:")
print(f"  v_1 (zayıf yön): [{eigenvectors[0,0]:.3f}, {eigenvectors[1,0]:.3f}]")
print(f"  v_2 (güçlü yön): [{eigenvectors[0,1]:.3f}, {eigenvectors[1,1]:.3f}]")

sensitivity_ratio = np.sqrt(F_theory[0,0] / F_theory[1,1])
print(f"\nDuyarlılık Oranı: |∂S/∂f_d0| : |∂S/∂z_c| ≈ {sensitivity_ratio:.1f}:1")

# ==================== SAVE GRADIENT DATA ====================
print("\n" + "=" * 70)
print("SAVING GRADIENT DATA FOR PLOTTING")
print("=" * 70)

np.savez('gradient_data.npz',
         z_grid=z_grid,
         grad_S_f=grad_S_f[:, f_idx, zc_idx],
         grad_S_zc=grad_S_zc[:, f_idx, zc_idx],
         F_11=F_theory[0,0],
         F_22=F_theory[1,1],
         F_12=F_theory[0,1])
print("✓ Gradient data saved to gradient_data.npz")

print("\n" + "=" * 70)
print("HESAPLAMA TAMAMLANDI")
print("=" * 70)