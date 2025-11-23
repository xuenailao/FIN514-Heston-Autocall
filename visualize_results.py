"""
Visualization of Heston Calibration and Valuation Results
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Load results
step1 = np.load('heston_calibration_step1.npz')
final = np.load('final_results.npz', allow_pickle=True)

v0 = float(step1['v0'])
kappa = float(step1['kappa'])
theta = float(step1['theta'])
maturities = step1['maturities']
atm_vol = step1['atm_vol']
fitted_vol = step1['fitted_vol']

# Heston parameters
heston_params = final['heston_params'].item()
sigma = heston_params['sigma']
rho = heston_params['rho']

print("\nCreating 4 visualizations...")

# Create figure with 4 subplots
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: ATM Volatility Surface Calibration
# ============================================================================

ax1 = fig.add_subplot(2, 2, 1)

ax1.scatter(maturities, atm_vol * 100, alpha=0.6, s=50, color='blue', label='Market ATM Vols')
ax1.plot(maturities, fitted_vol * 100, 'r-', linewidth=2, label='Heston Fit')
ax1.set_xlabel('Time to Maturity (years)', fontsize=11)
ax1.set_ylabel('Implied Volatility (%)', fontsize=11)
ax1.set_title('ATM Volatility Surface Calibration', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add RMSE
rmse = np.sqrt(np.mean((fitted_vol - atm_vol)**2)) * 100
ax1.text(0.05, 0.95, f'RMSE = {rmse:.2f}%', transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Plot 2: Variance Term Structure
# ============================================================================

ax2 = fig.add_subplot(2, 2, 2)

# Expected variance under Heston
t_grid = np.linspace(0, 4, 100)
expected_var = theta + (v0 - theta) * np.exp(-kappa * t_grid)

ax2.plot(t_grid, expected_var * 100, 'b-', linewidth=2, label='E[v(t)]')
ax2.axhline(theta * 100, color='r', linestyle='--', linewidth=2, label=f'θ = {theta*100:.2f}%')
ax2.axhline(v0 * 100, color='g', linestyle='--', linewidth=2, label=f'v₀ = {v0*100:.2f}%')
ax2.set_xlabel('Time (years)', fontsize=11)
ax2.set_ylabel('Variance (%)', fontsize=11)
ax2.set_title('Heston Variance Term Structure', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# Add parameters
param_text = f'κ = {kappa:.2f}\nθ = {theta:.4f}\nv₀ = {v0:.4f}'
ax2.text(0.95, 0.05, param_text, transform=ax2.transAxes,
         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================================
# Plot 3: Valuation Comparison
# ============================================================================

ax3 = fig.add_subplot(2, 2, 3)

mc_val = final['monte_carlo'].item()['value']
mc_se = final['monte_carlo'].item()['se']
bin_val = final['binomial'].item()['value']
prosp_val = float(final['prospectus'])

methods = ['Monte Carlo\n(Heston)', 'Binomial\nTree', 'Prospectus\nEstimate']
values = [mc_val, bin_val, prosp_val]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax3.bar(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add error bars for Monte Carlo
ax3.errorbar(0, mc_val, yerr=1.96*mc_se, fmt='none', ecolor='black',
             capsize=10, capthick=2, linewidth=2, label='95% CI')

ax3.set_ylabel('Product Value ($)', fontsize=11)
ax3.set_title('Autocallable Product Valuation Comparison', fontsize=13, fontweight='bold')
ax3.set_ylim([940, 1000])
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(loc='upper right', fontsize=9)

# Add difference annotations
diff_mc_prosp = mc_val - prosp_val
diff_bin_prosp = bin_val - prosp_val

ax3.text(0.5, 0.05, f'MC vs Prospectus: ${diff_mc_prosp:+.2f}\nBin vs Prospectus: ${diff_bin_prosp:+.2f}',
         transform=ax3.transAxes, fontsize=9, verticalalignment='bottom',
         horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# Plot 4: Heston Parameter Summary
# ============================================================================

ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# Parameter table
param_data = [
    ['Parameter', 'Value', 'Description'],
    ['─'*15, '─'*15, '─'*30],
    ['v₀', f'{v0:.6f}', f'Initial variance ({np.sqrt(v0)*100:.2f}% vol)'],
    ['κ', f'{kappa:.4f}', 'Mean reversion speed'],
    ['θ', f'{theta:.6f}', f'Long-term variance ({np.sqrt(theta)*100:.2f}% vol)'],
    ['σ', f'{sigma:.4f}', 'Volatility of volatility'],
    ['ρ', f'{rho:.4f}', 'Correlation (S vs v)'],
    ['', '', ''],
    ['Feller', f'{2*kappa*theta:.4f}', f'2κθ > σ² = {sigma**2:.4f} ✓'],
    ['', '', ''],
    ['Monte Carlo', f'${mc_val:.2f}', f'SE = ${mc_se:.2f}'],
    ['Binomial', f'${bin_val:.2f}', f'N = 2000 steps'],
    ['Prospectus', f'${prosp_val:.2f}', 'Market estimate'],
]

# Create table
table = ax4.table(cellText=param_data, cellLoc='left', loc='center',
                  colWidths=[0.25, 0.25, 0.5])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style separator
for i in range(3):
    table[(1, i)].set_facecolor('#ecf0f1')

# Style Feller row
for i in range(3):
    table[(8, i)].set_facecolor('#d5f4e6')

# Style value rows
for i in range(3):
    table[(10, i)].set_facecolor('#fff9e6')
    table[(11, i)].set_facecolor('#fff9e6')
    table[(12, i)].set_facecolor('#fff9e6')

ax4.set_title('Heston Model Parameters & Results', fontsize=13, fontweight='bold', pad=20)

# ============================================================================
# Save figure
# ============================================================================

plt.tight_layout()
plt.savefig('heston_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: heston_results.png")

# ============================================================================
# Create additional plot: Sample Heston paths
# ============================================================================

print("\nGenerating sample Heston paths...")

from scipy.stats import norm

# Simulate a few paths
np.random.seed(42)
T = 3.8959
N_steps = 500
N_paths_plot = 10

S0 = 6610.0
r = 0.04
q = 0.0245

dt = T / N_steps

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for path in range(N_paths_plot):
    S = np.zeros(N_steps + 1)
    v = np.zeros(N_steps + 1)
    S[0] = S0
    v[0] = v0

    for i in range(N_steps):
        Z1 = np.random.standard_normal()
        Z2 = np.random.standard_normal()

        Z_S = Z1
        Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v[i+1] = v[i] + kappa * (theta - max(v[i], 0)) * dt + \
                 sigma * np.sqrt(max(v[i], 0) * dt) * Z_v
        v[i+1] = max(v[i+1], 0)

        S[i+1] = S[i] * np.exp((r - q - 0.5*max(v[i], 0)) * dt + \
                                np.sqrt(max(v[i], 0) * dt) * Z_S)

    time_grid = np.linspace(0, T, N_steps + 1)

    # Plot stock paths
    ax1.plot(time_grid, S, alpha=0.6, linewidth=1.5)

    # Plot variance paths
    ax2.plot(time_grid, v * 100, alpha=0.6, linewidth=1.5)

# Stock price plot
ax1.axhline(S0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Initial Level')
ax1.axhline(S0 * 0.7, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Contingent Barrier (70%)')
ax1.set_xlabel('Time (years)', fontsize=11)
ax1.set_ylabel('SPX Level', fontsize=11)
ax1.set_title(f'Sample Heston Stock Price Paths (n={N_paths_plot})', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Variance plot
ax2.axhline(theta * 100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Long-term variance θ')
ax2.axhline(v0 * 100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Initial variance v₀')
ax2.set_xlabel('Time (years)', fontsize=11)
ax2.set_ylabel('Variance (%)', fontsize=11)
ax2.set_title('Sample Heston Variance Paths', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heston_sample_paths.png', dpi=300, bbox_inches='tight')
print("✓ Saved: heston_sample_paths.png")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. heston_results.png - Main results dashboard")
print("  2. heston_sample_paths.png - Sample Heston paths")
print("\n" + "="*80)
