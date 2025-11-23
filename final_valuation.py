"""
FIN514 Project 2: Complete Heston Calibration and Autocallable Valuation
Author: [Your Name]
Date: November 2025
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*15 + "FIN514 PROJECT 2: STOCHASTIC VOLATILITY")
print(" "*10 + "Heston Model Calibration and Product Valuation")
print("="*80)

# ============================================================================
# STEP 1: CALIBRATE HESTON PARAMETERS FROM DATASET 1
# ============================================================================

print("\n" + "="*80)
print("STEP 1: CALIBRATING v0, kappa, theta FROM ATM VOLATILITIES")
print("="*80)

# Dataset 1: ATM implied volatilities and maturities
atm_vol = np.array([14.63, 17.43, 18.98, 18.13, 17.09, 17.70, 18.08, 18.42, 18.79,
                    17.94, 18.20, 18.76, 19.10, 19.21, 18.66, 18.97, 19.06, 19.33,
                    19.16, 18.89, 19.00, 18.93, 18.65, 18.27, 18.46, 18.51, 18.44,
                    18.29, 18.43, 18.38, 18.37, 18.52, 18.45, 18.62, 18.76, 18.79,
                    18.84, 18.94, 19.01, 19.17, 19.21, 19.28, 19.48, 19.60, 19.65,
                    19.71, 19.84, 19.88, 19.87, 20.10, 20.37, 20.51, 20.84]) / 100.0

# Approximate maturities (days from 22 Nov 2025)
days_to_maturity = np.array([2, 3, 4, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20,
                             23, 24, 25, 26, 27, 30, 31, 32, 34, 37, 38, 39, 41,
                             47, 48, 54, 55, 69, 90, 97, 118, 129, 146, 159, 174,
                             208, 220, 237, 272, 300, 312, 328, 363, 391, 419,
                             572, 755, 1118, 1490])

maturities = days_to_maturity / 365.0

# Market parameters
S0 = 6610.0
r = 0.04
q = 0.0245

print(f"\nMarket data:")
print(f"  Spot Price S0: ${S0:.2f}")
print(f"  Risk-free rate r: {r*100:.2f}%")
print(f"  Dividend yield q: {q*100:.2f}%")
print(f"  Number of maturities: {len(maturities)}")
print(f"  Maturity range: {maturities[0]:.4f} to {maturities[-1]:.4f} years")

# Calibrate v0, kappa, theta
v0 = atm_vol[0]**2
print(f"\nv0 (initial variance): {v0:.6f} (vol = {np.sqrt(v0)*100:.2f}%)")

def objective_kappa_theta(params):
    kappa, theta = params
    expected_var = theta + (v0 - theta) * np.exp(-kappa * maturities)
    return np.sum((expected_var - atm_vol**2)**2)

result = minimize(objective_kappa_theta, x0=[2.0, atm_vol[-1]**2],
                 bounds=[(0.1, 10.0), (0.01, 1.0)], method='L-BFGS-B')

kappa, theta = result.x

print(f"\nCalibrated parameters:")
print(f"  kappa (mean reversion): {kappa:.4f}")
print(f"  theta (long-term variance): {theta:.6f} (vol = {np.sqrt(theta)*100:.2f}%)")

# Fit quality
fitted_var = theta + (v0 - theta) * np.exp(-kappa * maturities)
rmse = np.sqrt(np.mean((np.sqrt(fitted_var) - atm_vol)**2))
print(f"  RMSE: {rmse*100:.4f}%")

# ============================================================================
# STEP 2: SET REMAINING HESTON PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: SETTING sigma (vol-of-vol) AND rho (correlation)")
print("="*80)

# Use standard values based on SPX calibrations in literature
sigma = 0.5   # vol-of-vol
rho = -0.7    # correlation (typically negative for equity indices)

print(f"\nParameters (based on literature/market practice):")
print(f"  sigma (vol-of-vol): {sigma:.4f}")
print(f"  rho (correlation): {rho:.4f}")

# Feller condition
feller = 2 * kappa * theta
print(f"\nFeller condition check:")
print(f"  2*kappa*theta = {feller:.4f}")
print(f"  sigma^2 = {sigma**2:.4f}")
print(f"  Feller satisfied (2kθ > σ^2): {feller > sigma**2}")

# ============================================================================
# HESTON MODEL IMPLEMENTATION
# ============================================================================

def heston_char_func(phi, S, v, kappa, theta, sigma, rho, tau, r, q):
    """Heston characteristic function"""
    b = kappa
    d = np.sqrt((rho*sigma*phi*1j - b)**2 + sigma**2*(phi*1j + phi**2))
    g = (b - rho*sigma*phi*1j - d) / (b - rho*sigma*phi*1j + d)

    C = (r - q)*phi*1j*tau + kappa*theta/sigma**2 * (
        (b - rho*sigma*phi*1j - d)*tau - 2*np.log((1 - g*np.exp(-d*tau))/(1 - g))
    )
    D = (b - rho*sigma*phi*1j - d)/sigma**2 * ((1 - np.exp(-d*tau))/(1 - g*np.exp(-d*tau)))

    return np.exp(C + D*v + 1j*phi*np.log(S))


def heston_monte_carlo(S0, v0, kappa, theta, sigma, rho, r, q, T, N_steps, N_paths, seed=42):
    """Simulate Heston paths using Euler scheme"""
    np.random.seed(seed)
    dt = T / N_steps

    S = np.zeros((N_paths, N_steps + 1))
    v = np.zeros((N_paths, N_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for i in range(N_steps):
        Z1 = np.random.standard_normal(N_paths)
        Z2 = np.random.standard_normal(N_paths)

        Z_S = Z1
        Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v[:, i+1] = v[:, i] + kappa * (theta - np.maximum(v[:, i], 0)) * dt + \
                    sigma * np.sqrt(np.maximum(v[:, i], 0) * dt) * Z_v
        v[:, i+1] = np.maximum(v[:, i+1], 0)

        S[:, i+1] = S[:, i] * np.exp((r - q - 0.5*np.maximum(v[:, i], 0)) * dt + \
                                      np.sqrt(np.maximum(v[:, i], 0) * dt) * Z_S)

    return S, v


# ============================================================================
# AUTOCALLABLE PRODUCT VALUATION
# ============================================================================

print("\n" + "="*80)
print("AUTOCALLABLE CONTINGENT COUPON VALUATION")
print("="*80)

Face = 1000
coupon_rate = 0.073
barrier_pct = 0.70
autocall_pct = 1.00
T_maturity = 1422 / 365.0  # From prospectus

contingent_barrier = barrier_pct * S0
autocall_barrier = autocall_pct * S0

print(f"\nProduct Specifications:")
print(f"  Face Value: ${Face}")
print(f"  Coupon Rate: {coupon_rate*100:.1f}% per year")
print(f"  Quarterly Coupon: ${Face * coupon_rate / 4:.2f}")
print(f"  Contingent Barrier: ${contingent_barrier:.2f} ({barrier_pct*100:.0f}% of S0)")
print(f"  Autocall Barrier: ${autocall_barrier:.2f} ({autocall_pct*100:.0f}% of S0)")
print(f"  Maturity: {T_maturity:.4f} years ({int(T_maturity*365)} days)")
print(f"  Pricing Date: 22 Nov 2025")

# Monte Carlo valuation
print("\n" + "-"*80)
print("MONTE CARLO VALUATION WITH HESTON MODEL")
print("-"*80)

N_paths = 100000
N_steps = 600

print(f"\nSimulation Settings:")
print(f"  Number of paths: {N_paths:,}")
print(f"  Time steps: {N_steps}")

print(f"\nSimulating paths...")
S_paths, v_paths = heston_monte_carlo(S0, v0, kappa, theta, sigma, rho, r, q,
                                       T_maturity, N_steps, N_paths)

# Pricing logic
Cpn = Face * coupon_rate / 4
autocall_times = np.arange(0.25, T_maturity + 0.01, 0.25)
dt = T_maturity / N_steps
autocall_idx = [int(t/dt) for t in autocall_times if int(t/dt) < N_steps]

payoffs = np.zeros(N_paths)

for path in range(N_paths):
    total_pv = 0.0
    autocalled = False

    # Check autocall dates
    for i, idx in enumerate(autocall_idx):
        if idx >= len(S_paths[path]):
            continue

        S_t = S_paths[path, idx]
        t = autocall_times[i]

        if S_t >= autocall_barrier:
            # Autocalled: receive face + coupon
            total_pv = (Face + Cpn) * np.exp(-r * t)
            autocalled = True
            break
        elif S_t >= contingent_barrier:
            # Get coupon only
            total_pv += Cpn * np.exp(-r * t)

    # Final payoff if not autocalled
    if not autocalled:
        S_T = S_paths[path, -1]

        if S_T >= contingent_barrier:
            total_pv += (Face + Cpn) * np.exp(-r * T_maturity)
        else:
            # Downside participation
            total_pv += Face * (S_T / S0) * np.exp(-r * T_maturity)

    payoffs[path] = total_pv

mc_value = np.mean(payoffs)
mc_se = np.std(payoffs) / np.sqrt(N_paths)

print(f"\nMonte Carlo Results:")
print(f"  Product Value: ${mc_value:.2f}")
print(f"  Standard Error: ${mc_se:.2f}")
print(f"  95% CI: [${mc_value - 1.96*mc_se:.2f}, ${mc_value + 1.96*mc_se:.2f}]")

# Binomial tree valuation
print("\n" + "-"*80)
print("BINOMIAL TREE VALUATION")
print("-"*80)

# Use average realized volatility from Heston simulation
realized_vols = []
for path in range(min(1000, N_paths)):
    log_returns = np.diff(np.log(S_paths[path, :]))
    realized_vols.append(np.std(log_returns) * np.sqrt(252))

avg_vol = np.mean(realized_vols)

print(f"\nAverage realized volatility from Heston: {avg_vol*100:.2f}%")
print(f"Initial volatility (sqrt(v0)): {np.sqrt(v0)*100:.2f}%")
print(f"Long-term volatility (sqrt(theta)): {np.sqrt(theta)*100:.2f}%")

# Use a volatility between initial and long-term
binomial_vol = np.sqrt(theta + (v0 - theta) * np.exp(-kappa * T_maturity / 2))

print(f"Using binomial vol (mid-point): {binomial_vol*100:.2f}%")

N_bin = 2000
dt_bin = T_maturity / N_bin
u = np.exp(binomial_vol * np.sqrt(dt_bin))
d = 1 / u
p_up = (np.exp((r - q) * dt_bin) - d) / (u - d)

# Build tree
stock = np.zeros((N_bin + 1, N_bin + 1))
option = np.zeros((N_bin + 1, N_bin + 1))

# Terminal stock prices
for i in range(N_bin + 1):
    stock[N_bin, i] = S0 * (u ** i) * (d ** (N_bin - i))
    S_T = stock[N_bin, i]

    if S_T >= contingent_barrier:
        option[N_bin, i] = Face + Cpn
    else:
        option[N_bin, i] = Face * S_T / S0

# Backward induction
autocall_steps = [int(t/dt_bin) for t in autocall_times if int(t/dt_bin) < N_bin]

for j in range(N_bin - 1, -1, -1):
    for i in range(j + 1):
        stock[j, i] = S0 * (u ** i) * (d ** (j - i))
        cont = np.exp(-r * dt_bin) * (p_up * option[j+1, i+1] + (1-p_up) * option[j+1, i])

        if j in autocall_steps:
            if stock[j, i] >= autocall_barrier:
                cont = Face + Cpn
            elif stock[j, i] >= contingent_barrier:
                cont += Cpn

        option[j, i] = cont

bin_value = option[0, 0]

print(f"\nBinomial Tree Results:")
print(f"  Product Value: ${bin_value:.2f}")
print(f"  Number of steps: {N_bin}")

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPARISON AND ANALYSIS")
print("="*80)

prospectus_value = 965.2  # From term sheet

print(f"\nValuation Summary:")
print(f"  {'Method':<30} {'Value':>15} {'vs. Prospectus':>20}")
print(f"  {'-'*30} {'-'*15} {'-'*20}")
print(f"  {'Monte Carlo (Heston)':<30} ${mc_value:>13.2f} ${mc_value - prospectus_value:>+18.2f}")
print(f"  {'Binomial Tree':<30} ${bin_value:>13.2f} ${bin_value - prospectus_value:>+18.2f}")
print(f"  {'Prospectus Estimate':<30} ${prospectus_value:>13.2f}")

print(f"\nDifference between methods: ${abs(mc_value - bin_value):.2f}")
print(f"Relative difference: {abs(mc_value - bin_value)/mc_value*100:.2f}%")

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS")
print("="*80)

print(f"\nTesting sensitivity to key parameters...")

# Sensitivity to volatility
print(f"\nSensitivity to Volatility (binomial tree):")
vols = [binomial_vol * 0.9, binomial_vol, binomial_vol * 1.1]
for v in vols:
    u_test = np.exp(v * np.sqrt(dt_bin))
    d_test = 1 / u_test
    p_test = (np.exp((r - q) * dt_bin) - d_test) / (u_test - d_test)

    # Quick binomial valuation (simplified)
    opt_test = np.zeros((N_bin + 1, N_bin + 1))
    for i in range(N_bin + 1):
        S_T = S0 * (u_test ** i) * (d_test ** (N_bin - i))
        opt_test[N_bin, i] = (Face + Cpn) if S_T >= contingent_barrier else Face * S_T / S0

    for j in range(N_bin - 1, -1, -1):
        for i in range(j + 1):
            opt_test[j, i] = np.exp(-r * dt_bin) * (p_test * opt_test[j+1, i+1] + (1-p_test) * opt_test[j+1, i])

    print(f"  Vol = {v*100:.2f}%: Value = ${opt_test[0,0]:.2f}")

# Save all results
print("\n" + "="*80)
results = {
    'heston_params': {'v0': v0, 'kappa': kappa, 'theta': theta, 'sigma': sigma, 'rho': rho},
    'market_params': {'S0': S0, 'r': r, 'q': q},
    'monte_carlo': {'value': mc_value, 'se': mc_se, 'N_paths': N_paths},
    'binomial': {'value': bin_value, 'vol': binomial_vol, 'N_steps': N_bin},
    'prospectus': prospectus_value
}

np.savez('final_results.npz', **results)

print("FINAL RESULTS SAVED TO: final_results.npz")
print("="*80)
