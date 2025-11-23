"""
FIN514 Project 2: Heston Model Calibration and Valuation
Stochastic Volatility Model for SPX Options
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Dataset 1: ATM Implied Volatilities and Forward Prices
# ============================================================================

exp_dates = ['24 Nov 2025', '25 Nov 2025', '26 Nov 2025', '28 Nov 2025',
             '1 Dec 2025', '2 Dec 2025', '3 Dec 2025', '4 Dec 2025', '5 Dec 2025',
             '8 Dec 2025', '9 Dec 2025', '10 Dec 2025', '11 Dec 2025', '12 Dec 2025',
             '15 Dec 2025', '16 Dec 2025', '17 Dec 2025', '18 Dec 2025', '19 Dec 2025',
             '22 Dec 2025', '23 Dec 2025', '24 Dec 2025', '26 Dec 2025', '29 Dec 2025',
             '30 Dec 2025', '31 Dec 2025', '2 Jan 2026', '8 Jan 2026', '9 Jan 2026',
             '15 Jan 2026', '16 Jan 2026', '30 Jan 2026', '20 Feb 2026', '27 Feb 2026',
             '20 Mar 2026', '31 Mar 2026', '17 Apr 2026', '30 Apr 2026', '15 May 2026',
             '18 Jun 2026', '30 Jun 2026', '17 Jul 2026', '21 Aug 2026', '18 Sep 2026',
             '30 Sep 2026', '16 Oct 2026', '20 Nov 2026', '18 Dec 2026', '15 Jan 2027',
             '17 Jun 2027', '17 Dec 2027', '15 Dec 2028', '21 Dec 2029']

imp_fwd = np.array([6616.18, 6616.64, 6618.35, 6620.32, 6620.65, 6621.05, 6621.69,
                    6622.05, 6623.52, 6623.84, 6624.56, 6625.04, 6625.95, 6627.86,
                    6627.92, 6628.38, 6628.99, 6629.85, 6632.33, 6632.82, 6633.13,
                    6634.85, 6636.25, 6636.78, 6637.64, 6638.44, 6644.91, 6647.29,
                    6649.26, 6652.38, 6655.26, 6664.13, 6675.19, 6679.27, 6690.19,
                    6695.22, 6707.90, 6714.97, 6723.93, 6741.26, 6746.07, 6757.70,
                    6775.91, 6788.93, 6794.18, 6804.37, 6820.93, 6832.84, 6851.14,
                    6927.57, 7020.12, 7231.23, 7480.36])

atm_vol = np.array([14.63, 17.43, 18.98, 18.13, 17.09, 17.70, 18.08, 18.42, 18.79,
                    17.94, 18.20, 18.76, 19.10, 19.21, 18.66, 18.97, 19.06, 19.33,
                    19.16, 18.89, 19.00, 18.93, 18.65, 18.27, 18.46, 18.51, 18.44,
                    18.29, 18.43, 18.38, 18.37, 18.52, 18.45, 18.62, 18.76, 18.79,
                    18.84, 18.94, 19.01, 19.17, 19.21, 19.28, 19.48, 19.60, 19.65,
                    19.71, 19.84, 19.88, 19.87, 20.10, 20.37, 20.51, 20.84]) / 100.0

# Pricing date (assumed to be around Nov 2025 based on the data)
S0 = 6610.0  # Approximate spot price
pricing_date = "22 Nov 2025"

# Calculate time to maturity in years (approximate from pricing date)
from datetime import datetime

def parse_date(date_str):
    return datetime.strptime(date_str, '%d %b %Y')

pricing_dt = parse_date(pricing_date)
maturities = np.array([(parse_date(d) - pricing_dt).days / 365.0 for d in exp_dates])

print("="*70)
print("FIN514 PROJECT 2: HESTON MODEL CALIBRATION")
print("="*70)
print(f"\nPricing Date: {pricing_date}")
print(f"Spot Price S0: {S0:.2f}")
print(f"\nDataset 1 Statistics:")
print(f"Number of maturities: {len(maturities)}")
print(f"Time to maturity range: {maturities[0]:.4f} to {maturities[-1]:.4f} years")
print(f"ATM volatility range: {atm_vol.min()*100:.2f}% to {atm_vol.max()*100:.2f}%")
print()

# ============================================================================
# HESTON MODEL IMPLEMENTATION
# ============================================================================

def heston_char_func(phi, S0, v0, kappa, theta, sigma, rho, tau, r, q):
    """
    Heston characteristic function

    Parameters:
    -----------
    phi : complex
        Argument of characteristic function
    S0 : float
        Current stock price
    v0 : float
        Current variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance
    sigma : float
        Volatility of volatility
    rho : float
        Correlation between stock and variance
    tau : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    """
    # Parameters for the characteristic function
    a = kappa * theta

    # Avoid numerical issues
    u = 0.5
    b = kappa

    d = np.sqrt((rho * sigma * phi * 1j - b)**2 + sigma**2 * (phi * 1j + phi**2))
    g = (b - rho * sigma * phi * 1j - d) / (b - rho * sigma * phi * 1j + d)

    C = (r - q) * phi * 1j * tau + a / sigma**2 * (
        (b - rho * sigma * phi * 1j - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
    )

    D = (b - rho * sigma * phi * 1j - d) / sigma**2 * (
        (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )

    return np.exp(C + D * v0 + 1j * phi * np.log(S0))


def heston_price_call(S0, K, v0, kappa, theta, sigma, rho, tau, r, q):
    """
    Price a European call option using the Heston model
    """
    # Integration for P1
    def integrand1(phi):
        char_func = heston_char_func(phi - 1j, S0, v0, kappa, theta, sigma, rho, tau, r, q)
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func / (1j * phi))

    # Integration for P2
    def integrand2(phi):
        char_func = heston_char_func(phi, S0, v0, kappa, theta, sigma, rho, tau, r, q)
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func / (1j * phi))

    # Calculate probabilities
    try:
        P1 = 0.5 + 1/np.pi * quad(integrand1, 0, 100)[0]
        P2 = 0.5 + 1/np.pi * quad(integrand2, 0, 100)[0]
    except:
        P1 = 0.5
        P2 = 0.5

    # Call price
    call_price = S0 * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2

    return call_price


def heston_implied_vol(S0, K, v0, kappa, theta, sigma, rho, tau, r, q):
    """
    Calculate implied volatility from Heston price using Black-Scholes inversion
    """
    call_price = heston_price_call(S0, K, v0, kappa, theta, sigma, rho, tau, r, q)

    # Black-Scholes implied volatility (using Newton-Raphson)
    F = S0 * np.exp((r - q) * tau)
    d1 = lambda vol: (np.log(F/K) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
    d2 = lambda vol: d1(vol) - vol * np.sqrt(tau)
    bs_price = lambda vol: np.exp(-r * tau) * (F * norm.cdf(d1(vol)) - K * norm.cdf(d2(vol)))

    # Newton-Raphson iteration
    vol = 0.2  # Initial guess
    for i in range(50):
        price = bs_price(vol)
        vega = S0 * np.exp(-q * tau) * norm.pdf(d1(vol)) * np.sqrt(tau)
        if abs(vega) < 1e-10:
            break
        vol_new = vol - (price - call_price) / vega
        if abs(vol_new - vol) < 1e-6:
            break
        vol = vol_new

    return vol


# ============================================================================
# STEP 1: Initial Calibration with Dataset 1 (v0, kappa, theta)
# ============================================================================

print("="*70)
print("STEP 1: CALIBRATING v0, kappa, theta FROM ATM VOLATILITIES")
print("="*70)

# Set v0 to shortest dated ATM variance
v0 = atm_vol[0]**2
print(f"\nv0 (initial variance) set to: {v0:.6f} (vol = {np.sqrt(v0)*100:.2f}%)")

# Initial guess: theta = longest dated variance
theta_init = atm_vol[-1]**2
print(f"theta (initial guess) set to: {theta_init:.6f} (vol = {np.sqrt(theta_init)*100:.2f}%)")

# E[v(t)] = theta + (v0 - theta) * exp(-kappa * t)
# For ATM options, variance ≈ E[v(t)]
def objective_kappa_theta(params):
    kappa, theta = params

    # Feller condition penalty
    feller_penalty = 0
    if 2 * kappa * theta < 0.1**2:  # Soft constraint
        feller_penalty = 10 * (0.1**2 - 2 * kappa * theta)**2

    # Expected variance at each maturity
    expected_var = theta + (v0 - theta) * np.exp(-kappa * maturities)

    # Sum of squared errors
    squared_errors = np.sum((expected_var - atm_vol**2)**2)

    return squared_errors + feller_penalty

# Optimize kappa and theta
result = minimize(objective_kappa_theta,
                 x0=[2.0, theta_init],
                 bounds=[(0.1, 10.0), (0.01, 1.0)],
                 method='L-BFGS-B')

kappa, theta = result.x

print(f"\nCalibrated parameters:")
print(f"  kappa (mean reversion speed): {kappa:.6f}")
print(f"  theta (long-term variance): {theta:.6f} (vol = {np.sqrt(theta)*100:.2f}%)")
print(f"  v0 (initial variance): {v0:.6f} (vol = {np.sqrt(v0)*100:.2f}%)")
print(f"  Feller condition 2*kappa*theta = {2*kappa*theta:.6f} (should be > 0)")

# Calculate fitted values
fitted_var = theta + (v0 - theta) * np.exp(-kappa * maturities)
fitted_vol = np.sqrt(fitted_var)

print(f"\nFit quality (RMSE): {np.sqrt(np.mean((fitted_vol - atm_vol)**2))*100:.4f}%")

# ============================================================================
# STEP 2: Prepare for full calibration (need Dataset 2 and 3)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: SETTING UP FOR FULL CALIBRATION")
print("="*70)

print("\nNote: For full calibration with Dataset 2 and Dataset 3, we need:")
print("  - Dataset 2: All strikes for specific maturities (19 Dec 2025, 20 Mar 2026)")
print("  - Dataset 3: ~25 strikes at quarterly maturities through Dec 2029")
print("\nThese datasets are in Excel format in dataset2/ and dataset3/ folders")
print("We will need to load and process these files.")

# Initial parameter estimates for full calibration
print(f"\nInitial parameter estimates:")
print(f"  v0 = {v0:.6f}")
print(f"  kappa = {kappa:.6f}")
print(f"  theta = {theta:.6f}")
print(f"  sigma (vol-of-vol) = 0.5 (initial guess)")
print(f"  rho (correlation) = -0.7 (initial guess)")
print(f"  r (risk-free rate) = 0.04 (initial guess)")

# Save calibration results
np.savez('heston_calibration_step1.npz',
         v0=v0, kappa=kappa, theta=theta,
         maturities=maturities, atm_vol=atm_vol,
         fitted_vol=fitted_vol)

print("\nStep 1 calibration results saved to: heston_calibration_step1.npz")
print("="*70)
