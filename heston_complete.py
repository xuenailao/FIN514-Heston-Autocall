"""
Complete Heston Model Calibration and Monte Carlo Valuation
FIN514 Project 2
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*20 + "FIN514 PROJECT 2")
print(" "*10 + "HESTON MODEL CALIBRATION AND VALUATION")
print("="*80)

# ============================================================================
# PARAMETERS FROM DATASET 1
# ============================================================================

S0 = 6610.0  # Spot price on pricing date (22 Nov 2025)
pricing_date = "22 Nov 2025"

# Load step 1 calibration
step1 = np.load('heston_calibration_step1.npz')
v0 = float(step1['v0'])
kappa = float(step1['kappa'])
theta = float(step1['theta'])

# Initial guesses for remaining parameters
sigma = 0.5   # vol-of-vol
rho = -0.7    # correlation
r = 0.04      # risk-free rate
q = 0.0245    # dividend yield

print(f"\nPricing Date: {pricing_date}")
print(f"Spot Price S0: ${S0:.2f}")
print(f"\nHeston Model Parameters (Initial):")
print(f"  v0 (initial variance): {v0:.6f} (vol = {np.sqrt(v0)*100:.2f}%)")
print(f"  kappa (mean reversion): {kappa:.4f}")
print(f"  theta (long-term var): {theta:.6f} (vol = {np.sqrt(theta)*100:.2f}%)")
print(f"  sigma (vol-of-vol): {sigma:.4f}")
print(f"  rho (correlation): {rho:.4f}")
print(f"  r (risk-free rate): {r*100:.2f}%")
print(f"  q (dividend yield): {q*100:.2f}%")

# Feller condition check
feller = 2 * kappa * theta
print(f"\nFeller condition: 2*kappa*theta = {feller:.4f} > sigma^2 = {sigma**2:.4f}")
print(f"Feller satisfied: {feller > sigma**2}")

# ============================================================================
# HESTON MODEL FUNCTIONS
# ============================================================================

def heston_char_func(phi, S, v, kappa, theta, sigma, rho, tau, r, q):
    """Heston characteristic function"""
    b = kappa
    u = 0.5
    d = np.sqrt((rho*sigma*phi*1j - b)**2 + sigma**2*(phi*1j + phi**2))
    g = (b - rho*sigma*phi*1j - d) / (b - rho*sigma*phi*1j + d)

    C = (r - q)*phi*1j*tau + kappa*theta/sigma**2 * (
        (b - rho*sigma*phi*1j - d)*tau - 2*np.log((1 - g*np.exp(-d*tau))/(1 - g))
    )
    D = (b - rho*sigma*phi*1j - d)/sigma**2 * ((1 - np.exp(-d*tau))/(1 - g*np.exp(-d*tau)))

    return np.exp(C + D*v + 1j*phi*np.log(S))


def heston_call_price(S, K, v, kappa, theta, sigma, rho, tau, r, q):
    """European call price under Heston"""
    def integrand1(phi):
        cf = heston_char_func(phi - 1j, S, v, kappa, theta, sigma, rho, tau, r, q)
        return np.real(np.exp(-1j*phi*np.log(K)) * cf / (1j*phi))

    def integrand2(phi):
        cf = heston_char_func(phi, S, v, kappa, theta, sigma, rho, tau, r, q)
        return np.real(np.exp(-1j*phi*np.log(K)) * cf / (1j*phi))

    try:
        P1 = 0.5 + 1/np.pi * quad(integrand1, 0, 100, limit=50)[0]
        P2 = 0.5 + 1/np.pi * quad(integrand2, 0, 100, limit=50)[0]
        call = S*np.exp(-q*tau)*P1 - K*np.exp(-r*tau)*P2
        return max(call, 0.0)
    except:
        return max(S*np.exp(-q*tau) - K*np.exp(-r*tau), 0.0)

# ============================================================================
# HESTON MONTE CARLO SIMULATION
# ============================================================================

def heston_monte_carlo_paths(S0, v0, kappa, theta, sigma, rho, r, q, T, N_steps, N_paths, random_seed=42):
    """
    Simulate Heston model paths using Euler discretization

    Returns: (S_paths, v_paths) where each is shape (N_paths, N_steps+1)
    """
    np.random.seed(random_seed)

    dt = T / N_steps

    # Initialize arrays
    S = np.zeros((N_paths, N_steps + 1))
    v = np.zeros((N_paths, N_steps + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    # Generate correlated random numbers
    for i in range(N_steps):
        # Two independent standard normals
        Z1 = np.random.standard_normal(N_paths)
        Z2 = np.random.standard_normal(N_paths)

        # Correlate them
        Z_S = Z1
        Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Update variance (with full truncation scheme)
        v[:, i+1] = v[:, i] + kappa * (theta - np.maximum(v[:, i], 0)) * dt + \
                    sigma * np.sqrt(np.maximum(v[:, i], 0) * dt) * Z_v
        v[:, i+1] = np.maximum(v[:, i+1], 0)  # Ensure non-negative variance

        # Update stock price
        S[:, i+1] = S[:, i] * np.exp((r - q - 0.5*np.maximum(v[:, i], 0)) * dt + \
                                      np.sqrt(np.maximum(v[:, i], 0) * dt) * Z_S)

    return S, v


# ============================================================================
# AUTOCALLABLE CONTINGENT COUPON PRODUCT VALUATION
# ============================================================================

def value_autocallable_mc(S_paths, params):
    """
    Value autocallable contingent coupon using Monte Carlo paths

    Product details (from prospectus):
    - Face value: $1000
    - Contingent coupon barrier: 70% of initial SPX level
    - Autocall barrier: 100% of initial SPX level (at the money)
    - Coupon rate: 7.3% per year (quarterly)
    - Autocall observation dates: quarterly
    - Maturity: ~4 years
    """
    Face = 1000
    coupon_rate = 0.073
    barrier_pct = 0.70
    autocall_pct = 1.00

    S0 = params['S0']
    r = params['r']
    q = params['q']

    contingent_barrier = barrier_pct * S0
    autocall_barrier = autocall_pct * S0

    # Quarterly coupon
    NC = 4  # Coupons per year
    Cpn = Face * coupon_rate / NC

    # Autocall observation dates (quarterly for ~4 years = 16 observations)
    # Simplified: quarterly dates at 0.25, 0.5, 0.75, 1.0, 1.25, ..., 4.0 years
    autocall_times = np.arange(0.25, 4.01, 0.25)  # 16 observations
    T_final = 4.0

    N_paths = S_paths.shape[0]
    N_steps = S_paths.shape[1] - 1
    dt = T_final / N_steps

    payoffs = np.zeros(N_paths)
    autocall_times_idx = [int(t/dt) for t in autocall_times if t/dt <= N_steps]

    for path_idx in range(N_paths):
        payoff = 0.0
        autocalled = False

        # Check each autocall date
        for ac_idx, step_idx in enumerate(autocall_times_idx):
            if step_idx >= N_steps:
                continue

            S_t = S_paths[path_idx, step_idx]
            t = autocall_times[ac_idx]

            # Check if autocalled
            if S_t >= autocall_barrier:
                # Autocall triggered: receive face + coupon, discounted
                payoff = (Face + Cpn) * np.exp(-r * t)
                autocalled = True
                break
            # If not autocalled but above contingent barrier, get coupon
            elif S_t >= contingent_barrier:
                payoff += Cpn * np.exp(-r * t)

        # If not autocalled, check final payoff
        if not autocalled:
            S_T = S_paths[path_idx, -1]

            # Final payment includes face value
            if S_T >= S0:
                # Above initial level: get face + coupon
                payoff += (Face + Cpn) * np.exp(-r * T_final)
            elif S_T >= contingent_barrier:
                # Above contingent barrier: get face + coupon
                payoff += (Face + Cpn) * np.exp(-r * T_final)
            else:
                # Below contingent barrier: get downside participation
                payoff += Face * (S_T / S0) * np.exp(-r * T_final)

        payoffs[path_idx] = payoff

    return np.mean(payoffs), np.std(payoffs) / np.sqrt(N_paths)


# ============================================================================
# BINOMIAL TREE VALUATION (USING HESTON IMPLIED VOLATILITY)
# ============================================================================

def binomial_tree_autocallable(S0, barrier_pct, autocall_pct, T, r, q, sigma, N_steps,
                                Face=1000, coupon_rate=0.073):
    """
    Value autocallable using binomial tree
    Uses constant volatility (Heston implied vol)
    """
    dt = T / N_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Build stock price tree
    stock = np.zeros((N_steps + 1, N_steps + 1))
    option = np.zeros((N_steps + 1, N_steps + 1))

    # Initialize stock prices at maturity
    for i in range(N_steps + 1):
        stock[N_steps, i] = S0 * (u ** i) * (d ** (N_steps - i))

    # Terminal payoffs
    Cpn = Face * coupon_rate / 4
    contingent_barrier = barrier_pct * S0

    for i in range(N_steps + 1):
        S_T = stock[N_steps, i]
        if S_T >= S0:
            option[N_steps, i] = Face + Cpn
        elif S_T >= contingent_barrier:
            option[N_steps, i] = Face + Cpn
        else:
            option[N_steps, i] = Face * S_T / S0

    # Backward induction
    autocall_steps = [int(0.25*i/dt) for i in range(1, 17) if 0.25*i <= T]

    for j in range(N_steps - 1, -1, -1):
        for i in range(j + 1):
            stock[j, i] = S0 * (u ** i) * (d ** (j - i))

            # Continuation value
            cont = np.exp(-r * dt) * (p * option[j+1, i+1] + (1-p) * option[j+1, i])

            # Check autocall
            if j in autocall_steps:
                if stock[j, i] >= autocall_pct * S0:
                    cont = Face + Cpn
                elif stock[j, i] >= contingent_barrier:
                    cont += Cpn

            option[j, i] = cont

    return option[0, 0]


# ============================================================================
# VALUATION WITH BOTH METHODS
# ============================================================================

print("\n" + "="*80)
print("AUTOCALLABLE CONTINGENT COUPON VALUATION")
print("="*80)

print("\nProduct Specifications:")
print("  Face Value: $1,000")
print("  Coupon Rate: 7.3% per year (1.825% quarterly)")
print("  Contingent Barrier: 70% of initial SPX")
print("  Autocall Barrier: 100% of initial SPX")
print("  Maturity: ~4 years")
print("  Observation: Quarterly")

# 1. Monte Carlo with Heston Model
print("\n" + "-"*80)
print("METHOD 1: MONTE CARLO SIMULATION WITH HESTON MODEL")
print("-"*80)

T_maturity = 3.897  # Approximate maturity from Dataset 1 (1422 days / 365)
N_steps_mc = 500
N_paths = 50000

print(f"\nSimulation parameters:")
print(f"  Number of paths: {N_paths:,}")
print(f"  Time steps: {N_steps_mc}")
print(f"  Maturity: {T_maturity:.4f} years")

print(f"\nSimulating {N_paths:,} Heston paths...")
S_paths, v_paths = heston_monte_carlo_paths(
    S0, v0, kappa, theta, sigma, rho, r, q,
    T_maturity, N_steps_mc, N_paths
)

params = {'S0': S0, 'r': r, 'q': q}
mc_value, mc_se = value_autocallable_mc(S_paths, params)

print(f"\nMonte Carlo Valuation:")
print(f"  Product Value: ${mc_value:.2f}")
print(f"  Standard Error: ${mc_se:.2f}")
print(f"  95% CI: [${mc_value - 1.96*mc_se:.2f}, ${mc_value + 1.96*mc_se:.2f}]")

# 2. Binomial Tree with Heston Implied Volatility
print("\n" + "-"*80)
print("METHOD 2: BINOMIAL TREE WITH HESTON IMPLIED VOLATILITY")
print("-"*80)

# Use ATM Heston implied vol
K_atm = S0
heston_atm_price = heston_call_price(S0, K_atm, v0, kappa, theta, sigma, rho, T_maturity, r, q)

# Back out implied vol from Heston price
def bs_call(S, K, T, r, q, vol):
    d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Newton-Raphson for implied vol
vol_guess = np.sqrt(v0)
for _ in range(20):
    bs_price = bs_call(S0, K_atm, T_maturity, r, q, vol_guess)
    vega = S0 * np.exp(-q*T_maturity) * norm.pdf((np.log(S0/K_atm) + (r-q+0.5*vol_guess**2)*T_maturity)/(vol_guess*np.sqrt(T_maturity))) * np.sqrt(T_maturity)
    if abs(vega) < 1e-10:
        break
    vol_guess = vol_guess - (bs_price - heston_atm_price) / vega

heston_impl_vol = vol_guess

print(f"\nHeston ATM implied volatility: {heston_impl_vol*100:.2f}%")
print(f"Binomial tree steps: 1500")

N_bin_steps = 1500
bin_value = binomial_tree_autocallable(
    S0, 0.70, 1.00, T_maturity, r, q, heston_impl_vol, N_bin_steps
)

print(f"\nBinomial Tree Valuation:")
print(f"  Product Value: ${bin_value:.2f}")

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nMonte Carlo (Heston): ${mc_value:.2f}")
print(f"Binomial Tree:        ${bin_value:.2f}")
print(f"Difference:           ${abs(mc_value - bin_value):.2f}")
print(f"Relative difference:  {abs(mc_value - bin_value)/mc_value*100:.2f}%")

# Prospectus estimate (from notebook)
prospectus_val = 965.2  # From term sheet
print(f"\nProspectus estimate:  ${prospectus_val:.2f}")
print(f"MC vs Prospectus:     ${mc_value - prospectus_val:.2f}")
print(f"Bin vs Prospectus:    ${bin_value - prospectus_val:.2f}")

# Save results
results = {
    'monte_carlo_value': mc_value,
    'monte_carlo_se': mc_se,
    'binomial_value': bin_value,
    'heston_params': {
        'v0': v0, 'kappa': kappa, 'theta': theta,
        'sigma': sigma, 'rho': rho, 'r': r, 'q': q, 'S0': S0
    }
}

np.savez('valuation_results.npz', **results)
print("\n" + "="*80)
print("Results saved to: valuation_results.npz")
print("="*80)
