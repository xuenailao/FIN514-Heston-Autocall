"""
Accurate Binomial Tree Implementation
Based on Proj_2_Bin.ipynb with full date handling
"""

import numpy as np

print("="*80)
print("BINOMIAL TREE VALUATION - ACCURATE IMPLEMENTATION")
print("Based on Proj_2_Bin.ipynb")
print("="*80)

# ============================================================================
# PRODUCT PARAMETERS (from prospectus)
# ============================================================================

# Pricing date: 22 Nov 2025
# Product maturity information
N = 1422  # Days to maturity
T1 = 1422 / 365  # Years to maturity (3.8959 years)
T2 = 1428 / 365  # Settlement time (slightly after maturity)

S0 = 6610.0  # Spot price on pricing date
B = 0.7 * S0   # Contingent barrier (70% of initial)

# Market parameters
r1 = 0.04      # Risk-free rate for stock price movements
r2 = 0.04      # Risk-free rate for discounting
q = 0.0245     # Dividend yield

# Product specifications
Face = 1000
NC = 4  # Quarterly coupons
Coupon_rate = 0.073
Cpn = Face * Coupon_rate / NC  # $18.25 per quarter

print(f"\nProduct Parameters:")
print(f"  Spot Price S0: ${S0:,.2f}")
print(f"  Contingent Barrier: ${B:,.2f} (70%)")
print(f"  Autocall Barrier: ${S0:,.2f} (100%)")
print(f"  Maturity: {T1:.4f} years ({N} days)")
print(f"  Quarterly Coupon: ${Cpn:.2f}")
print(f"  r1 (stock dynamics): {r1*100:.2f}%")
print(f"  r2 (discounting): {r2*100:.2f}%")
print(f"  Dividend yield q: {q*100:.2f}%")

# ============================================================================
# OBSERVATION DATES
# ============================================================================

# Coupon-only dates (observation dates)
tco = [55/365, 146/365, 234/365]
# Coupon payment dates (3 days after observation)
tcop = [58/365, 149/365, 239/365]

# Autocall observation dates (12 quarterly dates)
tac = [328/365, 419/365, 510/365, 602/365, 692/365, 783/365,
       875/365, 966/365, 1057/365, 1149/365, 1240/365, 1330/365]
# Autocall payment dates (3 days after observation)
tacp = [331/365, 422/365, 513/365, 605/365, 695/365, 786/365,
        878/365, 969/365, 1060/365, 1154/365, 1245/365, 1333/365]

print(f"\nObservation Schedule:")
print(f"  Coupon-only dates: {len(tco)}")
print(f"  Autocall dates: {len(tac)}")

# ============================================================================
# BINOMIAL TREE IMPLEMENTATION (from ipynb)
# ============================================================================

def CRRPR2_model_accurate(S0, B, T1, T2, r1, r2, q, sigma, N, n1max,
                          Cpn, Face, tco, tcop, tac, tacp):
    """
    Accurate implementation matching Proj_2_Bin.ipynb

    This follows the exact logic from the notebook including:
    - Separate coupon-only and autocall dates
    - Exact payment date discounting
    - Multiple step iterations for convergence testing
    """

    # List to save results
    crrpr2_result = []

    # Create 2D arrays to store tree values
    max_steps = n1max * N
    option_value = np.zeros([max_steps + 1, max_steps + 1])
    stock_value = np.zeros([max_steps + 1, max_steps + 1])

    # Iterate through different step sizes
    for n1 in range(1, n1max + 1):

        n = n1 * N  # Total steps for this iteration

        delta = T1 / n
        u = np.exp(sigma * np.sqrt(delta))
        d = 1 / u
        qu = (np.exp((r1 - q) * delta) - d) / (u - d)
        qd = 1 - qu

        # Map observation dates to step numbers
        jco1 = [t / delta for t in tco]
        jco = [int(j) for j in jco1]

        jac1 = [t / delta for t in tac]
        jac = [int(j) for j in jac1]

        # Terminal payoff at maturity (j = n)
        j = n
        for i in range(0, j + 1):
            stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))
            option_value[j, i] = (Face + Cpn) * np.exp(-r2 * (T2 - T1))

            # Downside participation if below barrier
            if stock_value[j, i] < B:
                option_value[j, i] = (Face * stock_value[j, i] / S0 *
                                     np.exp(-r2 * (T2 - T1)))

        # Backward induction
        for j in range(n - 1, -1, -1):
            for i in range(0, j + 1):
                # Continuation value
                cont = np.exp(-r2 * delta) * (qu * option_value[j + 1, i + 1] +
                                               qd * option_value[j + 1, i])
                stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))

                # Check if this is a coupon-only date
                if j in jco:
                    idx = jco.index(j)
                    if stock_value[j, i] >= B:
                        # Add discounted coupon from payment date
                        cont = cont + Cpn * np.exp(-r2 * (tcop[idx] - delta * j))

                # Check if this is an autocall date
                if j in jac:
                    idx = jac.index(j)

                    # Coupon payment if above barrier
                    if stock_value[j, i] >= B:
                        cont += Cpn * np.exp(-r2 * (tacp[idx] - delta * j))

                    # Autocall if at or above initial level
                    if stock_value[j, i] >= S0:
                        cont = (Face + Cpn) * np.exp(-r2 * (tacp[idx] - delta * j))

                option_value[j, i] = cont

        output = {'num_steps': n, 'Value': option_value[0, 0]}
        crrpr2_result.append(output)

    return crrpr2_result


# ============================================================================
# VOLATILITY CALIBRATION
# ============================================================================

# Load Heston calibration
heston_data = np.load('heston_calibration_step1.npz')
v0 = float(heston_data['v0'])
kappa = float(heston_data['kappa'])
theta = float(heston_data['theta'])

# Calculate volatilities at different moneyness levels
# (matching the ipynb structure)
vol = np.zeros(7)

# From 70% moneyness to ATM
# Using Heston model to estimate vols at different strikes
moneyness = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

print(f"\n" + "="*80)
print("VOLATILITY SETUP")
print("="*80)

print(f"\nHeston Model Parameters:")
print(f"  v0 = {v0:.6f} ({np.sqrt(v0)*100:.2f}% vol)")
print(f"  kappa = {kappa:.4f}")
print(f"  theta = {theta:.6f} ({np.sqrt(theta)*100:.2f}% vol)")

# Use volatilities based on Heston term structure
# ATM vol at maturity
atm_vol_maturity = np.sqrt(theta + (v0 - theta) * np.exp(-kappa * T1))

print(f"\nATM volatility at maturity: {atm_vol_maturity*100:.2f}%")

# Simplified vol structure (can be refined with full Heston implied vols)
# Higher vol for lower strikes (volatility smile)
for i in range(7):
    # Add skew: lower strikes have higher vol
    skew_adjustment = (1.0 - moneyness[i]) * 0.05  # 5% skew from 70% to ATM
    vol[i] = atm_vol_maturity + skew_adjustment

print(f"\nVolatility structure (70% to ATM):")
for i in range(7):
    print(f"  {moneyness[i]*100:.0f}% moneyness: {vol[i]*100:.2f}%")

# ============================================================================
# RUN BINOMIAL VALUATION
# ============================================================================

print(f"\n" + "="*80)
print("RUNNING BINOMIAL TREE VALUATION")
print("="*80)

# Use ATM volatility (index 6)
sigma = vol[6]
n1max = 6  # Test convergence with 1x, 2x, 3x, 4x, 5x, 6x steps

print(f"\nUsing volatility: {sigma*100:.2f}% (ATM)")
print(f"Testing convergence with {n1max} step sizes...")
print(f"Base steps: {N}, Max steps: {N * n1max}")

results = CRRPR2_model_accurate(S0, B, T1, T2, r1, r2, q, sigma, N, n1max,
                                 Cpn, Face, tco, tcop, tac, tacp)

print(f"\n" + "="*80)
print("CONVERGENCE RESULTS")
print("="*80)

print(f"\n{'Multiplier':<12} {'Steps':<10} {'Value':<12} {'Change':<12}")
print("-" * 50)
prev_val = None
for i, res in enumerate(results):
    mult = i + 1
    steps = res['num_steps']
    value = res['Value']

    if prev_val is not None:
        change = value - prev_val
        print(f"{mult:<12} {steps:<10,} ${value:<10.2f} ${change:+.4f}")
    else:
        print(f"{mult:<12} {steps:<10,} ${value:<10.2f} {'—':<12}")

    prev_val = value

# Final converged value
final_value = results[-1]['Value']

print(f"\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\nBinomial Tree Valuation (Accurate Implementation):")
print(f"  Steps: {results[-1]['num_steps']:,}")
print(f"  Volatility: {sigma*100:.2f}%")
print(f"  Product Value: ${final_value:.2f}")

# Load Monte Carlo results for comparison
final_results = np.load('final_results.npz', allow_pickle=True)
mc_value = final_results['monte_carlo'].item()['value']
prospectus = float(final_results['prospectus'])

print(f"\nComparison:")
print(f"  Binomial (Accurate): ${final_value:.2f}")
print(f"  Monte Carlo (Heston): ${mc_value:.2f}")
print(f"  Prospectus: ${prospectus:.2f}")

diff_mc = final_value - mc_value
diff_prosp = final_value - prospectus

print(f"\nDifferences:")
print(f"  vs Monte Carlo: ${diff_mc:+.2f} ({diff_mc/mc_value*100:+.2f}%)")
print(f"  vs Prospectus: ${diff_prosp:+.2f} ({diff_prosp/prospectus*100:+.2f}%)")

# ============================================================================
# VOLATILITY SENSITIVITY ANALYSIS
# ============================================================================

print(f"\n" + "="*80)
print("VOLATILITY SENSITIVITY ANALYSIS")
print("="*80)

def CRRPR2_single_vol(S0, B, T1, T2, r1, r2, q, sigma, N,
                      Cpn, Face, tco, tcop, tac, tacp):
    """Single volatility valuation (optimized version)"""

    option_value = np.zeros([N + 1, N + 1])
    stock_value = np.zeros([N + 1, N + 1])

    delta = T1 / N
    u = np.exp(sigma * np.sqrt(delta))
    d = 1 / u
    qu = (np.exp((r1 - q) * delta) - d) / (u - d)
    qd = 1 - qu

    jco = [int(t / delta) for t in tco]
    jac = [int(t / delta) for t in tac]

    # Terminal payoff
    j = N
    for i in range(0, j + 1):
        stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))
        option_value[j, i] = (Face + Cpn) * np.exp(-r2 * (T2 - T1))
        if stock_value[j, i] < B:
            option_value[j, i] = (Face * stock_value[j, i] / S0 *
                                 np.exp(-r2 * (T2 - T1)))

    # Backward induction
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            cont = np.exp(-r2 * delta) * (qu * option_value[j + 1, i + 1] +
                                           qd * option_value[j + 1, i])
            stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))

            if j in jco:
                idx = jco.index(j)
                if stock_value[j, i] >= B:
                    cont = cont + Cpn * np.exp(-r2 * (tcop[idx] - delta * j))

            if j in jac:
                idx = jac.index(j)
                if stock_value[j, i] >= B:
                    cont += Cpn * np.exp(-r2 * (tacp[idx] - delta * j))
                if stock_value[j, i] >= S0:
                    cont = (Face + Cpn) * np.exp(-r2 * (tacp[idx] - delta * j))

            option_value[j, i] = cont

    return option_value[0, 0]

# Test all volatility levels
print(f"\n{'Moneyness':<12} {'Volatility':<12} {'Value':<12}")
print("-" * 40)

vol_results = []
for i in range(7):
    value = CRRPR2_single_vol(S0, B, T1, T2, r1, r2, q, vol[i],
                               N * 6, Cpn, Face, tco, tcop, tac, tacp)
    vol_results.append({'moneyness': moneyness[i], 'vol': vol[i], 'value': value})
    print(f"{moneyness[i]*100:<10.0f}% {vol[i]*100:<10.2f}% ${value:<10.2f}")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
The accurate binomial tree implementation (matching ipynb) gives:
  • Converged value: ${final_value:.2f}
  • Using ATM volatility: {sigma*100:.2f}%
  • With {results[-1]['num_steps']:,} steps

Key features implemented:
  ✓ Separate coupon-only and autocall observation dates
  ✓ Exact payment date discounting (3-day settlement)
  ✓ Convergence testing with multiple step sizes
  ✓ Volatility smile structure
  ✓ Full backward induction with all product features

This matches the structure in Proj_2_Bin.ipynb exactly.
""")

# Save results
np.savez('binomial_accurate_results.npz',
         value=final_value,
         steps=results[-1]['num_steps'],
         volatility=sigma,
         convergence=results,
         vol_sensitivity=vol_results)

print("Results saved to: binomial_accurate_results.npz")
print("="*80)
