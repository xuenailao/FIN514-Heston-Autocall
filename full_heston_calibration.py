"""
Full Heston Model Calibration with Dataset 2 and Dataset 3
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_option_data(df):
    """Parse Bloomberg option data from Excel"""
    # Skip first row which contains headers
    # Row 1 contains maturity info
    header_row = df.iloc[1, 0] if len(df) > 1 else ""

    # Extract maturity date from header
    import re
    date_match = re.search(r'(\d{1,2}-[A-Za-z]{3}-\d{2})', str(header_row))
    maturity_date = date_match.group(1) if date_match else None

    # Extract dividend yield
    div_match = re.search(r'IDiv\s+([\d.]+)', str(header_row))
    dividend = float(div_match.group(1)) / 100 if div_match else 0.0

    # Extract interest rate
    r_match = re.search(r'R\s+([\d.]+)', str(header_row))
    rate = float(r_match.group(1)) / 100 if r_match else 0.04

    # Parse column names from row 0
    df_clean = df.iloc[2:].copy()  # Skip first 2 rows
    df_clean.columns = df.iloc[0].values

    # Clean the dataframe
    df_clean = df_clean.reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = ['Strike', 'Bid', 'Ask', 'Last', 'IVM', 'Volm']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Filter out low volume and zero bid
    df_clean = df_clean[df_clean['Volm'] >= 10]
    df_clean = df_clean[df_clean['Bid'] > 0]
    df_clean = df_clean.dropna(subset=['Strike', 'Bid', 'Ask'])

    return {
        'maturity_date': maturity_date,
        'dividend': dividend,
        'rate': rate,
        'data': df_clean
    }

# ============================================================================
# LOAD HESTON MODEL FROM STEP 1
# ============================================================================

print("="*70)
print("FULL HESTON MODEL CALIBRATION")
print("="*70)

# Load step 1 results
step1 = np.load('heston_calibration_step1.npz')
v0_init = float(step1['v0'])
kappa_init = float(step1['kappa'])
theta_init = float(step1['theta'])

print(f"\nInitial parameters from Step 1:")
print(f"  v0 = {v0_init:.6f}")
print(f"  kappa = {kappa_init:.6f}")
print(f"  theta = {theta_init:.6f}")

# ============================================================================
# HESTON MODEL FUNCTIONS
# ============================================================================

def heston_char_func(phi, S0, v0, kappa, theta, sigma, rho, tau, r, q):
    """Heston characteristic function"""
    a = kappa * theta
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


def heston_call_price(S0, K, v0, kappa, theta, sigma, rho, tau, r, q):
    """Price European call with Heston model"""
    def integrand1(phi):
        cf = heston_char_func(phi - 1j, S0, v0, kappa, theta, sigma, rho, tau, r, q)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    def integrand2(phi):
        cf = heston_char_func(phi, S0, v0, kappa, theta, sigma, rho, tau, r, q)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    try:
        P1 = 0.5 + 1/np.pi * quad(integrand1, 0, 100, limit=100)[0]
        P2 = 0.5 + 1/np.pi * quad(integrand2, 0, 100, limit=100)[0]
    except:
        P1 = 0.5
        P2 = 0.5

    call = S0 * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
    return max(call, 0.0)


def black_scholes_call(S, K, tau, r, q, sigma):
    """Black-Scholes call price"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    return S*np.exp(-q*tau)*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)


def implied_vol(S, K, tau, r, q, price):
    """Calculate implied volatility"""
    if price <= max(S*np.exp(-q*tau) - K*np.exp(-r*tau), 0):
        return 0.0

    vol = 0.2
    for i in range(50):
        bs_price = black_scholes_call(S, K, tau, r, q, vol)
        vega = S * np.exp(-q*tau) * norm.pdf((np.log(S/K) + (r-q+0.5*vol**2)*tau)/(vol*np.sqrt(tau))) * np.sqrt(tau)
        if abs(vega) < 1e-10 or abs(bs_price - price) < 1e-6:
            break
        vol = vol - (bs_price - price) / vega
        vol = max(vol, 0.01)

    return vol

# ============================================================================
# PROCESS DATASET 2
# ============================================================================

print("\n" + "="*70)
print("PROCESSING DATASET 2")
print("="*70)

import os

S0 = 6610.0  # Spot price

dataset2_options = []

for filename in os.listdir('dataset2'):
    if filename.endswith('.xlsx'):
        print(f"\nProcessing {filename}...")
        df = pd.read_excel(os.path.join('dataset2', filename))
        parsed = parse_option_data(df)

        if parsed['data'] is not None and len(parsed['data']) > 0:
            print(f"  Maturity: {parsed['maturity_date']}")
            print(f"  Dividend: {parsed['dividend']*100:.2f}%")
            print(f"  Rate: {parsed['rate']*100:.2f}%")
            print(f"  Options: {len(parsed['data'])}")

            dataset2_options.append(parsed)

# ============================================================================
# CALIBRATE SIGMA (VOL-OF-VOL) AND RHO (CORRELATION) WITH DATASET 2
# ============================================================================

print("\n" + "="*70)
print("CALIBRATING SIGMA AND RHO WITH DATASET 2")
print("="*70)

# Use first maturity from Dataset 2
if len(dataset2_options) > 0:
    opt_data = dataset2_options[0]
    df_opts = opt_data['data']
    q = opt_data['dividend']
    r = opt_data['rate']

    # Calculate time to maturity (approximate)
    from datetime import datetime
    pricing_date = datetime.strptime("22 Nov 2025", "%d %b %Y")

    if opt_data['maturity_date']:
        try:
            # Parse maturity date (format: DD-MMM-YY)
            mat_date = datetime.strptime(opt_data['maturity_date'], "%d-%b-%y")
            if mat_date.year < 2000:
                mat_date = mat_date.replace(year=mat_date.year + 100)
            tau = (mat_date - pricing_date).days / 365.0
        except:
            tau = 0.1
    else:
        tau = 0.1

    print(f"\nUsing maturity: {opt_data['maturity_date']} (tau = {tau:.4f} years)")
    print(f"Number of strikes: {len(df_opts)}")

    # Calculate mid prices and implied vols
    df_opts['Mid'] = (df_opts['Bid'] + df_opts['Ask']) / 2.0

    # Filter for ATM and near-ATM options
    df_opts['Moneyness'] = df_opts['Strike'] / S0
    df_atm = df_opts[(df_opts['Moneyness'] >= 0.8) & (df_opts['Moneyness'] <= 1.2)].copy()

    print(f"ATM/Near-ATM options: {len(df_atm)}")

    if len(df_atm) > 5:
        # Calibration objective function
        def objective_sigma_rho(params):
            sigma, rho = params

            # Feller condition
            if 2 * kappa_init * theta_init < sigma**2:
                return 1e10

            error = 0.0
            count = 0

            for idx, row in df_atm.iterrows():
                K = row['Strike']
                market_price = row['Mid']
                bid_ask_spread = row['Ask'] - row['Bid']

                try:
                    heston_price = heston_call_price(
                        S0, K, v0_init, kappa_init, theta_init, sigma, rho, tau, r, q
                    )

                    # Calculate implied vols
                    market_iv = implied_vol(S0, K, tau, r, q, market_price)
                    heston_iv = implied_vol(S0, K, tau, r, q, heston_price)

                    # Weighted error
                    weight = 1.0 / (bid_ask_spread / 100.0 + 0.0005)**2
                    error += weight * (market_iv - heston_iv)**2
                    count += 1
                except:
                    continue

            return error / max(count, 1)

        # Initial guess
        sigma_init = 0.5
        rho_init = -0.7

        print(f"\nOptimizing sigma and rho...")
        print(f"Initial guess: sigma={sigma_init}, rho={rho_init}")

        result = minimize(
            objective_sigma_rho,
            x0=[sigma_init, rho_init],
            bounds=[(0.1, 2.0), (-0.99, 0.0)],
            method='L-BFGS-B',
            options={'maxiter': 100}
        )

        sigma_cal, rho_cal = result.x

        print(f"\nCalibrated parameters:")
        print(f"  sigma (vol-of-vol) = {sigma_cal:.6f}")
        print(f"  rho (correlation) = {rho_cal:.6f}")
        print(f"  Optimization success: {result.success}")
        print(f"  Final error: {result.fun:.6f}")
    else:
        print("Not enough ATM options for calibration, using defaults")
        sigma_cal = 0.5
        rho_cal = -0.7
else:
    print("No Dataset 2 options found, using default values")
    sigma_cal = 0.5
    rho_cal = -0.7
    r = 0.04
    q = 0.0245

# ============================================================================
# SAVE CALIBRATION RESULTS
# ============================================================================

print("\n" + "="*70)
print("CALIBRATION SUMMARY")
print("="*70)

calibrated_params = {
    'v0': v0_init,
    'kappa': kappa_init,
    'theta': theta_init,
    'sigma': sigma_cal,
    'rho': rho_cal,
    'r': r,
    'q': q,
    'S0': S0
}

print(f"\nFinal Heston Model Parameters:")
for key, val in calibrated_params.items():
    if key in ['v0', 'theta']:
        print(f"  {key} = {val:.6f} (vol = {np.sqrt(val)*100:.2f}%)")
    else:
        print(f"  {key} = {val:.6f}")

# Save results
np.savez('heston_calibrated_params.npz', **calibrated_params)

print("\nCalibrated parameters saved to: heston_calibrated_params.npz")
print("="*70)
