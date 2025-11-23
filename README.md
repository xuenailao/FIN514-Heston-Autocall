# FIN514 Project 2: Heston Model Calibration and Valuation

## Quick Start

To run the complete project:

```bash
python3 final_valuation.py
```

This will:
1. Calibrate the Heston model from Dataset 1 (ATM volatilities)
2. Value the autocallable contingent coupon product using Monte Carlo simulation
3. Value the product using a binomial tree
4. Perform sensitivity analysis
5. Save results to `final_results.npz`

---

## Project Structure

```
FIN514/
├── Dataset1.txt                    # ATM implied vols and forward prices
├── dataset2/                       # Full option surface (2 maturities)
│   ├── 25 Dec 19 All.xlsx
│   └── 26 Mar 20 All.xlsx
├── dataset3/                       # Quarterly maturities (~25 strikes each)
│   ├── 260320.xlsx
│   ├── 260618.xlsx
│   ├── 260918.xlsx
│   ├── 261218.xlsx
│   ├── 270617.xlsx
│   ├── 271217.xlsx
│   └── 291221.xlsx
├── Proj_2_Bin.ipynb               # Original binomial notebook (reference)
├── heston_calibration.py          # Step 1: Calibrate v0, kappa, theta
├── process_datasets.py            # Parse Excel files
├── full_heston_calibration.py     # Full calibration (all parameters)
├── heston_complete.py             # Complete implementation
├── final_valuation.py             # MAIN SCRIPT - Run this!
├── binomial_accurate.py           # Accurate binomial (matches ipynb)
├── PROJECT_SUMMARY.md             # Detailed project report
└── README.md                      # This file
```

---

## Running Individual Steps

### Step 1: Initial Calibration (v0, kappa, theta)

```bash
python3 heston_calibration.py
```

**Output:**
- Prints calibrated parameters
- Saves `heston_calibration_step1.npz`

### Step 2: Process Excel Datasets

```bash
python3 process_datasets.py
```

**Output:**
- Reads Dataset 2 and Dataset 3 Excel files
- Saves `processed_datasets.pkl`

### Step 3: Complete Valuation

```bash
python3 final_valuation.py
```

**Output:**
- Complete calibration
- Monte Carlo valuation
- Binomial tree valuation
- Sensitivity analysis
- Saves `final_results.npz`

### Step 4: Accurate Binomial Implementation (matches ipynb)

```bash
python3 binomial_accurate.py
```

**Output:**
- Exact implementation matching Proj_2_Bin.ipynb
- Separate coupon-only and autocall dates
- Convergence testing (6 step sizes)
- Volatility sensitivity analysis
- Saves `binomial_accurate_results.npz`

---

## Results

### Heston Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| v₀ | 0.021404 | Initial variance (14.63% vol) |
| κ | 10.0000 | Mean reversion speed |
| θ | 0.040078 | Long-term variance (20.02% vol) |
| σ | 0.5000 | Vol-of-vol |
| ρ | -0.7000 | Correlation |

### Product Valuation

| Method | Value | vs. Prospectus |
|--------|-------|----------------|
| **Monte Carlo (Heston)** | **$968.80** | **+0.37%** ⭐ |
| **Binomial Tree (Simplified)** | **$987.09** | +2.27% |
| **Binomial Tree (Accurate/ipynb)** | **$993.00** | +2.88% |
| **Prospectus Estimate** | **$965.20** | — |

The Monte Carlo valuation with the calibrated Heston model ($968.80) is within 0.4% of the prospectus estimate, making it the most accurate method.

**Note:** The binomial tree implementations use constant volatility and therefore overvalue the product compared to the stochastic volatility Monte Carlo approach.

---

## Dependencies

```bash
# Python 3.9+
numpy>=2.0.2
scipy>=1.13.1
pandas>=2.3.3
openpyxl>=3.1.5
```

Install dependencies:
```bash
pip3 install --user numpy scipy pandas openpyxl
```

---

## Key Files Explanation

### Dataset1.txt
Contains ATM implied volatilities and forward prices for 53 maturities (from Bloomberg OVDV). Used for initial calibration of mean-reverting variance parameters.

### dataset2/
Contains full option chains (all strikes) for two specific maturities:
- 19 Dec 2025 (27 days)
- 20 Mar 2026 (118 days)

Used for calibrating vol-of-vol (σ) and correlation (ρ).

### dataset3/
Contains ~25 strikes at quarterly maturities through Dec 2029. Used for final validation and full surface calibration.

### final_valuation.py
**Main script** that performs:
1. Heston model calibration from Dataset 1
2. Monte Carlo simulation (100,000 paths)
3. Binomial tree valuation (2,000 steps)
4. Sensitivity analysis
5. Comparison with prospectus estimate

---

## Product Details

**Autocallable Contingent Coupon Note on SPX**

- **Issuer:** Morgan Stanley
- **Face Value:** $1,000
- **Coupon:** 7.3% annual (1.825% quarterly)
- **Maturity:** 3.8959 years (1,422 days)
- **Pricing Date:** 22 Nov 2025
- **Spot:** $6,610.00

**Barriers:**
- **Contingent Barrier:** $4,627 (70% of spot)
- **Autocall Barrier:** $6,610 (100% of spot)

**Payoff:**
- Quarterly autocall check: If SPX ≥ $6,610 → receive $1,018.25
- Quarterly coupon: If SPX ≥ $4,627 (and not autocalled) → receive $18.25
- Final payoff: If not autocalled and SPX < $4,627 → receive $1,000 × (SPX/6,610)

---

## Monte Carlo Implementation

The Heston Monte Carlo uses:

1. **Euler discretization** with full truncation scheme
2. **100,000 paths** for statistical accuracy
3. **600 time steps** (approximately daily)
4. **Correlated Brownian motions** (ρ = -0.7)
5. **Variance truncation** to ensure v(t) ≥ 0

### Dynamics

```
dv(t) = κ[θ - v⁺(t)]dt + σ√[v⁺(t)dt]dW_v
dS(t) = S(t)[(r - q - 0.5v⁺(t))dt + √[v⁺(t)dt]dW_S]

where v⁺(t) = max(v(t), 0)
      Corr(dW_S, dW_v) = ρdt
```

---

## Binomial Tree Implementation

The binomial tree uses:

1. **CRR (Cox-Ross-Rubinstein) framework**
2. **2,000 time steps** for accuracy
3. **Constant volatility** = 20.02% (Heston long-term vol)
4. **Backward induction** with quarterly autocall checks

---

## Validation

✅ Feller condition satisfied: 2κθ = 0.8016 > σ² = 0.25
✅ Calibration RMSE: 1.28%
✅ Monte Carlo SE: $0.38 (tight confidence interval)
✅ Close to prospectus: 0.37% difference

---

## References

- Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
- Product prospectus: https://www.sec.gov/Archives/edgar/data/1682472/000191870425014438/form424b2.htm
- Bloomberg data: OVDV (volatility surface), OMON (option monitor)

---

## Author

FIN514 Fall 2025
University of Illinois Urbana-Champaign

---

## License

This project is for educational purposes (FIN514 coursework).
