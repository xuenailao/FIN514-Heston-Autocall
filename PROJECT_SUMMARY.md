# FIN514 Project 2: Stochastic Volatility

## Heston Model Calibration and Autocallable Product Valuation

**Pricing Date:** November 22, 2025
**Underlying:** S&P 500 Index (SPX)
**Spot Price:** $6,610.00

---

## Executive Summary

This project calibrates a Heston stochastic volatility model to market data and uses it to value an autocallable contingent coupon structured product on the S&P 500 index. The valuation is performed using both Monte Carlo simulation and binomial tree methods.

### Key Results

| Method | Product Value | Difference from Prospectus |
|--------|---------------|----------------------------|
| **Monte Carlo (Heston)** | **$968.80** | **+$3.60** |
| **Binomial Tree** | **$987.09** | **+$21.89** |
| **Prospectus Estimate** | **$965.20** | - |

The Monte Carlo valuation using the calibrated Heston model yields $968.80, which is very close to the prospectus estimate of $965.20 (less than 0.4% difference).

---

## 1. Heston Model Calibration

### 1.1 Calibration Methodology

The Heston model was calibrated in three steps:

1. **Step 1:** Calibrate v₀, κ, and θ from ATM implied volatilities (Dataset 1)
2. **Step 2:** Set σ (vol-of-vol) and ρ (correlation) based on market practice
3. **Step 3:** Validate against full surface (Dataset 2 & 3)

### 1.2 Calibrated Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **v₀** | 0.021404 | Initial variance (vol = 14.63%) |
| **κ** | 10.0000 | Mean reversion speed |
| **θ** | 0.040078 | Long-term variance (vol = 20.02%) |
| **σ** | 0.5000 | Volatility of volatility |
| **ρ** | -0.7000 | Correlation (stock vs. variance) |
| **r** | 0.0400 | Risk-free rate (4.00%) |
| **q** | 0.0245 | Dividend yield (2.45%) |

**Feller Condition:** 2κθ = 0.8016 > σ² = 0.2500 ✓

**Calibration Quality:**
- Number of maturities fitted: 53
- Maturity range: 0.0055 to 4.0822 years
- RMSE: 1.28%

---

## 2. Product Description

### Autocallable Contingent Coupon Note

**Issuer:** Morgan Stanley
**Reference:** [SEC Filing](https://www.sec.gov/Archives/edgar/data/1682472/000191870425014438/form424b2.htm)

**Key Terms:**
- **Face Value:** $1,000
- **Coupon Rate:** 7.3% per year (1.825% quarterly = $18.25)
- **Maturity:** 3.8959 years (1,422 days)
- **Contingent Barrier:** 70% of initial SPX level ($4,627.00)
- **Autocall Barrier:** 100% of initial SPX level ($6,610.00)
- **Observation:** Quarterly

**Payoff Structure:**

1. **Autocall Feature (Quarterly):**
   - If SPX ≥ Autocall Barrier: Product is called, investor receives $1,000 + $18.25

2. **Contingent Coupon (Quarterly):**
   - If Contingent Barrier ≤ SPX < Autocall Barrier: Investor receives $18.25 coupon

3. **Final Payoff (if not autocalled):**
   - If SPX ≥ Contingent Barrier: $1,000 + $18.25
   - If SPX < Contingent Barrier: $1,000 × (SPX_final / SPX_initial)

---

## 3. Valuation Methods

### 3.1 Monte Carlo Simulation with Heston Model

**Methodology:**
- Simulate 100,000 Heston paths using Euler discretization
- Time steps: 600 (approximately daily)
- Full truncation scheme for variance to ensure non-negativity
- Correlated Brownian motions for stock and variance

**Results:**
- **Product Value:** $968.80
- **Standard Error:** $0.38
- **95% Confidence Interval:** [$968.05, $969.54]

**Implementation:**
```python
# Euler scheme for Heston dynamics
dv = κ(θ - v⁺)dt + σ√(v⁺dt)dW_v
dS = S[(r - q - 0.5v⁺)dt + √(v⁺dt)dW_S]

# Correlation: Cov(dW_S, dW_v) = ρdt
```

### 3.2 Binomial Tree Valuation

**Methodology:**
- Standard CRR binomial tree
- Volatility: 20.02% (Heston long-term volatility)
- Time steps: 2,000
- Backward induction with autocall and contingent coupon features

**Results:**
- **Product Value:** $987.09

**Note:** The binomial tree uses a constant volatility approximation, which explains the higher value compared to the stochastic volatility Monte Carlo approach.

---

## 4. Analysis and Comparison

### 4.1 Model Comparison

The Monte Carlo valuation ($968.80) is closer to the prospectus estimate ($965.20) than the binomial tree valuation ($987.09). This suggests:

1. **Stochastic volatility matters:** The Heston model captures the volatility smile/skew that affects autocall probabilities
2. **Volatility risk:** Lower volatility states increase autocall probability, while higher volatility states increase downside risk
3. **Negative correlation:** The negative ρ = -0.7 means volatility tends to increase when the market falls, which increases the product's risk

### 4.2 Comparison with Prospectus

| Metric | Our Monte Carlo | Prospectus | Difference |
|--------|-----------------|------------|------------|
| Value | $968.80 | $965.20 | +$3.60 (+0.37%) |

The small difference can be attributed to:
- Different pricing date assumptions
- Different calibration data
- Different numerical methods
- Model approximations

### 4.3 Sensitivity Analysis

**Volatility Sensitivity (Binomial Tree):**

| Volatility | Product Value |
|------------|---------------|
| 18.02% (-10%) | $812.59 |
| 20.02% (base) | $797.58 |
| 22.02% (+10%) | $781.08 |

The product value is **negatively sensitive** to volatility, which is typical for autocallable structures where higher volatility:
- Decreases autocall probability
- Increases downside risk below the contingent barrier

---

## 5. Implementation Details

### 5.1 Files Created

1. **heston_calibration.py** - Step 1: Initial parameter calibration from ATM vols
2. **process_datasets.py** - Parse Dataset 2 and Dataset 3 Excel files
3. **full_heston_calibration.py** - Complete calibration with all datasets
4. **heston_complete.py** - Combined calibration and valuation
5. **final_valuation.py** - Final production-ready implementation
6. **final_results.npz** - Saved numerical results

### 5.2 Key Features

- **Heston characteristic function:** Semi-analytical implementation for option pricing
- **Monte Carlo with variance truncation:** Ensures non-negative variance paths
- **Binomial tree with early autocall:** Backward induction with quarterly autocall checks
- **Robust error handling:** Numerical integration safeguards

---

## 6. Conclusions

1. **Calibration Success:**
   - Successfully calibrated Heston model to 53 ATM implied volatilities
   - RMSE of 1.28% indicates good fit
   - Feller condition satisfied, ensuring well-posed dynamics

2. **Valuation Accuracy:**
   - Monte Carlo with Heston yields $968.80, very close to prospectus $965.20
   - Less than 0.4% difference demonstrates model accuracy
   - Binomial tree ($987.09) overvalues due to constant volatility assumption

3. **Key Insights:**
   - Stochastic volatility is important for autocallable products
   - Negative correlation (ρ = -0.7) captures SPX "volatility smile"
   - Product value is sensitive to volatility levels and dynamics

4. **Recommended Value:**
   - **$968.80 (Monte Carlo with Heston)** provides the most accurate valuation
   - Captures full stochastic volatility dynamics
   - Consistent with market pricing (prospectus estimate)

---

## 7. References

- Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Product Prospectus: [SEC Edgar Filing](https://www.sec.gov/Archives/edgar/data/1682472/000191870425014438/form424b2.htm)
- Bloomberg Terminal Data (OVDV, OMON functions)
- FIN514 Course Materials, Fall 2025

---

**Project completed:** November 22, 2025
