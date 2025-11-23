#!/bin/bash

# FIN514 Project 2 - Complete Execution Script
# This script runs the entire project from start to finish

echo "================================================================================"
echo "                      FIN514 PROJECT 2 - EXECUTION SCRIPT"
echo "                 Heston Model Calibration and Product Valuation"
echo "================================================================================"
echo ""

# Step 1: Initial Calibration
echo "Step 1: Calibrating Heston parameters (v0, kappa, theta) from Dataset 1..."
echo "--------------------------------------------------------------------------------"
python3 heston_calibration.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 failed!"
    exit 1
fi
echo ""

# Step 2: Process Datasets
echo "Step 2: Processing Dataset 2 and Dataset 3 (Excel files)..."
echo "--------------------------------------------------------------------------------"
python3 process_datasets.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed!"
    exit 1
fi
echo ""

# Step 3: Complete Valuation
echo "Step 3: Running complete valuation (Monte Carlo + Binomial Tree)..."
echo "--------------------------------------------------------------------------------"
python3 final_valuation.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed!"
    exit 1
fi
echo ""

# Step 4: Generate Visualizations
echo "Step 4: Generating visualizations..."
echo "--------------------------------------------------------------------------------"
python3 visualize_results.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 failed!"
    exit 1
fi
echo ""

# Step 5: Display Results
echo "Step 5: Displaying final results..."
echo "--------------------------------------------------------------------------------"
python3 show_results.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 5 failed!"
    exit 1
fi
echo ""

# Summary
echo "================================================================================"
echo "                            EXECUTION COMPLETE"
echo "================================================================================"
echo ""
echo "Generated Files:"
echo "  - heston_calibration_step1.npz   : Initial calibration results"
echo "  - processed_datasets.pkl         : Parsed Excel data"
echo "  - final_results.npz              : Complete valuation results"
echo "  - heston_results.png             : Main results dashboard"
echo "  - heston_sample_paths.png        : Sample Heston path visualization"
echo ""
echo "Documentation:"
echo "  - README.md                      : Project documentation"
echo "  - PROJECT_SUMMARY.md             : Detailed report"
echo ""
echo "To view results again, run: python3 show_results.py"
echo "================================================================================"
