import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pkpd_physics_stage3 import PKPDSimulator, generate_synthetic_population
from qnn_engine_stage3 import QuantumDosagePredictor

def find_optimal_dose(params_df, sim_engine, interval, threshold=3.3, percentile=90):
    """
    Finds the lowest dose (in 0.5mg increments) where 'percentile'% of subjects 
    maintain biomarker < threshold.
    """
    dose = 0.5
    max_dose = 500.0
    
    while dose < max_dose:
        success_count = 0
        total_patients = len(params_df)
        
        for _, row in params_df.iterrows():
            # Simulate at steady state (last interval of a long run) returns time, conc. We need conc < threshold.
            # Simulating 4 weeks to reach steady state
            t, bio = sim_engine.simulate_regimen(row.to_dict(), dose, interval, duration_hrs=24*28)
            
            # Check only the final interval (steady state)
            # Indices for last interval
            idx_start = np.argmax(t >= (24*28 - interval))
            bio_steady = bio[idx_start:]
            
            # If ANY point in steady state is > threshold, fail.
            # Requirement: "Suppression BELOW threshold throughout interval"
            if np.all(bio_steady < threshold):
                success_count += 1
        
        coverage = (success_count / total_patients) * 100
        if coverage >= percentile:
            return dose, coverage
        
        dose += 0.5
        
    return max_dose, 0.0

def main():
    print("=====================================================================")
    print("   STAGE 3: QUANTUM NEURAL NETWORK DOSAGE PREDICTION")
    print("=====================================================================\n")

    # ------------------------------------------------------------------
    # 1. Generate Synthetic Training Data (Stage 1/2 Proxy)
    # ------------------------------------------------------------------
    print("--- 1. Data Generation ---")
    X_train, Y_train, p_names = generate_synthetic_population(n=100, seed=42) # Keeping N small for Demo Speed
    X_test, Y_test, _ = generate_synthetic_population(n=20, seed=101)
    print(f"Generated {len(X_train)} training patients and {len(X_test)} test patients.")

    # ------------------------------------------------------------------
    # 2. Train Quantum Model
    # ------------------------------------------------------------------
    print("\n--- 2. Model Training (Quantum Kernel SVR) ---")
    qnn = QuantumDosagePredictor()
    qnn.fit(X_train, Y_train)

    # ------------------------------------------------------------------
    # 3. Validation
    # ------------------------------------------------------------------
    print("\n--- 3. Validation on Hold-out Set ---")
    Y_pred = qnn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R^2:  {r2:.4f}")
    
    # ------------------------------------------------------------------
    # 4. Clinical Optimization Outcomes (Phase 3 Questions)
    # ------------------------------------------------------------------
    print("\n=====================================================================")
    print("   CLINICAL OPTIMIZATION OUTCOMES")
    print("=====================================================================")
    
    sim = PKPDSimulator()
    
    # --- Scenario A: Standard Population (Question 1 & 2) ---
    print("\n[Scenario A] Standard Population (N=200)")
    X_std, _, _ = generate_synthetic_population(n=200, weight_range=(50,120), meds_prob=0.5, seed=2024)
    Y_std_pred = qnn.predict(X_std)
    df_std = pd.DataFrame(Y_std_pred, columns=p_names)
    
    # Q1: Once-Daily Dose (90% coverage)
    opt_dose_daily, cov_daily = find_optimal_dose(df_std, sim, interval=24, percentile=90)
    print(f"Q1 (Daily): Optimal Dose = {opt_dose_daily} mg (Coverage: {cov_daily:.1f}%)")
    
    # Q2: Once-Weekly Dose (90% coverage, multiples of 5mg). find_optimal_dose is 0.5 steps
    opt_dose_weekly, cov_weekly = find_optimal_dose(df_std, sim, interval=168, percentile=90)
    # Enforce 5mg multiples
    if opt_dose_weekly % 5 != 0:
        opt_dose_weekly = (int(opt_dose_weekly // 5) + 1) * 5
    print(f"Q2 (Weekly): Optimal Dose = {opt_dose_weekly} mg")

    # --- Scenario B: Heavy Weight Distribution (Question 3) ---
    print("\n[Scenario B] Heavy Weight Population (70-140kg) (N=200)")
    X_heavy, _, _ = generate_synthetic_population(n=200, weight_range=(70,140), meds_prob=0.5, seed=2025)
    Y_heavy_pred = qnn.predict(X_heavy)
    df_heavy = pd.DataFrame(Y_heavy_pred, columns=p_names)
    
    opt_dose_heavy, cov_heavy = find_optimal_dose(df_heavy, sim, interval=24, percentile=90)
    print(f"Q3 (Heavy, Daily): Optimal Dose = {opt_dose_heavy} mg")

    # --- Scenario C: No Concomitant Meds (Question 4) ---
    print("\n[Scenario C] No Concomitant Medications (N=200)")
    X_nomeds, _, _ = generate_synthetic_population(n=200, weight_range=(50,120), meds_prob=0.0, seed=2026)
    Y_nomeds_pred = qnn.predict(X_nomeds)
    df_nomeds = pd.DataFrame(Y_nomeds_pred, columns=p_names)
    
    opt_dose_nomeds, cov_nomeds = find_optimal_dose(df_nomeds, sim, interval=24, percentile=90)
    print(f"Q4 (No Meds, Daily): Optimal Dose = {opt_dose_nomeds} mg")

    # --- Scenario D: 75% Efficacy Target (Question 5) ---
    print("\n[Scenario D] 75% Population Coverage Target")
    opt_dose_75, cov_75 = find_optimal_dose(df_std, sim, interval=24, percentile=75)
    print(f"Q5 (Standard, Daily, 75% Cov): Optimal Dose = {opt_dose_75} mg")
    print(f"   -> Reduction from 90% target: {opt_dose_daily - opt_dose_75} mg")

    dataset_file = "Quantum_Innovation_PKPD_Dataset.rtf"
    
    try:
        from dataset_validation_stage3 import run_dataset_validation
        #  pass the 'qnn' object which has the trained SVR and Scalers
        validation_results = run_dataset_validation(dataset_file, qnn)

        pd.DataFrame(validation_results).to_csv("stage3_validation_metrics.csv", index=False)
        print("   Validation metrics saved to 'stage3_validation_metrics.csv'")
        
    except FileNotFoundError:
        print(f"\n[WARNING] Could not find '{dataset_file}'. Skipping dataset validation.")
    except Exception as e:
        print(f"\n[ERROR] Dataset validation failed: {e}")

if __name__ == "__main__":
    main()