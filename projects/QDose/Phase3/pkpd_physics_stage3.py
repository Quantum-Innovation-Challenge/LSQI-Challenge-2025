import numpy as np
import pandas as pd
from scipy.integrate import odeint

# ==============================================================================
# CLASS: PK/PD Simulator (The "Ground Truth" Physics)
# ==============================================================================
class PKPDSimulator:
    """
    Defines the Two-Compartment PK model linked to an Indirect Response PD model.
    Used for generating synthetic ground truth and simulating clinical trials.
    """
    def __init__(self):
        # Biomarker Baseline (R0) = kin / kout
        pass

    @staticmethod
    def model_odes(y, t, params, dose_amount, dose_interval):
        """
        Differential equations for 2-Comp PK + Indirect Response PD (Inhibition).
        State y: [Ac (Central), Ap (Peripheral), R (Response/Biomarker)]
        """
        Ac, Ap, R = y
        CL, Vc, Vp, Q, kin, kout, IC50 = params
        
        # Approximating infusion/dosing as instantaneous input for ODE continuity 
        # (Real dosing handled in simulation loop)
        Cp = Ac / Vc
        
        # PK Equations (2-Compartment)
        # dAc/dt = -k10*Ac - k12*Ac + k21*Ap
        k10 = CL / Vc
        k12 = Q / Vc
        k21 = Q / Vp
        
        dAc_dt = -k10 * Ac - k12 * Ac + k21 * Ap
        dAp_dt =  k12 * Ac - k21 * Ap
        
        # PD Equation (Indirect Response, Inhibition of production Imax=1)
        # dR/dt = kin * (1 - Cp/(IC50 + Cp)) - kout * R
        # Simplified Inhibition: kin * (1 - Imax * Cp / (IC50 + Cp))
        inhibition = Cp / (IC50 + Cp)
        dR_dt = kin * (1 - inhibition) - kout * R
        
        return [dAc_dt, dAp_dt, dR_dt]

    def simulate_regimen(self, params_dict, dose_mg, interval_hrs, duration_hrs=24*7):
        """Simulates a dosing regimen for a single patient."""
        # Unpack parameters
        p = [params_dict[k] for k in ['CL', 'Vc', 'Vp', 'Q', 'kin', 'kout', 'IC50']]
        
        t_points = np.linspace(0, interval_hrs, 100) # High res for accuracy
        total_doses = int(duration_hrs / interval_hrs)
        
        # Initial State (Steady State assumed 0 drug, Baseline Response)
        R0 = params_dict['kin'] / params_dict['kout']
        y0 = [0.0, 0.0, R0]
        
        history = {'time': [], 'biomarker': []}
        current_time = 0
        
        for i in range(total_doses):
            # Administer Dose (IV Bolus assumption for stability or Short Infusion)
            # Add dose to Central Compartment (Ac)
            y0[0] += dose_mg * 1000 # Convert mg to ng (assuming Volume in L, Conc in ng/mL? scaling needed)
            # Correction: If Conc is ng/mL and Vol is L. Amount = ng. 
            # Dose mg -> * 1e6 ng. 
            # Let's standardize: Dose in mg, Vol in L, Conc in mg/L (equivalent to ug/mL).
            # To get ng/mL, we multiply mg/L * 1000.
            # Let's keep system in mg and L. Final output * 1000 for ng/mL.
            
            # Solve ODE for this interval
            sol = odeint(self.model_odes, y0, t_points, args=(p, 0, 0))
            
            # Store results
            history['time'].append(t_points + current_time)
            history['biomarker'].append(sol[:, 2]) # R state
            
            # Update state for next loop
            y0 = sol[-1]
            current_time += interval_hrs
            
        # Flatten
        time_flat = np.concatenate(history['time'])
        bio_flat = np.concatenate(history['biomarker'])
        
        return time_flat, bio_flat

# ==============================================================================
# DATA GENERATOR
# ==============================================================================
def generate_synthetic_population(n=500, weight_range=(50, 120), meds_prob=0.5, seed=42):
    """
    Generates (X, Y) data Calibrated to the Challenge Dataset.
    """
    np.random.seed(seed)
    
    # Covariates
    weights = np.random.uniform(weight_range[0], weight_range[1], n)
    meds = np.random.binomial(1, meds_prob, n)
    
    X = np.vstack([weights, meds]).T
    
    # --- CALIBRATED PARAMETERS (Based on Dataset Snippet) ---
    # Baseline Biomarker (R0) ~ 15-20. 
    # Let's set kin=2.0, kout=0.1 -> R0=20.
    pop_params = {
        'CL': 0.15,   # Increased Clearance (harder to keep drug in system)
        'Vc': 5.0,    # Increased Volume
        'Vp': 12.0,   
        'Q':  0.50,   
        'kin': 2.0,   # Production rate
        'kout': 0.12, # Elimination rate (Baseline ~ 16.6)
        'IC50': 25.0  # Increased IC50 (Lower potency -> Requires higher dose)
    }
    
    Y_list = []
    
    for w, m in zip(weights, meds):
        # Variability
        eta = np.random.normal(0, 0.2, 7) # 20% Variability
        
        # Covariate Effects
        # Meds = 1 might INCREASE Clearance (making it harder to treat)
        med_factor = 1.3 if m == 1 else 1.0 
        
        cl = pop_params['CL'] * (w/70)**0.75 * med_factor * np.exp(eta[0])
        vc = pop_params['Vc'] * (w/70)**1.0 * np.exp(eta[1])
        vp = pop_params['Vp'] * (w/70)**1.0 * np.exp(eta[2])
        q  = pop_params['Q']  * (w/70)**0.75 * np.exp(eta[3])
        
        kin  = pop_params['kin'] * np.exp(eta[4])
        kout = pop_params['kout'] * np.exp(eta[5])
        ic50 = pop_params['IC50'] * np.exp(eta[6])
        
        Y_list.append([cl, vc, vp, q, kin, kout, ic50])
        
    Y = np.array(Y_list)
    param_names = ['CL', 'Vc', 'Vp', 'Q', 'kin', 'kout', 'IC50']
    
    return X, Y, param_names