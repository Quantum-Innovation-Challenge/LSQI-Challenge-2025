import pennylane as qml
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# ==============================================================================
# QUANTUM KERNEL DEFINITION
# ==============================================================================
# Use default.qubit (CPU) for compatibility, change to "lightning.gpu" if cuQuantum installed.
dev_kernel = qml.device("default.qubit", wires=2)

@qml.qnode(dev_kernel)
def quantum_kernel_circuit(x1, x2):
    """
    Quantum Feature Map: ZZFeatureMap-inspired structure.
    Maps classical data x into Hilbert space. Kernel is overlap |<phi(x1)|phi(x2)>|^2.
    """
    # Encoding x1
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(x1[0], wires=0)
    qml.RZ(x1[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ( (np.pi - x1[0]) * (np.pi - x1[1]), wires=1) # Interaction term
    
    # Inverse Encoding x2 (Adjoint)
    qml.RZ( -1 * (np.pi - x2[0]) * (np.pi - x2[1]), wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(-x2[1], wires=1)
    qml.RZ(-x2[0], wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=0)
    
    return qml.probs(wires=range(2))

def kernel_matrix_func(A, B):
    """Computes the Gram matrix for SVR using the Quantum Circuit."""
    # A: (N, 2), B: (M, 2)
    # Returns (N, M)
    return np.array([[quantum_kernel_circuit(a, b)[0] for b in B] for a in A])

# ==============================================================================
# QNN REGRESSOR CLASS
# ==============================================================================
class QuantumDosagePredictor:
    def __init__(self):
        # We wrap the SVR in a MultiOutputRegressor to predict all 7 parameters at once
        # Using precomputed kernel means we pass the quantum kernel matrix
        self.model = MultiOutputRegressor(SVR(kernel='precomputed', C=10.0, epsilon=0.1))
        self.scaler_x = MinMaxScaler(feature_range=(0, np.pi)) # Scale for Quantum Gates
        self.scaler_y = StandardScaler() # Crucial for regression convergence
        self.X_train_scaled = None

    def fit(self, X_train, Y_train):
        print("   [QNN] Scaling Data...")
        self.X_train_scaled = self.scaler_x.fit_transform(X_train)
        Y_train_scaled = self.scaler_y.fit_transform(Y_train)
        
        print(f"   [QNN] Computing Quantum Kernel Matrix ({len(X_train)}x{len(X_train)})...")
        # Self-kernel
        K_train = kernel_matrix_func(self.X_train_scaled, self.X_train_scaled)
        
        print("   [QNN] Fitting SVR...")
        self.model.fit(K_train, Y_train_scaled)
        print("   [QNN] Training Complete.")

    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        
        # Compute kernel between Test and Train data (The "Support Vectors")
        K_test = kernel_matrix_func(X_test_scaled, self.X_train_scaled)
        
        Y_pred_scaled = self.model.predict(K_test)
        return self.scaler_y.inverse_transform(Y_pred_scaled)