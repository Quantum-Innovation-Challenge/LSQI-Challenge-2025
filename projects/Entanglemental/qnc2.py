"""
qnc2.py

Variational Quantum Circuit (VQC) to replace the MLP in NeuralODE_v1.ipynb.


Outputs and parameter mapping context:
- PK features (5): Ka, CL, Vc, Q, Vp
- PD features (4): Kin, Kout, Imax, IC50
- The notebook defines bounds used to scale parameters. We include matching default bounds here so that the
  9 quantum outputs in [-1, 1] can be mapped to these physical ranges if desired.


"""
from __future__ import annotations

import typing as _t

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

# --------------------------------------------------------------------------------------
# Wire layout
# --------------------------------------------------------------------------------------
# Total wires = 12
#   - Initial wires: 3 (init_0, init_1, init_2)
#   - PK wires: 5 (PK_0 .. PK_4)
#   - PD wires: 4 (PD_0 .. PD_3)
# Measurements are performed on the 9 ancillary wires (5 PK + 4 PD), yielding 9 outputs.

INIT_WIRES = ["init_0", "init_1", "init_2"]
PK_WIRES = [f"PK_{i}" for i in range(5)]
PD_WIRES = [f"PD_{i}" for i in range(4)]
ALL_WIRES = INIT_WIRES + PK_WIRES + PD_WIRES

# --------------------------------------------------------------------------------------
# Default bounds (matching NeuralODE_v1.ipynb ranges)
# These are provided so that VQC outputs in [-1, 1] can be mapped to meaningful ranges if desired.
# --------------------------------------------------------------------------------------
DEFAULT_PK_BOUNDS = pnp.array(
    [
        [0.5, 5.0],    # Ka: 0.5-5.0 /h
        [2.0, 20.0],   # CL: 2-20 L/h
        [20.0, 100.0], # Vc: 20-100 L
        [2.0, 30.0],   # Q: 2-30 L/h
        [40.0, 200.0], # Vp: 40-200 L
    ], dtype=float
)

DEFAULT_PD_BOUNDS = pnp.array(
    [
        [5.0, 25.0],   # Kin: 5-25 ng/mL/h
        [0.3, 3.0],    # Kout: 0.3-3.0 /h
        [0.3, 0.95],   # Imax: 30-95% max inhibition
        [1.0, 15.0],   # IC50: 1-15 ng/mL
    ], dtype=float
)

# --------------------------------------------------------------------------------------
# Parameterization
# --------------------------------------------------------------------------------------
# We define L layers. Each layer contains:
#   - Local Euler rotations on the 3 initial qubits: RX, RY, RZ (=> 3 params/qubit => 9 params/layer)
#   - Entangling CRY from each initial qubit to each PK qubit (=> 3*5 = 15 params/layer)
#   - Entangling CRY from each initial qubit to each PD qubit (=> 3*4 = 12 params/layer)
#   - Local RY on each PK qubit (=> 5 params/layer)
#   - Local RY on each PD qubit (=> 4 params/layer)
# Total parameters per layer = 9 + 15 + 12 + 5 + 4 = 45.
# With L=2, total = 90 (< 135 as required). This is a Variational Quantum Circuit (VQC).

class VariationalPKPDAnsatz:
    """Encapsulates a layered variational circuit topology that evenly entangles
    the 3 initial qubits with the 5 PK and then the 4 PD qubits.

    Parameters
    ----------
    layers : int
        Number of variational layers. Each layer contributes 45 parameters.
        Must satisfy layers*45 <= 135 to meet the requirement.
    device_shots : int | None
        Number of shots for the device. If None, analytic expectations are used.
    """

    def __init__(self, layers: int = 2, device_shots: int | None = None):
        assert layers >= 1, "layers must be >= 1"
        assert layers * 45 <= 135, "Parameter budget exceeded: layers*45 must be <= 135"
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=ALL_WIRES, shots=device_shots)

        # Build the QNode upon initialization
        self._qnode = qml.QNode(self._circuit, self.dev, interface="auto")

    # -----------------------------------------
    # Shapes and initialization of variational parameters
    # -----------------------------------------
    def weights_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the expected shapes for the weights dictionary.

        Returns
        -------
        dict
            Keys and shapes:
            - 'init_euler' : (layers, 3, 3)   # [layer][init_qubit][axis(RX,RY,RZ)]
            - 'pk_coup'    : (layers, 3, 5)   # [layer][init_qubit][pk_qubit]
            - 'pd_coup'    : (layers, 3, 4)   # [layer][init_qubit][pd_qubit]
            - 'pk_local'   : (layers, 5)      # [layer][pk_qubit]
            - 'pd_local'   : (layers, 4)      # [layer][pd_qubit]
        """
        L = self.layers
        return {
            "init_euler": (L, 3, 3),
            "pk_coup": (L, 3, 5),
            "pd_coup": (L, 3, 4),
            "pk_local": (L, 5),
            "pd_local": (L, 4),
        }

    def init_weights(self, seed: int | None = 42) -> dict[str, pnp.ndarray]:
        """Initialize weights with small random values.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.
        """
        if seed is not None:
            pnp.random.seed(seed)
        shapes = self.weights_shapes()
        # Small random initial angles
        return {
            name: 0.1 * pnp.random.randn(*shape) for name, shape in shapes.items()
        }

    def count_parameters(self, weights: dict[str, pnp.ndarray]) -> int:
        """Count the total number of trainable parameters in the circuit weights."""
        return int(sum(p.size for p in weights.values()))

    # -----------------------------------------
    # Circuit definition
    # -----------------------------------------
    @staticmethod
    def _encode_inputs(inputs: pnp.ndarray) -> None:
        """Encode classical inputs on the 3 initial qubits with simple RY rotations.

        Parameters
        ----------
        inputs : array-like, shape (3,)
            Three classical features to seed the initial qubits.
        """
        assert len(inputs) == 3, "inputs must have length 3 (for 3 initial qubits)"
        for i, w in enumerate(INIT_WIRES):
            qml.RY(inputs[i], wires=w)

    @staticmethod
    def _entangle_initials() -> None:
        """Static entanglement among the 3 initial qubits (no extra parameters).
        Adds a ring of CNOTs to spread correlations before coupling to ancillae."""
        qml.CNOT(wires=[INIT_WIRES[0], INIT_WIRES[1]])
        qml.CNOT(wires=[INIT_WIRES[1], INIT_WIRES[2]])
        qml.CNOT(wires=[INIT_WIRES[2], INIT_WIRES[0]])

    def _layer(self, layer_idx: int, weights: dict[str, pnp.ndarray]) -> None:
        """Apply one variational layer using the provided weights."""
        L_euler = weights["init_euler"][layer_idx]
        W_pk = weights["pk_coup"][layer_idx]
        W_pd = weights["pd_coup"][layer_idx]
        pk_local = weights["pk_local"][layer_idx]
        pd_local = weights["pd_local"][layer_idx]

        # 1) Local Euler rotations on initial qubits
        for q in range(3):
            rx, ry, rz = L_euler[q]
            qml.RX(rx, wires=INIT_WIRES[q])
            qml.RY(ry, wires=INIT_WIRES[q])
            qml.RZ(rz, wires=INIT_WIRES[q])

        # Optional static entanglement among initials to distribute information
        self._entangle_initials()

        # 2) Coupling to PK qubits first (as required): controlled RY
        for i in range(3):
            for j in range(5):
                qml.CRY(W_pk[i, j], wires=[INIT_WIRES[i], PK_WIRES[j]])

        # Local rotations on PK qubits (group-specific processing)
        for j in range(5):
            qml.RY(pk_local[j], wires=PK_WIRES[j])

        # 3) Then coupling to PD qubits: controlled RY
        for i in range(3):
            for k in range(4):
                qml.CRY(W_pd[i, k], wires=[INIT_WIRES[i], PD_WIRES[k]])

        # Local rotations on PD qubits
        for k in range(4):
            qml.RY(pd_local[k], wires=PD_WIRES[k])

    def _circuit(self, inputs: pnp.ndarray, weights: dict[str, pnp.ndarray]):
        """QNode circuit: encodes inputs, applies L layers, measures 9 expvals.

        Parameters
        ----------
        inputs : array-like, shape (3,)
            Classical features to initialize the 3 initial qubits.
        weights : dict[str, array]
            Variational parameters with shapes given by weights_shapes().

        Returns
        -------
        list[pennylane.measure]
            9 expectation values (5 on PK wires, then 4 on PD wires), each in [-1, 1].
        """
        # Encode classical inputs on initial qubits
        self._encode_inputs(inputs)

        # Apply L variational layers
        for l in range(self.layers):
            self._layer(l, weights)

        # Measure expectation values corresponding to 9 outputs
        measurements = [qml.expval(qml.PauliZ(w)) for w in (PK_WIRES + PD_WIRES)]
        return measurements

    # -----------------------------------------
    # Public API
    # -----------------------------------------
    def qnode(self):
        """Return the compiled QNode for direct use (e.g., in optimization loops)."""
        return self._qnode

    def forward(self, inputs: pnp.ndarray, weights: dict[str, pnp.ndarray]) -> pnp.ndarray:
        """Evaluate the QNode and return a (9,) array of expectation values in [-1, 1]."""
        out = self._qnode(inputs, weights)
        return pnp.stack(out)

    @staticmethod
    def map_outputs_to_bounds(outputs: pnp.ndarray,
                              pk_bounds: pnp.ndarray = DEFAULT_PK_BOUNDS,
                              pd_bounds: pnp.ndarray = DEFAULT_PD_BOUNDS) -> pnp.ndarray:
        """Map 9 outputs in [-1, 1] to the provided (PK, PD) bounds.

        Parameters
        ----------
        outputs : array-like, shape (9,)
            Output expectation values from the circuit, ordered as 5 PK then 4 PD.
        pk_bounds : array-like, shape (5, 2)
            Min/max for PK parameters.
        pd_bounds : array-like, shape (4, 2)
            Min/max for PD parameters.

        Returns
        -------
        array, shape (9,)
            Values scaled into the requested bounds.
        """
        assert outputs.shape[0] == 9, "Expected 9 outputs (5 PK + 4 PD)"
        # Map [-1, 1] -> [0, 1]
        s = 0.5 * (outputs + 1.0)
        # Scale first 5 to PK bounds
        pk = pk_bounds[:, 0] + s[:5] * (pk_bounds[:, 1] - pk_bounds[:, 0])
        # Scale last 4 to PD bounds
        pd = pd_bounds[:, 0] + s[5:] * (pd_bounds[:, 1] - pd_bounds[:, 0])
        return pnp.concatenate([pk, pd])

    # -----------------------------------------
    # Drawing utilities
    # -----------------------------------------
    def draw(self, inputs: pnp.ndarray, weights: dict[str, pnp.ndarray], show: bool = True,
             filename: str | None = None):
        """Draw the circuit using PennyLane's matplotlib drawer.

        See: https://docs.pennylane.ai/en/stable/code/qml_drawer.html
        """
        drawer = qml.draw_mpl(self._qnode)
        fig, ax = drawer(inputs, weights)
        fig.suptitle("Variational PK/PD Circuit (qnc2)")
        if filename:
            fig.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax


# --------------------------------------------------------------------------------------
# Simple usage demo (safe to run):
# - Constructs the ansatz
# - Initializes weights
# - Prints parameter count
# - Evaluates once
# - Draws the circuit
# --------------------------------------------------------------------------------------

def _demo():
    ansatz = VariationalPKPDAnsatz(layers=2, device_shots=None)  # 2*45 = 90 params

    # Example classical inputs for the 3 initial qubits.
    # In a real integration, these could encode BW, COMED, dose_intensity or other features.
    inputs = pnp.array([0.1, -0.2, 0.3], dtype=float)

    # Initialize trainable parameters
    weights = ansatz.init_weights(seed=123)

    # Parameter counter
    n_params = ansatz.count_parameters(weights)
    print(f"Total trainable parameters: {n_params}")

    # Evaluate once
    raw_outputs = ansatz.forward(inputs, weights)
    print("Raw circuit outputs ([-1, 1]^9):", raw_outputs)

    # Optional: Map to PK/PD bounds
    mapped = ansatz.map_outputs_to_bounds(raw_outputs)
    print("Mapped outputs to PK/PD bounds:")
    print("  PK (Ka, CL, Vc, Q, Vp):   ", mapped[:5])
    print("  PD (Kin, Kout, Imax, IC50):", mapped[5:])

    # Draw the circuit with matplotlib (will display a window if running locally)
    try:
        ansatz.draw(inputs, weights, show=True)
    except Exception as e:
        # In headless environments, drawing may fail; don't crash the demo
        print(f"Drawing failed (likely no display). Error: {e}")


if __name__ == "__main__":
    _demo()
