import torch
import torch.nn as nn
import pennylane as qml
import math

class QuantumEntanglingLinear_new(nn.Module):
    def __init__(self, dim, num_layers=2):
        """
        Quantum-inspired layer:
        - Single-qubit rotations (trainable, 3 parameters per qubit)
        - Two-qubit cyclic CNOTs (fixed)
        Precompute matrices for efficiency.
        """
        super().__init__()
        assert math.log2(dim).is_integer(), "dim must be a power of 2"
        self.dim = dim
        self.n_qubits = int(math.log2(dim))
        self.num_layers = num_layers

        # Trainable single-qubit rotation angles: (num_layers, n_qubits, 3)
        self.local_angles = nn.Parameter(torch.randn(num_layers, self.n_qubits, 3))

        # Precompute fixed cyclic CNOT layer
        self.register_buffer("cnot_matrix", self.build_cyclic_cnot())

    def kron_n(self, matrices):
        out = matrices[0]
        for m in matrices[1:]:
            out = torch.kron(out, m)
        return out

    def single_qubit_gate(self, theta):
        """3-parameter single qubit gate Rx Ry Rz"""
        theta_x, theta_y, theta_z = theta
        Rx = torch.tensor([[torch.cos(theta_x/2), -torch.sin(theta_x/2)],
                           [torch.sin(theta_x/2), torch.cos(theta_x/2)]], dtype=torch.float32)
        Ry = torch.tensor([[torch.cos(theta_y/2), -torch.sin(theta_y/2)],
                           [torch.sin(theta_y/2), torch.cos(theta_y/2)]], dtype=torch.float32)
        Rz = torch.tensor([[torch.cos(theta_z/2), -torch.sin(theta_z/2)],
                           [torch.sin(theta_z/2), torch.cos(theta_z/2)]], dtype=torch.float32)
        return Rz @ Ry @ Rx

    def one_qubit_layer_matrix(self, layer_idx):
        """Compute full single-qubit layer matrix by tensoring all qubits"""
        matrices = [self.single_qubit_gate(self.local_angles[layer_idx, q]) for q in range(self.n_qubits)]
        return self.kron_n(matrices)

    def build_cyclic_cnot(self):
        """Precompute full cyclic CNOT layer matrix"""
        n = self.n_qubits
        dim = 2**n
        I = torch.eye(dim, dtype=torch.float32)
        cnot = I.clone()

        # Apply cyclic CNOTs (0->1, 1->2, ..., n-2->n-1)
        for control in range(n-1):
            target = control + 1
            for i in range(dim):
                if (i >> control) & 1:
                    j = i ^ (1 << target)
                    cnot[[i,j], :] = cnot[[j,i], :]

        # Wrap-around CNOT (n-1 -> 0)
        control = n-1
        target = 0
        for i in range(dim):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                cnot[[i,j], :] = cnot[[j,i], :]
        return cnot

    def forward(self, x):
        out = x
        for l in range(self.num_layers):
            # Compute single-qubit layer matrix
            single_matrix = self.one_qubit_layer_matrix(l).to(x.device)
            out = out @ single_matrix.T

            # Multiply by fixed cyclic CNOT matrix
            out = out @ self.cnot_matrix.T

        return out
class QuantumLikeLinear(nn.Module):
    def __init__(self, dim, num_rotations=None):
        """
        Args:
            dim (int): input/output dimension (like w in Linear(w, w))
            num_rotations (int): how many 2D rotations to stack
                                 if None, defaults to dim (one per axis pair)
        """
        super().__init__()
        self.dim = dim
        self.num_rotations = num_rotations or dim
        # trainable rotation angles
        self.angles = nn.Parameter(torch.randn(self.num_rotations))

        # define which indices each rotation acts on
        # here: nearest-neighbor pairs (0,1), (1,2), ..., (dim-2, dim-1)
        self.pairs = [(i, (i + 1) % dim) for i in range(self.num_rotations)]

    def forward(self, x):
        # x shape: (batch, dim)
        out = x
        for theta, (i, j) in zip(self.angles, self.pairs):
            c, s = torch.cos(theta), torch.sin(theta)

            # rotation matrix for indices (i,j)
            R = torch.eye(self.dim, device=x.device)
            R[i, i], R[j, j] = c, c
            R[i, j], R[j, i] = -s, s

            out = out @ R.T  # apply rotation
        return out
    
class QuantumEntanglingLinear(nn.Module):
    def __init__(self, dim, num_layers=2):
        """
        Quantum-inspired layer with entangling rotations.
        Args:
            dim (int): dimension (should correspond to number of "qubits")
            num_layers (int): how many layers of rotations + entanglers
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # trainable parameters
        self.local_angles = nn.Parameter(torch.randn(num_layers, dim))   # single rotations
        self.ent_angles = nn.Parameter(torch.randn(num_layers, dim//2))  # entanglers

    def forward(self, x):
        out = x
        for l in range(self.num_layers):
            # --- local rotations (like RY) ---
            for i, theta in enumerate(self.local_angles[l]):
                c, s = torch.cos(theta), torch.sin(theta)
                R = torch.eye(self.dim, device=x.device)
                R[i, i] = c
                R[i, (i+1) % self.dim] = -s
                R[(i+1) % self.dim, i] = s
                R[(i+1) % self.dim, (i+1) % self.dim] = c
                out = out @ R.T

            # --- entangling rotations (like exp(-i θ XX)) ---
            for k, theta in enumerate(self.ent_angles[l]):
                i, j = 2*k, 2*k+1
                # XX entangler matrix (simplified 2x2 rotation block across i,j)
                c, s = torch.cos(theta), torch.sin(theta)
                R = torch.eye(self.dim, device=x.device)
                R[i, i] = c; R[j, j] = c
                R[i, j] = -s; R[j, i] = s
                out = out @ R.T
        return out
    

def _apply_pairwise_rotations_(out, idx_i, idx_j, theta):
    """
    Vectorized right-multiplication by a block-diagonal R^T consisting of 2x2 blocks:
    R_block = [[c,-s],[s,c]]  ⇒ R^T_block = [[c, s],[-s, c]]
    out: (..., D)
    idx_i, idx_j: shape (K,)
    theta: shape (K,)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    col_i = out.index_select(-1, idx_i)            # (..., K)
    col_j = out.index_select(-1, idx_j)            # (..., K)

    new_i = c * col_i + s * col_j
    new_j = -s * col_i + c * col_j

    out.index_copy_(-1, idx_i, new_i)
    out.index_copy_(-1, idx_j, new_j)

class QuantumEntanglingLinearVectorized(nn.Module):
    """
    Loop-minimized version using checkerboard (even/odd) sweeps for local rotations
    + fully vectorized entanglers. One small Python loop remains over layers.
    """
    def __init__(self, dim, num_layers=2):
        super().__init__()
        assert dim >= 2
        self.dim = dim
        self.num_layers = num_layers

        # trainable parameters
        self.local_angles = nn.Parameter(torch.randn(num_layers, dim))      # θ for each site
        self.ent_angles   = nn.Parameter(torch.randn(num_layers, dim // 2)) # θ for pairs (0,1),(2,3),...

        # precompute index patterns on registration buffer (moved to device with module)
        D = dim
        even_i = torch.arange(0, D - (D % 2), 2)      # 0,2,4,...
        even_j = even_i + 1                           # 1,3,5,...

        odd_i  = torch.arange(1, D, 2)                # 1,3,5,...
        odd_j  = (odd_i + 1) % D                      # 2,4,6,...,0 (wrap)

        self.register_buffer("even_i", even_i, persistent=False)
        self.register_buffer("even_j", even_j, persistent=False)
        self.register_buffer("odd_i",  odd_i,  persistent=False)
        self.register_buffer("odd_j",  odd_j,  persistent=False)

    def forward(self, x):
        """
        x: (..., D)
        """
        out = x

        for l in range(self.num_layers):
            # ---- local rotations (vectorized checkerboard) ----
            theta_l = self.local_angles[l]  # (D,)

            if self.even_i.numel() > 0:
                _apply_pairwise_rotations_(out, self.even_i, self.even_j, theta_l.index_select(0, self.even_i))

            if self.odd_i.numel() > 0:
                _apply_pairwise_rotations_(out, self.odd_i,  self.odd_j, theta_l.index_select(0, self.odd_i))

            # ---- entanglers (disjoint pairs; fully vectorized) ----
            if self.even_i.numel() > 0:
                _apply_pairwise_rotations_(out, self.even_i, self.even_j, self.ent_angles[l])

        return out

class VQC_class(torch.nn.Module):
    def __init__(self, input_size, output_size, n_layers=1, n_qubits=4) -> None:
        super().__init__()
        self.sim_dev = qml.device("default.qubit", wires=n_qubits)
    #         self.sim_dev = qml.device("lightning.qubit", wires=n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_size = output_size
        self.activation = nn.Tanh()
    #         self.dropout = nn.Dropout(p=0.2)

    #         self.clayer_1 = torch.nn.Linear(input_size, n_qubits)
        self.clayer_2 = torch.nn.Linear(n_qubits, output_size)
        weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    #         weights_shape = (n_layers, n_qubits)
        self.weights = nn.Parameter(torch.randn(weights_shape))
    def QNode(self, inputs, weights):
        @qml.qnode(self.sim_dev, interface="torch", diff_method="backprop")# diff_method="backprop")
        def qnode(inputs, weights):
    #             qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), pad_with = 0.)
    #             qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = inputs.shape[:-1]
            inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        weights.to(inputs.device)
        res = qnode(inputs, weights)
        if isinstance(res, torch.Tensor):
            results = res.type(inputs.dtype)

        else:
            if len(inputs.shape) > 1:
                res = [torch.reshape(r, (inputs.shape[0], -1)) for r in res]
            results = torch.hstack(res).type(inputs.dtype)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results

    def forward(self, X):
    #         X = self.clayer_1(X)
    #         X = self.activation(X)
        X = self.QNode(X, self.weights)
        X = self.clayer_2(X)
        return X

    def _freeze_Qnode(self):
        self.weights.requires_grad = False

    def _unfreeze_Qnode(self):
        self.weights.requires_grad = True