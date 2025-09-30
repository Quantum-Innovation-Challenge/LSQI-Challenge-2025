import pennylane as qml
from pennylane import numpy as np


import torch
import torch.nn as nn


def qnn_circuit(inputs, weights):
    """Reusable quantum circuit definition"""
    n_layers, n_qubits, _ = weights.shape # strongly only
    for l in range(n_layers):
        # Data re-uploading with multi-axis embeddings
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

        # Entangling block
            # qml.BasicEntanglerLayers(weights[l:l+1], wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))

    # Measurement: single-qubit
    obs = []
    obs += [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    obs += [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    obs += [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)) for i in range(n_qubits)]

    return obs


class qnn_basic(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=8, n_layers=2,
                 dev_name="default.qubit", readout_axes="XYZ",
                 use_basic=False, use_corr=True):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.readout_axes = readout_axes
        self.use_basic = use_basic
        self.use_corr = use_corr

        self.input = nn.Linear(input_dim, n_qubits)

        readout_dim = n_qubits * len(readout_axes)
        readout_dim += n_qubits  # extra for correlation terms
        self.out = nn.Linear(readout_dim, output_dim)

        self.dev = qml.device(dev_name, wires=n_qubits)

        qnode = qml.QNode(qnn_circuit,
            self.dev,
            interface="torch"
        )
        
        weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}

        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.skip_proj = nn.Sequential(
            nn.LayerNorm(n_qubits),
            nn.GELU(),
            nn.Linear(n_qubits, readout_dim)
        )
        self.skip_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.input(x)
        q_feats = self.q_layer(x).to(x.dtype)
        q_feats = q_feats + self.skip_alpha * self.skip_proj(x)
        return self.out(q_feats)


class cur_best(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=10, n_layers=3, dev_name="default.qubit"):
        super().__init__()
        
        self.input = nn.Linear(input_dim, n_qubits) # For angle
        self.out = nn.Linear(n_qubits, output_dim) 

        self.n_qubits = n_qubits
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.dev = qml.device(dev_name, wires=n_qubits)

        # define quantum circuit
        def circuit(inputs, weights):
            # Ampl Encodding
            # qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            # for l in range(n_qubits):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
            
            ## Try data-reuploading???
            
            # Basic entanglement
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # QNN layer
        qnode = qml.QNode(circuit, self.dev, interface="torch")
        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        }
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        x = self.input(x)
        x = self.q_layer(x)
        return self.out(x)