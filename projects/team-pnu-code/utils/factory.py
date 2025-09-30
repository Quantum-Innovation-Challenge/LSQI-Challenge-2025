"""
Factory functions for building models and components
Separates model creation logic to avoid circular imports
"""
def create_model(config, loaders, pk_feats, pd_feats):
    """Create model based on configuration"""
    from models.unified_model import UnifiedPKPDModel
    model = UnifiedPKPDModel(
        config=config,
        pk_features=pk_feats,
        pd_features=pd_feats,   
        pk_input_dim=len(pk_feats),
        pd_input_dim=len(pd_feats)
    )
    return model

def create_trainer(config, model, loaders, device):
    """Create trainer based on configuration"""
    from training.unified_trainer import UnifiedPKPDTrainer
    trainer = UnifiedPKPDTrainer(
        model=model,
        config=config,
        data_loaders=loaders,
        device=device
    )
    return trainer

def build_encoder(encoder_type, input_dim, config):
    """Build encoder"""
    if encoder_type == "mlp":
        from models.encoders import MLPEncoder
        return MLPEncoder(input_dim, config.hidden, config.depth, config.dropout)

    elif encoder_type == "resmlp":
        from models.encoders import ResMLPEncoder
        return ResMLPEncoder(input_dim, config.hidden, config.depth, config.dropout)

    elif encoder_type == "moe":
        from models.encoders import MoEEncoder
        return MoEEncoder(
            in_dim=input_dim,
            hidden_dims=[config.hidden] * config.depth,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            dropout=config.dropout
        )

    elif encoder_type == "resmlp_moe":
        from models.encoders import ResMLPMoEEncoder
        encoder = ResMLPMoEEncoder(
            in_dim=input_dim,
            hidden_dim=config.hidden,
            num_layers=config.depth,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            mlp_ratio=getattr(config, 'mlp_ratio', 4.0),
            dropout=config.dropout,
            activation=getattr(config, 'activation', "gelu")
        )
        return encoder


    elif encoder_type == "cnn":
        from models.encoders import CNNEncoder
        return CNNEncoder(
            in_dim=input_dim,
            hidden=config.hidden,
            depth=config.depth,
            dropout=config.dropout,
            kernel_size=getattr(config, 'kernel_size', 3),
            num_filters=getattr(config, 'num_filters', 64)
        )

    elif encoder_type == "qmlp":
        from models.encoders import QMLPEncoder
        return QMLPEncoder(
            in_dim=input_dim,
            hidden=config.hidden,
            depth=config.depth,
            dropout=config.dropout,
            n_qubits=getattr(config, 'n_qubits', 4),
            n_layers=getattr(config, 'qnn_layers', 2)
        )

    elif encoder_type == "qresmlp":
        from models.encoders import QResMLPEncoder
        return QResMLPEncoder(
            in_dim=input_dim,
            hidden=config.hidden,
            num_layers=config.depth,
            dropout=config.dropout,
            n_qubits=getattr(config, 'n_qubits', 4),
            n_layers=getattr(config, 'qnn_layers', 2)
        )

    elif encoder_type == "qmoe":
        from models.encoders import QMoEEncoder
        return QMoEEncoder(
            in_dim=input_dim,
            hidden=[config.hidden] * config.depth,
            num_experts=getattr(config, 'num_experts', 4),
            top_k=getattr(config, 'top_k', 2),
            dropout=config.dropout,
            activation=getattr(config, 'activation', "relu"),
            n_qubits=getattr(config, 'n_qubits', 4),
            n_layers=getattr(config, 'qnn_layers', 1)
        )

    elif encoder_type == "qresmlp_moe":
        from models.encoders import QResMLPMoEEncoder
        return QResMLPMoEEncoder(
            in_dim=input_dim,
            hidden=config.hidden,
            num_layers=config.depth,
            activation=getattr(config, 'activation', "gelu"),
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            dropout=config.dropout,
            n_qubits=getattr(config, 'n_qubits', 4),
            n_layers=getattr(config, 'qnn_layers', 2)
        )

    elif encoder_type == "qnn":
        from models.encoders import QNNEncoder
        return QNNEncoder(
                    input_dim, config.hidden, config.depth, config.dropout,
                    n_qubits=getattr(config, 'n_qubits', 4),
                    n_layers=getattr(config, 'qnn_layers', 2)
            )
      
    elif encoder_type == 'resqnn_moe':
        from models.encoders import ResQNNMoEEncoder
        return ResQNNMoEEncoder(
            in_dim=input_dim,
            hidden_dim=config.hidden,
            num_layers=config.depth,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            n_qubits=getattr(config, 'n_qubits', 4),
            n_layers=getattr(config, 'qnn_layers', 2)
        )
    else:
        # Fallback to MLP for unknown encoders
        from models.encoders import MLPEncoder
        return MLPEncoder(input_dim, config.hidden, config.depth, config.dropout)

def build_head(head_type, hidden_dim, config=None):
    """Build head"""
    if head_type == "mse":
        from models.heads import MSEHead
        dropout_rate = config.mc_dropout_rate if config and config.use_mc_dropout else 0.0
        return MSEHead(hidden_dim, dropout_rate)
    elif head_type == "emax_gaussian":
        from models.heads import EmaxGaussianHead
        return EmaxGaussianHead(hidden_dim)
    elif head_type == "emax":
        from models.heads import EmaxHead
        return EmaxHead(hidden_dim)

    elif head_type == "binary_classification":
        from models.heads import BinaryClassificationHead
        return BinaryClassificationHead(hidden_dim)
    else:
        from models.heads import MSEHead
        dropout_rate = config.mc_dropout_rate if config and config.use_mc_dropout else 0.0
        return MSEHead(hidden_dim, dropout_rate)