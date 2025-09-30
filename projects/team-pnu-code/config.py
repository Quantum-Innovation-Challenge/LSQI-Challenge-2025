"""
    Configuration
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Configuration"""
    
    # === Basic settings ===
    mode: str = "independent"  # independent (separate), cascade (dual_stage), multi-task (shared)
    csv_path: str = "data/EstData.csv"
    
    # === Training settings ===
    epochs: int = 1000  # Optimized for TIME feature enhancement model
    batch_size: int = 16  # Increased for better GPU utilization and stability
    learning_rate: float = 1e-3  # Optimized for TIME feature enhancement convergence
    patience: int = 100  # Balanced early stopping for TIME feature model
    loss_type_pd: str = "mse"        # ["mse", "mae", "asymmetric", "quantile", "one_sided"]
    loss_type_pk: str = "mse"        # PK is generally mse/mae only
    quantile_q: float = 0.3            # quantile loss parameter
    over_weight: float = 2.0           # asymmetric loss parameter
    under_weight: float = 0.5
    hybrid_lambda: float = 0.3
	
    # === Model settings ===
    encoder: str = "mlp"  # "mlp", "resmlp", "moe", "resmlp_moe", qnn', "resqnn_moe"
    encoder_pk: Optional[str] = None  # PK-specific encoder
    encoder_pd: Optional[str] = None  # PD-specific encoder
    head_pk: str = "mse"  # "mse", "gauss", "poisson"
    head_pd: str = "mse"  # "mse", "gauss", "poisson", "emax", "emax_gaussian"
    
    # === Model hyperparameters ===
    hidden: int = 128  # Increased for TIME feature enhancement model capacity
    depth: int = 4  # Increased depth for better TIME pattern learning
    dropout: float = 0.3  # Increased for better regularization with TIME features
    
    # === MoE settings ===
    num_experts: int = 8  # Increased for better expert diversity
    top_k: int = 4  # Optimal for most cases
    
    # === CNN settings ===
    kernel_size: int = 3  # Good for time series patterns
    num_filters: int = 128  # Increased for better feature extraction
    
    # === QNN settings ===
    n_qubits: int = 6  # Increased for better quantum capacity
    qnn_layers: int = 3  # Increased for more complex quantum circuits
    quantum_ratio: float = 0.3  # Reduced for more stable hybrid performance
    use_entanglement: str = "linear"  # "linear", "circular", "all"
    use_data_reuploading: bool = True
    quantum_frequency: int = 2  # For quantum_resmlp
    
    # === Uncertainty Quantification ===
    use_mc_dropout: bool = False  # Disabled by default for faster training
    mc_dropout_rate: float = 0.2  # Matches main dropout rate
    mc_samples: int = 30  # Increased for better uncertainty estimation
    
    # === Data preprocessing ===
    use_fe: bool = True  # Enabled by default for better performance
    use_perkg: bool = False
    allow_future_dose: bool = False
    time_windows: List[int] = None
    
    # === Data augmentation ===
    aug_method: str = 'pd_response'  # "mixup", "jitter", "jitter_mixup", "gaussian_noise", "scaling", "time_warp", "feature_dropout", "cutmix", "random_erase", "label_smooth", "amplitude_scale", "enhanced_mixup", "random", "pk_curve", "pd_response", or None
    aug_ratio: float = 0.2  # Ratio of augmented samples to original data (e.g., 0.2 for 20%)
    aug_samples: int = 100  # Number of augmentation samples (used if aug_ratio is None)
    mixup_alpha: float = 0.3  # Mixup alpha parameter
    jitter_std: float = 0.05  # Standard deviation for DV jitter
    time_shift_ratio: float = 0.1  # Ratio for TIME shift
    
    # Enhanced augmentation parameters
    gaussian_noise_std: float = 0.02  # Standard deviation for Gaussian noise
    scale_range: tuple = (0.8, 1.2)  # Range for random scaling
    dropout_rate: float = 0.1  # Dropout rate for augmentation
    time_warp_factor: float = 0.1  # Factor for time warping
    amplitude_scale_range: tuple = (0.9, 1.1)  # Range for amplitude scaling
    cutmix_alpha: float = 1.0  # Alpha parameter for CutMix
    cutmix_prob: float = 0.1  # Probability for CutMix
    label_smooth_eps: float = 0.1  # Epsilon for label smoothing
    random_erase_prob: float = 0.1  # Probability for random erasing
    feature_dropout_prob: float = 0.1  # Probability for feature dropout

    # Supervised training augmentation
    use_aug_supervised: bool = False  # Use augmentation during supervised training
    aug_lambda: float = 0.15  # Weight for augmented loss (original + lambda * augmented)


    # === SimCLR Contrastive learning ===
    temperature: float = 0.1  # Temperature for SimCLR contrastive learning
    time_jitter_std: float = 0.1  # Time jitter standard deviation
    noise_std: float = 0.05  # Gaussian noise standard deviation
    contrastive_scale_range: tuple = (0.8, 1.2)  # Scaling range for contrastive augmentation
    pretraining_epochs: int = 50  # Number of pretraining epochs
    pretraining_patience: int = 20  # Patience for contrastive pretraining
    use_pt_contrast: bool = False  # Enable contrastive pretraining (set by --use_pt_contrast flag)
    use_pt_clf: bool = False  # Enable PD pretraining with classification task (set by --use_pt_clf flag)
    


    use_clf: bool = False  # Enable PD to classification task (set by --use_clf flag)
    threshold: float = 3.3  # Threshold for classification task
    clf_finetune: bool = False  # Enable PD to classification task (set by --clf_finetune flag)
    clf_lr: float = 1e-3  # Learning rate for classification head
    clf_patience: int = 100  # Patience for classification head
    clf_epochs: int = 1000  # Number of epochs for classification head


    # === Data splitting ===
    split_strategy: str = "stratify_dose_even"  # Best strategy for PK/PD data
    test_size: float = 0.1  # Increased for better test evaluation
    val_size: float = 0.1  # Increased for better validation
    random_state: int = 42  # Reproducible results
    
    # === Output settings ===
    output_dir: str = "results"
    run_name: Optional[str] = None
    verbose: bool = False
    device_id: int = 0
    
    def __post_init__(self):
        """Initialization after processing"""
        if self.time_windows is None:
            self.time_windows = [24, 48, 72, 96, 120, 144, 168]




def parse_time_windows(time_windows_str: str) -> List[int]:
    """Parse time windows string"""
    if not time_windows_str:
        return None
    try:
        return [int(x.strip()) for x in time_windows_str.split(',') if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid time windows format: {time_windows_str}. Use comma-separated integers (e.g., '24,48,72,96,120,144,168')")


def parse_args() -> Config:
    """Parse command line arguments and create Config object"""
    parser = create_argument_parser()
    args = parser.parse_args()
    config = Config(
        # Basic settings
        mode=args.mode,
        csv_path=args.csv,
        
        loss_type_pd=args.loss_type_pd,       # ["mse", "mae", "asymmetric", "quantile", "one_sided"]
        loss_type_pk=args.loss_type_pk,        # PK is generally mse/mae only
        quantile_q=args.quantile_q,             # quantile loss parameter
        hybrid_lambda=args.hybrid_lambda,       # MSE : Quantile weight ratio (0.7 → MSE 70%, Quantile 30%)
        over_weight=args.over_weight,            # asymmetric loss parameter
        under_weight=args.under_weight, 
        
        # Training settings
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,


		# Model settings
        encoder=args.encoder,
        encoder_pk=args.encoder_pk,
        encoder_pd=args.encoder_pd,
        head_pk=args.head_pk,
        head_pd=args.head_pd,
        
        # Model hyperparameters
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        
        # MoE settings
        num_experts=args.num_experts,
        top_k=args.top_k,
        
        # CNN settings
        kernel_size=args.kernel_size,
        num_filters=args.num_filters,
        
        # QNN settings
        n_qubits=args.n_qubits,
        qnn_layers=args.qnn_layers,
        quantum_ratio=args.quantum_ratio,
        use_entanglement=args.use_entanglement,
        use_data_reuploading=args.use_data_reuploading,
        quantum_frequency=args.quantum_frequency,
        
        # Uncertainty Quantification
        use_mc_dropout=args.use_mc_dropout,
        mc_dropout_rate=args.mc_dropout_rate,
        mc_samples=args.mc_samples,
        
        # Data preprocessing
        use_fe=args.use_fe,
        use_perkg=args.use_perkg,
        allow_future_dose=args.allow_future_dose,
        time_windows=parse_time_windows(args.time_windows) if args.time_windows else None,
        
        # Data augmentation
        aug_method=args.aug_method,
        aug_ratio=args.aug_ratio,
        aug_samples=args.aug_samples,
        mixup_alpha=args.mixup_alpha,
        jitter_std=args.jitter_std,
        time_shift_ratio=args.time_shift_ratio,

        # Enhanced augmentation parameters
        gaussian_noise_std=args.gaussian_noise_std,
        contrastive_scale_range=tuple(args.contrastive_scale_range),
        dropout_rate=args.dropout_rate,
        time_warp_factor=args.time_warp_factor,
        amplitude_scale_range=tuple(args.amplitude_scale_range),
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        label_smooth_eps=args.label_smooth_eps,
        random_erase_prob=args.random_erase_prob,
        feature_dropout_prob=args.feature_dropout_prob,

        # Supervised training augmentation
        use_aug_supervised=args.use_aug_supervised,
        aug_lambda=args.aug_lambda,

        temperature=args.temperature,
        time_jitter_std=args.time_jitter_std,   
        noise_std=args.noise_std,
        pretraining_epochs=args.pretraining_epochs,
        pretraining_patience=args.pretraining_patience,
        use_pt_contrast=args.use_pt_contrast,
        use_pt_clf=args.use_pt_clf,
        use_clf=args.use_clf,
        threshold=args.threshold,
        clf_finetune=args.clf_finetune,
        clf_lr=args.clf_lr,
        clf_patience=args.clf_patience,
        clf_epochs=args.clf_epochs,

        # Data splitting
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        
        # Output settings
        output_dir=args.out_dir,
        run_name=args.run_name,
        verbose=args.verbose,
        device_id=args.device_id
    )
    
    return config

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="PK/PD Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # === Basic settings ===
    parser.add_argument("--mode", 
                       choices=["independent", "cascade", "multi-task", "separate", "dual_stage", "shared"], # independent (separate), cascade (dual_stage), multi-task (shared)
                       default="cascade", help="Training mode")
    parser.add_argument("--csv", default="data/EstData.csv", help="Data CSV file path")
    # EstData_combined or EstData_weekly
    
    # === Training settings ===
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    
    parser.add_argument("--pretraining_epochs", type=int, default=3000, help="Number of contrastive pretraining epochs")
    parser.add_argument("--pretraining_patience", type=int, default=100, help="Patience for contrastive pretraining")
    
    parser.add_argument("--clf_epochs", type=int, default=500, help="Number of epochs for classification head")
    parser.add_argument("--clf_patience", type=int, default=50, help="Patience for classification head")
    parser.add_argument("--clf_finetune", action="store_true", help="Finetune classification head")
    parser.add_argument("--clf_lr", type=float, default=1e-4, help="Learning rate for classification head")

    parser.add_argument("--loss_type_pd", choices=["mse", "mae", "asymmetric", "quantile", "one_sided", "hybrid"], default="mse", help="PD loss type")
    parser.add_argument("--loss_type_pk", choices=["mse", "mae", "asymmetric", "quantile", "one_sided", "hybrid"], default="mse", help="PK loss type")
    parser.add_argument("--quantile_q", type=float, default=0.3, help="Quantile value for quantile/hybrid loss")
    parser.add_argument("--hybrid_lambda", type=float, default=0.7, help="Weight for hybrid MSE loss")
    parser.add_argument("--over_weight", type=float, default=2.0, help="Weight for over-prediction penalty (asymmetric)")
    parser.add_argument("--under_weight", type=float, default=0.5, help="Weight for under-prediction penalty (asymmetric)")


    # === Model settings ===
    parser.add_argument("--encoder", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "cnn", "qmlp", "qresmlp", "qmoe", "qresmlp_moe", "qnn", "resqnn_moe"],
                       default="mlp", help="Default encoder type")
    parser.add_argument("--encoder_pk", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "cnn", "qmlp", "qresmlp", "qmoe", "qresmlp_moe", "qnn", "resqnn_moe"], 
                       default=None, help="PK-specific encoder type")
    parser.add_argument("--encoder_pd", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "cnn", "qmlp", "qresmlp", "qmoe", "qresmlp_moe", "qnn", "resqnn_moe"], 
                       default=None, help="PD-specific encoder type")

    parser.add_argument("--head_pk", choices=["mse", "gauss", "poisson"], 
                       default="mse", help="PK head type")
    parser.add_argument("--head_pd", choices=["mse", "gauss", "poisson", "emax", "emax_gaussian"], 
                       default="mse", help="PD head type")
    
    # === Model hyperparameters ===
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=2, help="Network depth")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout ratio")
    
    # === MoE settings ===
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top experts to use")
    
    # === CNN settings ===
    parser.add_argument("--kernel_size", type=int, default=3, help="CNN kernel size")
    parser.add_argument("--num_filters", type=int, default=128, help="Number of CNN filters")
    
    # === QNN settings ===
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits for QNN")
    parser.add_argument("--qnn_layers", type=int, default=2, help="Number of quantum layers")
    parser.add_argument("--quantum_ratio", type=float, default=0.3, help="Quantum ratio for hybrid QNN")
    parser.add_argument("--use_entanglement", choices=["linear", "circular", "all"], 
                       default="linear", help="Entanglement pattern for QNN")
    parser.add_argument("--use_data_reuploading", action="store_true", help="Use data reuploading in QNN")
    parser.add_argument("--quantum_frequency", type=int, default=2, help="Quantum frequency for quantum_resmlp")
    
    # === Data preprocessing ===
    parser.add_argument("--use_fe", action="store_true", help="Feature engineering")
    parser.add_argument("--use_perkg", action="store_true", help="Per kg dose")
    parser.add_argument("--allow_future_dose", action="store_true", help="Allow future dose information")
    parser.add_argument("--time_windows", type=str, default=None,
                       help="Time windows (comma-separated, e.g., '24,48,72,96,120,144,168')")
    
    # === Data augmentation ===
    parser.add_argument("--aug_method", choices=[
        "mixup", "jitter", "jitter_mixup", "gaussian_noise", "scaling",
        "time_warp", "feature_dropout", "cutmix", "random_erase",
        "label_smooth", "amplitude_scale", "enhanced_mixup",
        "random", "pk_curve", "pd_response"
    ], help="Augmentation method")
    parser.add_argument("--aug_ratio", type=float, default=None,
                       help="Ratio of augmented samples to original data (e.g., 0.2 for 20%)")
    parser.add_argument("--aug_samples", type=int, default=100,
                       help="Number of augmentation samples (used if aug_ratio is None)")
    parser.add_argument("--mixup_alpha", type=float, default=0.3,
                       help="Mixup alpha parameter (default: 0.3)")
    parser.add_argument("--jitter_std", type=float, default=0.05,
                       help="Standard deviation for DV jitter (default: 0.05)")
    parser.add_argument("--time_shift_ratio", type=float, default=0.1,
                       help="Ratio for TIME shift (default: 0.1)")

    # Enhanced augmentation parameters
    parser.add_argument("--gaussian_noise_std", type=float, default=0.02,
                       help="Standard deviation for Gaussian noise (default: 0.02)")
    parser.add_argument("--scale_range", nargs=2, type=float, default=[0.8, 1.2],
                       help="Range for random scaling (default: [0.8, 1.2])")
    parser.add_argument("--time_warp_factor", type=float, default=0.1,
                       help="Factor for time warping (default: 0.1)")
    parser.add_argument("--amplitude_scale_range", nargs=2, type=float, default=[0.9, 1.1],
                       help="Range for amplitude scaling (default: [0.9, 1.1])")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                       help="Alpha parameter for CutMix (default: 1.0)")
    parser.add_argument("--cutmix_prob", type=float, default=0.1,
                       help="Probability for CutMix (default: 0.1)")
    parser.add_argument("--label_smooth_eps", type=float, default=0.1,
                       help="Epsilon for label smoothing (default: 0.1)")
    parser.add_argument("--random_erase_prob", type=float, default=0.1,
                       help="Probability for random erasing (default: 0.1)")
    parser.add_argument("--feature_dropout_prob", type=float, default=0.1,
                       help="Probability for feature dropout (default: 0.1)")
                       
    parser.add_argument("--use_aug_supervised", action="store_true",
                       help="Use augmentation during supervised training")
    parser.add_argument("--aug_lambda", type=float, default=0.3,
                       help="Weight for augmented loss (original + lambda * augmented)")

    # === SimCLR Contrastive learning ===
    parser.add_argument("--temperature", type=float, default=0.1, help="SimCLR temperature")
    parser.add_argument("--time_jitter_std", type=float, default=0.1, help="Time jitter standard deviation")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Gaussian noise standard deviation")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for augmentation")
    parser.add_argument("--contrastive_scale_range", nargs=2, type=float, default=[0.8, 1.2], help="Scaling range for contrastive augmentation")


    parser.add_argument("--use_pt_contrast", action="store_true", help="Enable contrastive pretraining")
    parser.add_argument("--use_pt_clf", action="store_true", help="Enable PD pretraining with classification task")

    parser.add_argument("--use_clf", action="store_true", help="PD to classification task") 
    parser.add_argument("--threshold", type=float, default=3.3, help="Threshold for classification task")


    # === Data splitting ===
    parser.add_argument("--split_strategy", 
                       choices=["stratify_dose_even", "stratify_dose_even_no_placebo_test", "leave_one_dose_out", "random_subject", "only_bw_range", "highest_bw_one_test", "stratify_dose_even_no_placebo_valtest"],
                       default="stratify_dose_even_no_placebo_valtest",
                       help="Data splitting strategy")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # === Uncertainty Quantification ===
    parser.add_argument("--use_mc_dropout", action="store_true", help="Use Monte Carlo Dropout for uncertainty quantification")
    parser.add_argument("--mc_dropout_rate", type=float, default=0.3, help="Dropout rate for MC Dropout")
    parser.add_argument("--mc_samples", type=int, default=30, help="Number of MC samples for uncertainty estimation")
    
    # === Output settings ===
    parser.add_argument("--out_dir", default="results", help="Result output directory")
    parser.add_argument("--run_name", help="Run name (auto-generated)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    
    return parser
