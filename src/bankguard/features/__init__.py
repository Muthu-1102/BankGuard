"""BankGuard features package."""

from bankguard.features.feature_definition import (
    DEFAULT_EXCLUDED_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TargetLeakageError,
    audit_feature_leakage,
    generate_feature_inventory_report,
    get_feature_names,
    identify_feature_types,
    load_feature_config,
    split_features_and_target,
)
from bankguard.features.leakage_audit import (
    EXCLUDED_FEATURES_REGISTRY,
    audit_dataset_leakage,
    audit_feature_matrix_leakage,
    generate_leakage_audit_report,
    run_leakage_audit_pipeline,
)
from bankguard.features.preprocessor import (
    BankGuardFeatureEngineer,
    BankGuardPreprocessor,
    create_preprocessing_pipeline,
    run_preprocessing_pipeline_report,
    temporal_train_val_split,
)

__all__ = [
    "TARGET_COLUMN",
    "ID_COLUMN",
    "DEFAULT_EXCLUDED_COLUMNS",
    "TargetLeakageError",
    "load_feature_config",
    "get_feature_names",
    "split_features_and_target",
    "identify_feature_types",
    "audit_feature_leakage",
    "generate_feature_inventory_report",
    "temporal_train_val_split",
    "BankGuardFeatureEngineer",
    "BankGuardPreprocessor",
    "create_preprocessing_pipeline",
    "run_preprocessing_pipeline_report",
    "EXCLUDED_FEATURES_REGISTRY",
    "audit_dataset_leakage",
    "audit_feature_matrix_leakage",
    "generate_leakage_audit_report",
    "run_leakage_audit_pipeline",
]
