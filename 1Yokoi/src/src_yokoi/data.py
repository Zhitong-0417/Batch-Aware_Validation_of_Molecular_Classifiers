"""SSV workflow utilities."""



from dataclasses import dataclass
from .common import *



class MolecularDataset:
    """Expression data with rows as features and columns as samples."""

    expression: pd.DataFrame
    class_labels: pd.Series
    batch_labels: pd.Series

    def __post_init__(self):
        if not isinstance(self.expression, pd.DataFrame):
            raise TypeError("expression must be a pandas DataFrame")
        if not isinstance(self.class_labels, pd.Series):
            raise TypeError("class_labels must be a pandas Series")
        if not isinstance(self.batch_labels, pd.Series):
            raise TypeError("batch_labels must be a pandas Series")
        if self.expression.empty:
            raise ValueError("expression must be non-empty")
        if not self.expression.columns.is_unique:
            raise ValueError("expression column sample IDs must be unique")
        if not self.class_labels.index.is_unique:
            raise ValueError("class label sample IDs must be unique")
        if not self.batch_labels.index.is_unique:
            raise ValueError("batch label sample IDs must be unique")
        if not self.expression.columns.equals(self.class_labels.index):
            raise ValueError("expression columns must exactly match class_labels.index")
        if not self.expression.columns.equals(self.batch_labels.index):
            raise ValueError("expression columns must exactly match batch_labels.index")

        self.expression = self.expression.copy()
        self.class_labels = self.class_labels.copy()
        self.batch_labels = self.batch_labels.copy()


def _ordered_members(sample_ids, selected):
    selected = set(selected)
    return [sample_id for sample_id in sample_ids if sample_id in selected]


def _ordered_unique(values):
    return list(pd.unique(pd.Series(values)))


def validate_molecular_dataset(dataset, *, require_binary=True, require_batches=True, min_batches=2):
    """Validate generic SSV molecular dataset invariants."""
    if not isinstance(dataset, MolecularDataset):
        raise TypeError("dataset must be a MolecularDataset")
    if not isinstance(dataset.expression, pd.DataFrame):
        raise TypeError("dataset.expression must be a pandas DataFrame")
    if not isinstance(dataset.class_labels, pd.Series):
        raise TypeError("dataset.class_labels must be a pandas Series")
    if not isinstance(dataset.batch_labels, pd.Series):
        raise TypeError("dataset.batch_labels must be a pandas Series")
    if dataset.expression.empty:
        raise ValueError("expression must be non-empty")
    if dataset.expression.shape[1] == 0:
        raise ValueError("expression must contain samples")
    if not dataset.expression.columns.is_unique:
        raise ValueError("expression sample IDs must be unique")
    if not dataset.expression.index.is_unique:
        raise ValueError("expression feature IDs must be unique")
    if not dataset.class_labels.index.is_unique:
        raise ValueError("class label sample IDs must be unique")
    if not dataset.batch_labels.index.is_unique:
        raise ValueError("batch label sample IDs must be unique")
    if not dataset.expression.columns.equals(dataset.class_labels.index):
        raise ValueError("expression columns must exactly match class_labels.index")
    if not dataset.expression.columns.equals(dataset.batch_labels.index):
        raise ValueError("expression columns must exactly match batch_labels.index")
    if dataset.class_labels.isna().any():
        raise ValueError("class labels must not contain missing values")
    if dataset.batch_labels.isna().any():
        raise ValueError("batch labels must not contain missing values")
    if require_binary and dataset.class_labels.nunique(dropna=False) != 2:
        raise ValueError("binary SSV workflow requires exactly two classes")
    if require_batches and dataset.batch_labels.nunique(dropna=False) < min_batches:
        raise ValueError(f"SSV workflow requires at least {min_batches} batch/group labels")
    return dataset
