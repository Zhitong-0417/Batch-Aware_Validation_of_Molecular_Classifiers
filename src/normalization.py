"""SSV workflow utilities."""



from .common import *



def _copy_numeric_frame(frame, name="matrix"):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must be non-empty")
    return frame.astype(float).copy()


def _restore_frame(values, index, columns):
    return pd.DataFrame(values, index=index.copy(), columns=columns.copy())


def identity_normalize(matrix):
    """Return an unchanged copy."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame")
    if matrix.empty:
        raise ValueError("matrix must be non-empty")
    return matrix.copy(deep=True)


def quantile_normalize(matrix):
    """Normalize columns to a shared empirical rank distribution."""
    frame = _copy_numeric_frame(matrix)
    values = frame.to_numpy()
    sorted_values = np.sort(values, axis=0)
    rank_means = sorted_values.mean(axis=1)
    ranks = frame.rank(axis=0, method="average", ascending=True).to_numpy() - 1

    normalized = np.empty_like(values, dtype=float)
    rank_positions = np.arange(len(rank_means), dtype=float)
    for col_idx in range(values.shape[1]):
        normalized[:, col_idx] = np.interp(ranks[:, col_idx], rank_positions, rank_means)
    return _restore_frame(normalized, frame.index, frame.columns)


def fit_quantile_reference(train):
    """Fit the sorted QN reference distribution from training data."""
    train_qn = quantile_normalize(train)
    return np.sort(train_qn.iloc[:, 0].to_numpy())


def apply_quantile_reference(test, reference):
    """Map each test column to a training-derived rank reference."""
    frame = _copy_numeric_frame(test)
    reference = np.asarray(reference, dtype=float)
    if len(reference) != frame.shape[0]:
        raise ValueError("reference length must equal the number of features")
    ranks = frame.rank(axis=0, method="average", ascending=True).to_numpy() - 1
    normalized = np.empty(frame.shape, dtype=float)
    rank_positions = np.arange(len(reference), dtype=float)
    for col_idx in range(frame.shape[1]):
        normalized[:, col_idx] = np.interp(ranks[:, col_idx], rank_positions, reference)
    return _restore_frame(normalized, frame.index, frame.columns)


def median_normalize(train, test):
    """Median-shift train and test using the scalar training median."""
    train_frame = _copy_numeric_frame(train, "train")
    test_frame = _copy_numeric_frame(test, "test")
    reference = np.nanmedian(train_frame.to_numpy())
    train_norm = train_frame.subtract(train_frame.median(axis=0) - reference, axis=1)
    test_norm = test_frame.subtract(test_frame.median(axis=0) - reference, axis=1)
    return train_norm, test_norm


def normalize_transformed_qn(train, test):
    """R quant.norm semantics: QN train, then frozen-reference test mapping."""
    train_qn = quantile_normalize(train)
    reference = np.sort(train_qn.iloc[:, 0].to_numpy())
    test_qn = apply_quantile_reference(test, reference)
    return train_qn, test_qn


def fit_vsn_approximation(train):
    """Fit a lightweight training-only VSN approximation."""
    train_frame = _copy_numeric_frame(train, "train")
    location = train_frame.median(axis=1)
    mad = train_frame.subtract(location, axis=0).abs().median(axis=1)
    scale = (mad * 1.4826).astype(float)
    positive_scale = scale[np.isfinite(scale) & (scale > 0)]
    fallback = float(np.nanmedian(positive_scale)) if not positive_scale.empty else 1.0
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    scale = scale.where(np.isfinite(scale) & (scale > 0), fallback)
    return {"location": location.astype(float), "scale": scale.astype(float)}


def apply_vsn_approximation(matrix, state):
    """Apply training-fitted VSN approximation parameters without refitting."""
    frame = _copy_numeric_frame(matrix)
    location = state["location"]
    scale = state["scale"]
    if not frame.index.equals(location.index) or not frame.index.equals(scale.index):
        raise ValueError("VSN state feature IDs must match matrix feature IDs")
    values = np.arcsinh(frame.subtract(location, axis=0).divide(scale, axis=0).to_numpy(dtype=float))
    return _restore_frame(values, frame.index, frame.columns)


def normalize_vsn(train, test):
    """Leakage-safe Python approximation to transformed-expression VSN."""
    state = fit_vsn_approximation(train)
    return apply_vsn_approximation(train, state), apply_vsn_approximation(test, state)


def _check_positive_factors(factors, name="scaling factors"):
    factors = pd.Series(factors, dtype=float)
    if not np.isfinite(factors.to_numpy()).all() or (factors <= 0).any():
        raise ValueError(f"{name} must be finite and positive")
    return factors


def _scale_columns(matrix, factors):
    frame = _copy_numeric_frame(matrix)
    factors = _check_positive_factors(pd.Series(factors, index=frame.columns), "column scaling factors")
    return frame.divide(factors, axis=1)


def normalize_tc(raw_counts):
    """Total-count CPM normalization."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = raw.sum(axis=0) / 1e6
    return _scale_columns(raw, factors), factors


def normalize_uq(raw_counts):
    """Upper-quartile scaling using strictly nonzero counts."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = raw.apply(lambda col: np.quantile(col[col != 0], 0.75), axis=0) / 1e6
    return _scale_columns(raw, factors), factors


def normalize_rna_median(raw_counts):
    """RNA median scaling using strictly nonzero counts per sample."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = raw.apply(lambda col: np.median(col[col != 0]), axis=0) / 1e6
    return _scale_columns(raw, factors), factors


def deseq_size_factors(raw_counts):
    """Estimate DESeq-style median-of-ratios size factors."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    positive_rows = (raw > 0).all(axis=1)
    if not positive_rows.any():
        raise ValueError("DESeq size factors require at least one feature positive in every sample")
    log_geo_means = np.log(raw.loc[positive_rows]).mean(axis=1)
    geo_means = np.exp(log_geo_means)
    ratios = raw.loc[positive_rows].divide(geo_means, axis=0)
    factors = ratios.median(axis=0)
    return _check_positive_factors(factors, "DESeq size factors")


def normalize_deseq(raw_counts):
    """DESeq size-factor normalization only."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = deseq_size_factors(raw)
    return _scale_columns(raw, factors), factors


def _select_tmm_reference(raw):
    lib_sizes = raw.sum(axis=0)
    upper_quartiles = raw.apply(lambda col: np.quantile(col[col > 0], 0.75), axis=0) / lib_sizes
    return int(np.argmin(np.abs(upper_quartiles - upper_quartiles.mean())))


def tmm_factors(raw_counts, ref_col=None, logratio_trim=0.30, sum_trim=0.05):
    """Estimate core edgeR-style TMM normalization factors."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    lib_sizes = raw.sum(axis=0).astype(float)
    _check_positive_factors(lib_sizes, "library sizes")
    ref_idx = _select_tmm_reference(raw) if ref_col is None else ref_col
    ref = raw.iloc[:, ref_idx]
    ref_lib = lib_sizes.iloc[ref_idx]
    factors = []

    for col_idx in range(raw.shape[1]):
        obs = raw.iloc[:, col_idx]
        obs_lib = lib_sizes.iloc[col_idx]
        keep = (obs > 0) & (ref > 0)
        if col_idx == ref_idx or keep.sum() < 2:
            factors.append(1.0)
            continue

        obs_p = obs[keep] / obs_lib
        ref_p = ref[keep] / ref_lib
        m_values = np.log2(obs_p / ref_p)
        a_values = 0.5 * np.log2(obs_p * ref_p)
        weights = 1 / ((1 / obs[keep]) - (1 / obs_lib) + (1 / ref[keep]) - (1 / ref_lib))
        finite = np.isfinite(m_values) & np.isfinite(a_values) & np.isfinite(weights) & (weights > 0)
        m_values = m_values[finite]
        a_values = a_values[finite]
        weights = weights[finite]
        n = len(m_values)
        if n < 2:
            factors.append(1.0)
            continue

        m_rank = pd.Series(m_values).rank(method="first").to_numpy()
        a_rank = pd.Series(a_values).rank(method="first").to_numpy()
        lo_m, hi_m = np.floor(n * logratio_trim) + 1, n - np.floor(n * logratio_trim)
        lo_a, hi_a = np.floor(n * sum_trim) + 1, n - np.floor(n * sum_trim)
        trim_keep = (m_rank >= lo_m) & (m_rank <= hi_m) & (a_rank >= lo_a) & (a_rank <= hi_a)
        if not trim_keep.any():
            factors.append(1.0)
            continue
        weighted_m = np.average(m_values[trim_keep], weights=weights[trim_keep])
        factors.append(2 ** weighted_m)

    factors = pd.Series(factors, index=raw.columns, dtype=float)
    factors = factors / np.exp(np.mean(np.log(factors)))
    return _check_positive_factors(factors, "TMM factors")


def normalize_tmm(raw_counts, ref_col=None):
    """TMM factor estimation followed by CPM with effective library sizes."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    lib_sizes = raw.sum(axis=0)
    factors = tmm_factors(raw, ref_col=ref_col)
    scaling = factors * lib_sizes / 1e6
    return _scale_columns(raw, scaling), scaling


def normalize_tmm_frozen(train_counts, test_counts):
    """Normalize first train sample plus test cohort, then return test columns only."""
    train = _copy_numeric_frame(train_counts, "train_counts")
    test = _copy_numeric_frame(test_counts, "test_counts")
    combined = pd.concat([train.iloc[:, [0]], test], axis=1)
    combined_norm, scaling = normalize_tmm(combined, ref_col=0)
    return combined_norm.iloc[:, 1:].copy(), scaling.iloc[1:].copy()


def normalize_rna_qn_frozen(train_counts, test_counts):
    train_qn = quantile_normalize(train_counts)
    reference = np.sort(train_qn.iloc[:, 0].to_numpy())
    test_qn = apply_quantile_reference(test_counts, reference)
    return train_qn, test_qn


def preprocess_train_test(train, test, method, expression_source):
    """Normalize train and test matrices using the R workflow method names."""
    if expression_source == "transformed_expression":
        if method == "non":
            return identity_normalize(train), identity_normalize(test)
        if method == "qn":
            return normalize_transformed_qn(train, test)
        if method == "mn":
            return median_normalize(train, test)
        if method == "vsn":
            return normalize_vsn(train, test)
        raise ValueError("unknown transformed_expression method")

    if expression_source == "raw_counts":
        if method == "non":
            return identity_normalize(train), identity_normalize(test)
        if method == "TC":
            return normalize_tc(train)[0], normalize_tc(test)[0]
        if method == "UQ":
            return normalize_uq(train)[0], normalize_uq(test)[0]
        if method == "med":
            return normalize_rna_median(train)[0], normalize_rna_median(test)[0]
        if method == "TMM":
            return normalize_tmm(train)[0], normalize_tmm(test)[0]
        if method == "TMM.frozen":
            train_norm = normalize_tmm(train)[0]
            test_norm = normalize_tmm_frozen(train, test)[0]
            return train_norm, test_norm
        if method == "DESeq":
            return normalize_deseq(train)[0], normalize_deseq(test)[0]
        if method == "QN":
            return quantile_normalize(train), quantile_normalize(test)
        if method == "QN.frozen":
            return normalize_rna_qn_frozen(train, test)
        raise ValueError("unknown raw_counts method")

    raise ValueError("expression_source must be 'transformed_expression' or 'raw_counts'")


def _assert_same_axes(original, normalized):
    assert original.shape == normalized.shape
    assert original.index.equals(normalized.index)
    assert original.columns.equals(normalized.columns)
    assert np.isfinite(normalized.to_numpy()).all()
