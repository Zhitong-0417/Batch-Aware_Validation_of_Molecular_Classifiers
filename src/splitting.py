"""SSV workflow utilities."""



from .common import *
from .data import *



def construct_meta_batches(dataset, mode="random", early_batch_ids=None, rng=None):
    """Construct two repetition-level meta-batches from raw batch labels."""
    sample_ids = list(dataset.expression.columns)
    batches = dataset.batch_labels
    raw_batches = pd.unique(batches.to_numpy())

    if mode == "temporal":
        if early_batch_ids is None:
            raise ValueError("early_batch_ids is required for temporal mode")
        early_batch_ids = set(early_batch_ids)
        group1_batches = [batch for batch in raw_batches if batch in early_batch_ids]
        group2_batches = [batch for batch in raw_batches if batch not in early_batch_ids]
    elif mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        # R batch_time=FALSE shuffles raw batch labels, then assigns whole labels to meta-batches.
        shuffled = rng.permutation(raw_batches)
        half_length = int(np.ceil(len(shuffled) / 2))
        group1_batches = list(shuffled[:half_length])
        group2_batches = list(shuffled[half_length:])
    else:
        raise ValueError("mode must be 'random' or 'temporal'")

    meta_batch_1 = [sample_id for sample_id in sample_ids if batches.loc[sample_id] in group1_batches]
    meta_batch_2 = [sample_id for sample_id in sample_ids if batches.loc[sample_id] in group2_batches]
    if not meta_batch_1 or not meta_batch_2:
        raise ValueError("both meta-batches must contain at least one sample")
    return meta_batch_1, meta_batch_2


def balance_meta_batches(meta_batch_1, meta_batch_2, rng):
    """Downsample the larger meta-batch to match the smaller one."""
    meta_batch_1 = list(meta_batch_1)
    meta_batch_2 = list(meta_batch_2)
    length_diff = abs(len(meta_batch_1) - len(meta_batch_2))
    if length_diff == 0:
        return meta_batch_1, meta_batch_2

    if len(meta_batch_1) > len(meta_batch_2):
        remove = set(rng.choice(meta_batch_1, size=length_diff, replace=False))
        return [sample_id for sample_id in meta_batch_1 if sample_id not in remove], meta_batch_2

    remove = set(rng.choice(meta_batch_2, size=length_diff, replace=False))
    return meta_batch_1, [sample_id for sample_id in meta_batch_2 if sample_id not in remove]


def r_sample_size(count, percentage):
    """Reproduce base R sample(size=...) coercion used by ml_sum().

    R accepts non-integer positive numeric size values in sample() and coerces
    them to integer by truncating toward zero, e.g. 984.9 -> 984.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if percentage < 0 or percentage > 100:
        raise ValueError("percentage must be between 0 and 100")
    return int(np.trunc(count * percentage / 100))


def split_meta_batches(meta_batch_1, meta_batch_2, split, rng, sample_order=None):
    """Split split-specific balanced meta-batches using the SSV percentage convention."""
    if split < 0 or split > 100:
        raise ValueError("split must be between 0 and 100")

    meta_batch_1 = list(meta_batch_1)
    meta_batch_2 = list(meta_batch_2)
    n_train_1 = r_sample_size(len(meta_batch_1), split)
    n_train_2 = r_sample_size(len(meta_batch_2), 100 - split)
    train_1 = set(rng.choice(meta_batch_1, size=n_train_1, replace=False)) if n_train_1 else set()
    train_2 = set(rng.choice(meta_batch_2, size=n_train_2, replace=False)) if n_train_2 else set()
    train_set = train_1 | train_2

    retained_set = set(meta_batch_1) | set(meta_batch_2)
    if sample_order is None:
        sample_order = meta_batch_1 + meta_batch_2
    train_ids = [sample_id for sample_id in sample_order if sample_id in train_set]
    test_ids = [sample_id for sample_id in sample_order if sample_id in retained_set and sample_id not in train_set]
    return train_ids, test_ids


def split_manifest_from_experiments(experiments):
    """Build a compact one-row-per-repetition/split audit manifest."""
    rows = []
    for experiment in experiments:
        for split_result in experiment["splits"]:
            train_ids = split_result["train_ids"]
            test_ids = split_result["test_ids"]
            rows.append(
                {
                    "repetition": experiment["repetition"],
                    "split_percentage": split_result["split_percentage"],
                    "meta_batch_1_labels": "|".join(map(str, experiment["meta_batch_1_labels"])),
                    "meta_batch_2_labels": "|".join(map(str, experiment["meta_batch_2_labels"])),
                    "meta_batch_1_n_before_balance": len(experiment["meta_batch_1_ids"]),
                    "meta_batch_2_n_before_balance": len(experiment["meta_batch_2_ids"]),
                    "meta_batch_1_n_after_balance": len(split_result["balanced_meta_batch_1_ids"]),
                    "meta_batch_2_n_after_balance": len(split_result["balanced_meta_batch_2_ids"]),
                    "train_1_n": len(set(train_ids) & set(split_result["balanced_meta_batch_1_ids"])),
                    "train_2_n": len(set(train_ids) & set(split_result["balanced_meta_batch_2_ids"])),
                    "train_n": len(train_ids),
                    "test_n": len(test_ids),
                    "overlap_n": len(set(train_ids) & set(test_ids)),
                }
            )
    return pd.DataFrame(rows)


def generate_split_experiments(
    dataset,
    split_percentages,
    times=1,
    mode="random",
    early_batch_ids=None,
    batch_balance=False,
    seed=2024,
):
    """Generate R ml_sum-style repetition meta-batches and split-specific train/test IDs."""
    sample_order = list(dataset.expression.columns)
    experiments = []
    for repetition in range(1, times + 1):
        # R uses set.seed(k + 2024) once per repetition.
        # NumPy uses a different RNG engine, so exact sampled IDs are not expected
        # to match R, but the seed hierarchy and RNG consumption order are preserved.
        rng = np.random.default_rng(seed + repetition)
        meta_batch_1, meta_batch_2 = construct_meta_batches(
            dataset,
            mode=mode,
            early_batch_ids=early_batch_ids,
            rng=rng,
        )

        meta_batch_1_labels = _ordered_unique(dataset.batch_labels.loc[meta_batch_1].to_numpy())
        meta_batch_2_labels = _ordered_unique(dataset.batch_labels.loc[meta_batch_2].to_numpy())

        splits = []
        for split in split_percentages:
            if batch_balance:
                balanced_meta_batch_1, balanced_meta_batch_2 = balance_meta_batches(meta_batch_1, meta_batch_2, rng)
            else:
                balanced_meta_batch_1, balanced_meta_batch_2 = list(meta_batch_1), list(meta_batch_2)

            train_ids, test_ids = split_meta_batches(
                balanced_meta_batch_1,
                balanced_meta_batch_2,
                split,
                rng,
                sample_order=sample_order,
            )
            splits.append(
                {
                    "split_percentage": split,
                    "train_ids": train_ids,
                    "test_ids": test_ids,
                    "balanced_meta_batch_1_ids": balanced_meta_batch_1,
                    "balanced_meta_batch_2_ids": balanced_meta_batch_2,
                }
            )

        experiments.append(
            {
                "repetition": repetition,
                "meta_batch_1_ids": meta_batch_1,
                "meta_batch_2_ids": meta_batch_2,
                "meta_batch_1_labels": meta_batch_1_labels,
                "meta_batch_2_labels": meta_batch_2_labels,
                "splits": splits,
            }
        )
    return experiments


def _ordered_dataset_members(dataset, selected):
    return _ordered_members(dataset.expression.columns, selected)


def balance_batches_once(dataset, train_batch, test_batch, rng):
    train_ids = list(dataset.batch_labels.index[dataset.batch_labels == train_batch])
    test_ids = list(dataset.batch_labels.index[dataset.batch_labels == test_batch])
    return balance_meta_batches(train_ids, test_ids, rng)


def balance_batches_by_class(dataset, train_batch, test_batch, rng):
    train_retained, test_retained = [], []
    for class_label in sorted(pd.unique(dataset.class_labels), key=lambda value: str(value)):
        train_ids = [sid for sid in dataset.expression.columns if dataset.batch_labels.loc[sid] == train_batch and dataset.class_labels.loc[sid] == class_label]
        test_ids = [sid for sid in dataset.expression.columns if dataset.batch_labels.loc[sid] == test_batch and dataset.class_labels.loc[sid] == class_label]
        keep = min(len(train_ids), len(test_ids))
        if keep:
            train_retained.extend(_ordered_dataset_members(dataset, set(rng.choice(train_ids, size=keep, replace=False))))
            test_retained.extend(_ordered_dataset_members(dataset, set(rng.choice(test_ids, size=keep, replace=False))))
    return _ordered_dataset_members(dataset, train_retained), _ordered_dataset_members(dataset, test_retained)


def construct_fixed_retained_population(dataset, train_batch, test_batch, rng, prevalence_mode="natural"):
    if prevalence_mode == "natural":
        return balance_batches_once(dataset, train_batch, test_batch, rng)
    if prevalence_mode == "class_stratified":
        return balance_batches_by_class(dataset, train_batch, test_batch, rng)
    raise ValueError("prevalence_mode must be 'natural' or 'class_stratified'")


def generate_directional_splits_pro(dataset, *, direction, split_percentages, n_repetitions, seed, prevalence_mode="natural"):
    if direction not in PRO_DIRECTION_BATCHES:
        raise ValueError(f"unknown Pro direction: {direction}")
    train_batch, test_batch = PRO_DIRECTION_BATCHES[direction]
    repetitions = []
    for repetition in range(1, n_repetitions + 1):
        rng = np.random.default_rng(seed + repetition)
        retained_train, retained_test = construct_fixed_retained_population(dataset, train_batch, test_batch, rng, prevalence_mode=prevalence_mode)
        retained_train = _ordered_dataset_members(dataset, retained_train)
        retained_test = _ordered_dataset_members(dataset, retained_test)
        splits = []
        for split in tuple(split_percentages):
            n_train_from_train = r_sample_size(len(retained_train), split)
            n_train_from_test = r_sample_size(len(retained_test), 100 - split)
            train_from_train = set(rng.choice(retained_train, size=n_train_from_train, replace=False)) if n_train_from_train else set()
            train_from_test = set(rng.choice(retained_test, size=n_train_from_test, replace=False)) if n_train_from_test else set()
            train_set = train_from_train | train_from_test
            retained_set = set(retained_train) | set(retained_test)
            train_ids = _ordered_dataset_members(dataset, train_set)
            test_ids = [sid for sid in dataset.expression.columns if sid in retained_set and sid not in train_set]
            splits.append({"split_percentage": split, "train_ids": train_ids, "test_ids": test_ids, "retained_train_batch_ids": retained_train, "retained_test_batch_ids": retained_test})
        repetitions.append({"repetition": repetition, "direction": direction, "train_batch": train_batch, "test_batch": test_batch, "prevalence_mode": prevalence_mode, "retained_train_batch_ids": retained_train, "retained_test_batch_ids": retained_test, "splits": splits})
    return repetitions


def pro_split_manifest(split_experiments):
    rows = []
    for experiment in split_experiments:
        retained_train = set(experiment["retained_train_batch_ids"])
        retained_test = set(experiment["retained_test_batch_ids"])
        for split_result in experiment["splits"]:
            train_ids = set(split_result["train_ids"])
            test_ids = set(split_result["test_ids"])
            rows.append({"repetition": experiment["repetition"], "direction": experiment["direction"], "prevalence_mode": experiment["prevalence_mode"], "split_percentage": split_result["split_percentage"], "train_batch": experiment["train_batch"], "test_batch": experiment["test_batch"], "retained_train_batch_n": len(retained_train), "retained_test_batch_n": len(retained_test), "train_from_train_batch_n": len(train_ids & retained_train), "train_from_test_batch_n": len(train_ids & retained_test), "train_n": len(train_ids), "test_n": len(test_ids), "overlap_n": len(train_ids & test_ids)})
    return pd.DataFrame(rows)
