"""SSV workflow utilities."""



from dataclasses import dataclass
from .common import *



class ClassifierResult:
    predicted_labels: pd.Series
    probabilities: pd.Series | None
    inner_cv_error: float | None
    outer_test_error: float | None
    selected_params: dict


def _require_sklearn():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for this classifier but is not installed.")


def _safe_stratified_cv(labels, requested_folds=5, random_state=2024):
    _require_sklearn()
    counts = pd.Series(labels).value_counts()
    if counts.min() < 2:
        raise ValueError("at least two samples per class are required for stratified CV")
    n_splits = int(min(requested_folds, counts.min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _validate_classifier_inputs(train_expression, train_labels, test_expression, test_labels=None):
    if not isinstance(train_expression, pd.DataFrame) or not isinstance(test_expression, pd.DataFrame):
        raise TypeError("train_expression and test_expression must be pandas DataFrames")
    if not isinstance(train_labels, pd.Series):
        raise TypeError("train_labels must be a pandas Series")
    if not train_expression.columns.equals(train_labels.index):
        raise ValueError("train_expression columns must exactly match train_labels.index")
    if not train_expression.index.equals(test_expression.index):
        raise ValueError("train and test feature IDs must match exactly")
    if test_labels is not None:
        if not isinstance(test_labels, pd.Series):
            raise TypeError("test_labels must be a pandas Series")
        if not test_expression.columns.equals(test_labels.index):
            raise ValueError("test_expression columns must exactly match test_labels.index")

    classes = pd.Index(pd.unique(train_labels))
    if len(classes) != 2:
        raise ValueError("classifier helpers currently require exactly two training classes")

    train_x = train_expression.T.copy()
    test_x = test_expression.T.copy()
    return train_x, train_labels.copy(), test_x, None if test_labels is None else test_labels.copy(), classes


def _finish_classifier_result(predictions, probabilities, test_index, test_labels, inner_cv_error, selected_params):
    predicted_labels = pd.Series(predictions, index=test_index, name="predicted_label")
    prob_series = None
    if probabilities is not None:
        prob_series = pd.Series(probabilities, index=test_index, name="positive_class_probability")
    outer_error = None
    if test_labels is not None:
        outer_error = float(np.mean(predicted_labels.to_numpy() != test_labels.to_numpy()))
    return ClassifierResult(predicted_labels, prob_series, inner_cv_error, outer_error, selected_params)


def _positive_class_index(estimator, positive_class):
    return int(np.where(estimator.classes_ == positive_class)[0][0])


def _fit_grid_classifier(train_x, y, test_x, estimator, param_grid, cv, positive_class=None):
    grid = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring="accuracy")
    grid.fit(train_x, y)
    predictions = grid.predict(test_x)
    probabilities = None
    if positive_class is not None and hasattr(grid.best_estimator_, "predict_proba"):
        class_idx = _positive_class_index(grid.best_estimator_, positive_class)
        probabilities = grid.best_estimator_.predict_proba(test_x)[:, class_idx]
    return predictions, probabilities, float(1 - grid.best_score_), dict(grid.best_params_)


def _fit_rf(train_x, y, test_x, cv, positive_class, random_state=2024):
    _require_sklearn()
    candidates = [
        {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 150, "max_features": "sqrt", "min_samples_leaf": 2, "max_depth": None},
        {"n_estimators": 150, "max_features": 0.5, "min_samples_leaf": 1, "max_depth": 4},
        {"n_estimators": 200, "max_features": "log2", "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 3, "max_depth": 6},
    ]
    best_score = -np.inf
    best_params = candidates[0]
    for params in candidates:
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=random_state, **params))]
        )
        score = cross_val_score(estimator, train_x, y, cv=cv, scoring="accuracy").mean()
        if score > best_score:
            best_score = float(score)
            best_params = params
    final_model = Pipeline(
        [("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=random_state, **best_params))]
    )
    final_model.fit(train_x, y)
    predictions = final_model.predict(test_x)
    rf_model = final_model.named_steps["model"]
    class_idx = _positive_class_index(rf_model, positive_class)
    probabilities = final_model.predict_proba(test_x)[:, class_idx]
    return predictions, probabilities, float(1 - best_score), best_params.copy()


def _fit_xgb(train_x, y, test_x, classes, cv, random_state=2024):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is unavailable because `xgboost` is not installed.")
    class_to_int = {cls: idx for idx, cls in enumerate(classes)}
    y_int = y.map(class_to_int).astype(int).to_numpy()
    dtrain = xgb.DMatrix(train_x.to_numpy(dtype=float), label=y_int)
    best_error = np.inf
    best_params = None
    best_rounds = 1
    for min_child_weight in [1, 5, 9]:
        params = {
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "eval_metric": "error",
            "seed": random_state,
        }
        cvout = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=cv.get_n_splits(),
            num_boost_round=100,
            early_stopping_rounds=20,
            stratified=True,
            shuffle=True,
            seed=random_state,
            verbose_eval=False,
        )
        errors = cvout["test-error-mean"].to_numpy()
        candidate_error = float(errors.min())
        if candidate_error < best_error:
            best_error = candidate_error
            best_params = params
            best_rounds = int(errors.argmin() + 1)
    model = xgb.train(best_params, dtrain, num_boost_round=best_rounds, verbose_eval=False)
    probabilities = model.predict(xgb.DMatrix(test_x.to_numpy(dtype=float)))
    predictions = classes.to_numpy()[(probabilities > 0.5).astype(int)]
    return predictions, probabilities, best_error, {"min_child_weight": best_params["min_child_weight"], "n_rounds": best_rounds}


def fit_classifier(train_expression, train_labels, test_expression, test_labels=None, method="svm", config=None, random_state=2024):
    """Fit/tune one SSV classifier and predict test samples."""
    config = {} if config is None else dict(config)
    train_x, y, test_x, test_y, classes = _validate_classifier_inputs(
        train_expression, train_labels, test_expression, test_labels
    )
    positive_class = config.get("positive_class", classes[-1])
    cv = _safe_stratified_cv(y, requested_folds=config.get("kfold", 5), random_state=random_state)

    if method == "svm":
        _require_sklearn()
        estimator = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf"))])
        grid = {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.1, 1]}
        predictions, probabilities, inner_error, selected = _fit_grid_classifier(train_x, y, test_x, estimator, grid, cv)
    elif method == "lasso":
        _require_sklearn()
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))]
        )
        grid = {"model__C": [0.01, 0.1, 1, 10]}
        predictions, probabilities, inner_error, selected = _fit_grid_classifier(
            train_x, y, test_x, estimator, grid, cv, positive_class=positive_class
        )
        selected["selection_note"] = "R reports lambda.min CV error and predicts with lambda.1se; sklearn uses the selected C."
    elif method == "xgb":
        predictions, probabilities, inner_error, selected = _fit_xgb(train_x, y, test_x, classes, cv, random_state)
    elif method == "knn":
        _require_sklearn()
        max_valid_k = len(y) - int(np.ceil(len(y) / cv.get_n_splits()))
        k_values = [k for k in [1, 3, 5, 7, 9] if k <= max_valid_k]
        estimator = Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())])
        predictions, probabilities, inner_error, selected = _fit_grid_classifier(
            train_x, y, test_x, estimator, {"model__n_neighbors": k_values}, cv, positive_class=positive_class
        )
    elif method == "rf":
        predictions, probabilities, inner_error, selected = _fit_rf(train_x, y, test_x, cv, positive_class, random_state)
    else:
        raise ValueError("method must be one of: svm, lasso, xgb, knn, rf")

    return _finish_classifier_result(predictions, probabilities, test_x.index, test_y, inner_error, selected)
