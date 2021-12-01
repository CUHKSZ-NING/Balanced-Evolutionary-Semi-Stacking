# Balanced-Evolutionary-Semi-Stacking

* Required Python 3 packages: 
    1. sklearn (https://github.com/scikit-learn/scikit-learn)
    2. imblearn (https://github.com/scikit-learn-contrib/imbalanced-learn)
    3. lightgbm (https://github.com/microsoft/LightGBM)

* BESS is compatible with most sklearn APIs but is not strictly tested.

* Run: `python3 main.py`

* Import: `from BalancedEvolutionarySemiStacking import BalancedEvolutionarySemiStacking`

* Train: `fit(X, y)`, with target `-1` as the unlabeled data, `0` as the majority class, and `1` as the minority class.

* Predict: `predict(X)` or `predict_proba(X)`
