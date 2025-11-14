import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

#### ============================ Importing Data ============================ ####
df = pd.read_excel("Data.xlsx")
print(df)

data1 = df[['R_12-18M', 'T10Y2Y', 'T10Y3M', 'BaaSpread', 'PERatioS&P']].dropna()
y = data1['R_12-18M']
X = data1[['T10Y2Y']].copy()  # <- copy() avoids SettingWithCopyWarning

# Columns used for training
feature_cols = X.columns.tolist()

# Inversion flag (1 if inverted) based on the first column of X (the slope measure)
X["Inverted"] = (X.iloc[:, 0] < 0).astype(int)

# Threshold to convert predicted probabilities into binary predictions.
threshold = 0.5 

#### ============================ Creating expanding window splits ============================ ####
def expanding_window_split(X, n_splits=5):
    n_samples = len(X)
    splits = []

    # Split indices into (n_splits + 2) nearly-equal blocks so there is enough for training and testing
    blocks = np.array_split(np.arange(n_samples), n_splits + 2)

    for i in range(n_splits):
        train_blocks = blocks[: i + 2]  # Training with the first (i+2) blocks
        if len(train_blocks) == 0:
            continue
        train_idx = np.concatenate(train_blocks)

        test_idx = blocks[i + 2]  # Testing with the block after the training blocks

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits

# Get expanding window splits - trains on increasing amounts of historical data
expanding_splits = expanding_window_split(X, n_splits=3)

# Display fold information
def display_fold_info(X, splits):
    print("=" * 70)
    print(f"Cross-validation Fold Structure")
    print(f"Total data points: {len(X)}")
    
    # Create date index if not available
    if hasattr(X, 'index') and hasattr(X.index, 'strftime'):
        dates = X.index
    else:
        # Assume monthly data from Nov 1980 to Apr 2024
        dates = pd.date_range(start='1980-11-01', periods=len(X), freq='MS')
    
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        train_start = dates[0]
        train_end = dates[train_idx[-1]]
        test_start = dates[test_idx[0]]
        test_end = dates[test_idx[-1]]
        
        print(f"\nFOLD {i}:")
        print(f"  Training:  {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')} ({len(train_idx):3d} months)")
        print(f"  Testing:   {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} ({len(test_idx):3d} months)")
    
    print(f"\n✅ Total folds: {len(splits)}")
    print("=" * 70)

display_fold_info(X, expanding_splits)

#### ============================  Time Series Cross-Validation for Model Evaluation ============================ ####

# helper lists for PR curves (per fold, per model)
all_y_logit, all_proba_logit = [], []
all_y_probit, all_proba_probit = [], []
all_y_gb, all_proba_gb = [], []
all_y_rf, all_proba_rf = [], []

# -------------------- Logistic Regression -------------------- #
clf = LogisticRegression(solver="liblinear", random_state=42)

f1s, precs, recs, aucs, pr_aucs = [], [], [], [], []
f1s_inv, precs_inv, recs_inv, aucs_inv, pr_aucs_inv = [], [], [], [], []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    clf.fit(X_tr[feature_cols], y_tr)
    proba = clf.predict_proba(X_te[feature_cols])[:, 1]
    preds = (proba >= threshold).astype(int)

    # store per-fold test data for PR curves
    all_y_logit.append(y_te.values)
    all_proba_logit.append(proba)

    f1s.append(f1_score(y_te, preds, zero_division=0))
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    aucs.append(roc_auc_score(y_te, proba))      
    pr_aucs.append(average_precision_score(y_te, proba))

    # When curve inverted
    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        f1s_inv.append(f1_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        precs_inv.append(precision_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        recs_inv.append(recall_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        aucs_inv.append(roc_auc_score(y_te[inv_mask], proba[inv_mask]))
        pr_aucs_inv.append(average_precision_score(y_te[inv_mask], proba[inv_mask]))

print("\n=== {}-fold Time Series CV (Logistic Regression) @ threshold = {:.2f} (mean ± std) ===".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs), np.std(pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs_inv), np.std(precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs_inv),  np.std(recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs_inv), np.std(pr_aucs_inv)))

# -------------------- Probit Classifier -------------------- #
probit_f1s, probit_precs, probit_recs, probit_aucs, probit_pr_aucs = [], [], [], [], []
probit_f1s_inv, probit_precs_inv, probit_recs_inv, probit_aucs_inv, probit_pr_aucs_inv = [], [], [], [], []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    X_tr_const = sm.add_constant(X_tr[feature_cols], has_constant='add')
    X_te_const = sm.add_constant(X_te[feature_cols], has_constant='add')
    
    probit_model = sm.Probit(y_tr, X_tr_const)
    probit_result = probit_model.fit(disp=0)
    
    probit_proba = probit_result.predict(X_te_const)
    probit_preds = (probit_proba >= threshold).astype(int)

    # store per-fold test data for PR curves
    all_y_probit.append(y_te.values)
    all_proba_probit.append(probit_proba)

    probit_f1s.append(f1_score(y_te, probit_preds, zero_division=0))
    probit_precs.append(precision_score(y_te, probit_preds, zero_division=0))
    probit_recs.append(recall_score(y_te, probit_preds, zero_division=0))
    probit_aucs.append(roc_auc_score(y_te, probit_proba))
    probit_pr_aucs.append(average_precision_score(y_te, probit_proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        probit_f1s_inv.append(f1_score(y_te[inv_mask], probit_preds[inv_mask], zero_division=0))
        probit_precs_inv.append(precision_score(y_te[inv_mask], probit_preds[inv_mask], zero_division=0))
        probit_recs_inv.append(recall_score(y_te[inv_mask], probit_preds[inv_mask], zero_division=0))
        probit_aucs_inv.append(roc_auc_score(y_te[inv_mask], probit_proba[inv_mask]))
        probit_pr_aucs_inv.append(average_precision_score(y_te[inv_mask], probit_proba[inv_mask]))

print("\n=== {}-fold Time Series CV (Probit) @ threshold = {:.2f} (mean ± std) ===".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs), np.std(probit_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs),  np.std(probit_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs), np.std(probit_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs_inv), np.std(probit_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs_inv),  np.std(probit_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs_inv), np.std(probit_pr_aucs_inv)))

# -------------------- Gradient Boosting Classifier -------------------- #
gb_f1s, gb_precs, gb_recs, gb_aucs, gb_pr_aucs = [], [], [], [], []
gb_f1s_inv, gb_precs_inv, gb_recs_inv, gb_aucs_inv, gb_pr_aucs_inv = [], [], [], [], []

gb_clf = GradientBoostingClassifier(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=2
)

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] 
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    gb_clf.fit(X_tr[feature_cols], y_tr)
    gb_proba = gb_clf.predict_proba(X_te[feature_cols])[:, 1]
    gb_preds = (gb_proba >= threshold).astype(int)

    # store per-fold test data for PR curves
    all_y_gb.append(y_te.values)
    all_proba_gb.append(gb_proba)

    gb_f1s.append(f1_score(y_te, gb_preds, zero_division=0))
    gb_precs.append(precision_score(y_te, gb_preds, zero_division=0))
    gb_recs.append(recall_score(y_te, gb_preds, zero_division=0))
    gb_aucs.append(roc_auc_score(y_te, gb_proba))
    gb_pr_aucs.append(average_precision_score(y_te, gb_proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        gb_f1s_inv.append(f1_score(y_te[inv_mask], gb_preds[inv_mask], zero_division=0))
        gb_precs_inv.append(precision_score(y_te[inv_mask], gb_preds[inv_mask], zero_division=0))
        gb_recs_inv.append(recall_score(y_te[inv_mask], gb_preds[inv_mask], zero_division=0))
        gb_aucs_inv.append(roc_auc_score(y_te[inv_mask], gb_proba[inv_mask]))
        gb_pr_aucs_inv.append(average_precision_score(y_te[inv_mask], gb_proba[inv_mask]))

print("\n=== {}-fold Time Series CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std) ===".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs), np.std(gb_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs_inv), np.std(gb_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs_inv),  np.std(gb_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs_inv), np.std(gb_pr_aucs_inv)))

# -------------------- Random Forest Classifier -------------------- #
rf_f1s, rf_precs, rf_recs, rf_aucs, rf_pr_aucs = [], [], [], [], []
rf_f1s_inv, rf_precs_inv, rf_recs_inv, rf_aucs_inv, rf_pr_aucs_inv = [], [], [], [], []

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    rf_clf.fit(X_tr[feature_cols], y_tr)
    rf_proba = rf_clf.predict_proba(X_te[feature_cols])[:, 1]
    rf_preds = (rf_proba >= threshold).astype(int)

    # store per-fold test data for PR curves
    all_y_rf.append(y_te.values)
    all_proba_rf.append(rf_proba)

    rf_f1s.append(f1_score(y_te, rf_preds, zero_division=0))
    rf_precs.append(precision_score(y_te, rf_preds, zero_division=0))
    rf_recs.append(recall_score(y_te, rf_preds, zero_division=0))
    rf_aucs.append(roc_auc_score(y_te, rf_proba))
    rf_pr_aucs.append(average_precision_score(y_te, rf_proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        rf_f1s_inv.append(f1_score(y_te[inv_mask], rf_preds[inv_mask], zero_division=0))
        rf_precs_inv.append(precision_score(y_te[inv_mask], rf_preds[inv_mask], zero_division=0))
        rf_recs_inv.append(recall_score(y_te[inv_mask], rf_preds[inv_mask], zero_division=0))
        rf_aucs_inv.append(roc_auc_score(y_te[inv_mask], rf_proba[inv_mask]))
        rf_pr_aucs_inv.append(average_precision_score(y_te[inv_mask], rf_proba[inv_mask]))

print("\n=== {}-fold Time Series CV (Random Forest) @ threshold = {:.2f} (mean ± std) ===".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs), np.std(rf_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs_inv), np.std(rf_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs_inv),  np.std(rf_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs_inv), np.std(rf_pr_aucs_inv))) 

## ============================ PR–AUC CURVES: 4 FIGURES, 3 FOLDS EACH ============================ ##

def plot_per_model_pr_curves(all_y, all_proba, model_name, filename_prefix):
    """
    Plot one figure for a given model, with one PR curve per fold.
    """
    if len(all_y) == 0:
        return
    
    plt.figure(figsize=(7, 6))
    
    for fold_idx, (y_fold, proba_fold) in enumerate(zip(all_y, all_proba), start=1):
        prec_curve, rec_curve, _ = precision_recall_curve(y_fold, proba_fold)
        ap = average_precision_score(y_fold, proba_fold)
        plt.plot(rec_curve, prec_curve, label=f"Fold {fold_idx} (PR AUC = {ap:.3f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curves – {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

# one PNG per econometric framework
plot_per_model_pr_curves(all_y_logit,  all_proba_logit,  "Logistic Regression", "logit")
plot_per_model_pr_curves(all_y_probit, all_proba_probit, "Probit",              "probit")
plot_per_model_pr_curves(all_y_gb,     all_proba_gb,     "Gradient Boosting",   "gb")
plot_per_model_pr_curves(all_y_rf,     all_proba_rf,     "Random Forest",       "rf")

## ============================ Computing coefficients, marginal effects, and VIFs ============================ ##

print("\n=== Statistical Analysis ===")
X_with_const = sm.add_constant(X[feature_cols], has_constant='add')

# Logit
logit_model = sm.Logit(y, X_with_const)
logit_results = logit_model.fit(disp=0)

print("\nLogistic Regression (Logit) Coefficients:")
print(logit_results.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

# Probit
probit_model_full = sm.Probit(y, X_with_const)
probit_results_full = probit_model_full.fit(disp=0)

print("\nProbit Regression Coefficients:")
print(probit_results_full.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

# Marginal Effects
probit_margeff = probit_results_full.get_margeff(at='overall')
print("\nProbit Average Marginal Effects (AME):")
print(probit_margeff.summary())

# Testing for Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_with_const.values, i)
    for i in range(X_with_const.shape[1])
]

print("\n=== Variance Inflation Factors (VIF) ===")
print(vif_data[vif_data["feature"] != "const"])

#### ============================ SPF Benchmark ============================ ####

# SPF precision
df_spf = pd.read_excel("Data.xlsx")
spf_data = df_spf[['R_12-18M', 'SPF']].dropna()
spf_precision = precision_score(spf_data['R_12-18M'], spf_data['SPF'], zero_division=0)

# Models' average precision 
all_precisions = precs + probit_precs + gb_precs + rf_precs
all_precisions_inv = precs_inv + probit_precs_inv + gb_precs_inv + rf_precs_inv
model_avg_precision = np.mean(all_precisions)
model_avg_precision_inv = np.mean(all_precisions_inv)

print(f"\nPrecision Comparison:")
print(f"SPF: {spf_precision:.3f}")
print(f"Models (avg): {model_avg_precision:.3f}")
print(f"Difference: {model_avg_precision - spf_precision:+.3f}")

print(f"\nPrecision Comparison (Inverted Yield Curve):")
print(f"SPF: {spf_precision:.3f}")
print(f"Models (avg): {model_avg_precision_inv:.3f}")
print(f"Difference: {model_avg_precision_inv - spf_precision:+.3f}")

t_stat, p_value = ttest_1samp(all_precisions, spf_precision, alternative='greater')
print("Precision improvement significance (models > SPF): p =", p_value)

#### ============================ Per-fold Precision & Recall ============================ ####
print("\n=== Per-fold Precision & Recall (by model) ===")

def _print_per_fold(name, precs_list, recs_list):
    if len(precs_list) == 0 and len(recs_list) == 0:
        print(f"\n{name}: No valid folds (no recorded precision/recall).")
        return
    n = max(len(precs_list), len(recs_list))
    print(f"\n{name} (n_valid_folds = {n}):")
    for i in range(n):
        p = precs_list[i] if i < len(precs_list) else float('nan')
        r = recs_list[i] if i < len(recs_list) else float('nan')
        if np.isnan(p) and np.isnan(r):
            print(f"  Fold {i+1}: no metric")
        else:
            print(f"  Fold {i+1}: Precision = {p:.3f}, Recall = {r:.3f}")

_print_per_fold('Logistic Regression', precs, recs)
_print_per_fold('Probit', probit_precs, probit_recs)
_print_per_fold('Gradient Boosting', gb_precs, gb_recs)
_print_per_fold('Random Forest', rf_precs, rf_recs)
