import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score
)

#### ============================ Importing Data ============================ ####
df = pd.read_excel("Data.xlsx")

data1 = df[['R_12-18M', 'T10Y2Y', 'T10Y3M', 'BaaSpread', 'PERatioS&P']].dropna()
y = data1['R_12-18M']
X = data1[['T10Y2Y', 'BaaSpread']]

# Preserve the original feature list (these will be the columns used for training).
feature_cols = X.columns.tolist()

# Inversion flag (1 if inverted) based on the first column of X (the slope measure)
X["Inverted"] = (X.iloc[:, 0] < 0).astype(int)

# Threshold to convert predicted probabilities into binary predictions.
threshold = 0.5 

# Creating expanding window splits for time series cross-validation
def expanding_window_split(X, n_splits=5):
    n_samples = len(X)
    splits = []
    
    # Calculate test size for each fold
    test_size = n_samples // (n_splits + 1) # Train on 4 increasing folds and test on the next one
    
    for i in range(n_splits):
        # Train on expanding window: from start to current point
        train_end = (i + 2) * test_size # Begin training with 2 folds and add it from there
        train_idx = np.arange(train_end) # Create a sequence of numbers from 0 to train_end - 1
        
        # Test on next period
        test_start = train_end
        test_end = min(test_start + test_size, n_samples) # To prevent the test set from going beyond the available data.
        test_idx = np.arange(test_start, test_end)
        
        if len(test_idx) > 0:  # Make sure we have test data
            splits.append((train_idx, test_idx))
    
    return splits

# Get expanding window splits - trains on increasing amounts of historical data
expanding_splits = expanding_window_split(X)

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

# Show fold structure
display_fold_info(X, expanding_splits)

#### ============================  Time Series Cross-Validation for Model Evaluation ============================ ####

# Logit
clf = LogisticRegression(solver="liblinear", random_state=42) #  optimization algorithm that finds the best coefficients (β values) for the logistic regression by iteratively adjusting one parameter at a time until it minimizes the prediction error on your recession data.
# Its outputs are probabilities which are then transformed into 1 or 0 based on the treshold

f1s, precs, recs, aucs, pr_aucs = [], [], [], [], []
f1s_inv, precs_inv, recs_inv, aucs_inv, pr_aucs_inv = [], [], [], [], []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] # Selecting training and testing data for the explanatory variables based on indices set on expanding_splits function
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx] # Selecting training and testing data for the dependent variable based on indices set on expanding_splits function

    if len(np.unique(y_te)) < 2: #  # Skip folds that can't be properly evaluated (e.g., only one class) because certain machine learning metrics become mathematically undefined or meaningless when only one class is present in the test set
        continue
        
    clf.fit(X_tr[feature_cols], y_tr) # Trains (fits) the logistic regression model on the training data for the current fold
    proba = clf.predict_proba(X_te[feature_cols])[:, 1] # Calculating probabilities of recession for the test set
    preds = (proba >= threshold).astype(int) # Converting probabilities into binary predictions based on the defined threshold

    f1s.append(f1_score(y_te, preds, zero_division=0)) # 'zero_division=0' -> Instead of crashing with a math error (dividing by zero), it returns 0
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    aucs.append(roc_auc_score(y_te, proba))      
    pr_aucs.append(average_precision_score(y_te, proba))

    # When curve inverted
    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2: # Check if there are at least one inverted period on the inv_mask and if both classes are present on the testing data for the dependent variable        
        f1s_inv.append(f1_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        precs_inv.append(precision_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        recs_inv.append(recall_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        aucs_inv.append(roc_auc_score(y_te[inv_mask], proba[inv_mask]))
        pr_aucs_inv.append(average_precision_score(y_te[inv_mask], proba[inv_mask]))

print("\n=== 4-fold Time Series CV (Logistic Regression) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs), np.std(pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs_inv), np.std(precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs_inv),  np.std(recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs_inv), np.std(pr_aucs_inv)))

# Probit Classifier (using statsmodels for consistency with statistical analysis)
probit_f1s, probit_precs, probit_recs, probit_aucs, probit_pr_aucs = [], [], [], [], []
probit_f1s_inv, probit_precs_inv, probit_recs_inv, probit_aucs_inv, probit_pr_aucs_inv = [], [], [], [], []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    # Add constant for statsmodels probit
    X_tr_const = sm.add_constant(X_tr[feature_cols], has_constant='add')
    X_te_const = sm.add_constant(X_te[feature_cols], has_constant='add')
    
    probit_model = sm.Probit(y_tr, X_tr_const)
    probit_result = probit_model.fit(disp=0)
    
    probit_proba = probit_result.predict(X_te_const)
    probit_preds = (probit_proba >= threshold).astype(int)

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

print("\n=== 4-fold Time Series CV (Probit) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs), np.std(probit_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs),  np.std(probit_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs), np.std(probit_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs_inv), np.std(probit_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs_inv),  np.std(probit_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs_inv), np.std(probit_pr_aucs_inv)))

# Gradient Boosting Classifier
gb_f1s, gb_precs, gb_recs, gb_aucs, gb_pr_aucs = [], [], [], [], []
gb_f1s_inv, gb_precs_inv, gb_recs_inv, gb_aucs_inv, gb_pr_aucs_inv = [], [], [], [], []

gb_clf = GradientBoostingClassifier(
    random_state=42, # seed value that makes your machine learning results reproducible by controlling randomness (e.g., same initialization, convergence path, etc.))
    n_estimators=300, # Number of boosting stages (trees) to be built     
    learning_rate=0.05, # Controls how much each tree contributes to the final prediction
    max_depth=2 # How deep each individual tree can grow. Max_depth=2: Each tree can only make 2 levels of decisions (splits)
)

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] 
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    gb_clf.fit(X_tr[feature_cols], y_tr)
    gb_proba = gb_clf.predict_proba(X_te[feature_cols])[:, 1]
    gb_preds = (gb_proba >= threshold).astype(int)

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

print("\n=== 4-fold Time Series CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs), np.std(gb_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs_inv), np.std(gb_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs_inv),  np.std(gb_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs_inv), np.std(gb_pr_aucs_inv)))

# Random Forest Classifier
rf_f1s, rf_precs, rf_recs, rf_aucs, rf_pr_aucs = [], [], [], [], []
rf_f1s_inv, rf_precs_inv, rf_recs_inv, rf_aucs_inv, rf_pr_aucs_inv = [], [], [], [], []

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1, # Minimum number of samples required at each leaf node
    random_state=42,
    n_jobs=-1 # Number of CPU cores to use for parallel processing. n_jobs=-1: Use ALL available CPU cores (fastest)
)

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    rf_clf.fit(X_tr[feature_cols], y_tr)
    rf_proba = rf_clf.predict_proba(X_te[feature_cols])[:, 1]
    rf_preds = (rf_proba >= threshold).astype(int)

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

print("\n=== 4-fold Time Series CV (Random Forest) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs), np.std(rf_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs_inv), np.std(rf_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs_inv),  np.std(rf_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs_inv), np.std(rf_pr_aucs_inv))) 

## ============================ Computing coefficients, marginal effects, and VIFs ============================ ##

print("\n=== Statistical Analysis ===")
X_with_const = sm.add_constant(X[feature_cols], has_constant='add')

# Logit
logit_model = sm.Logit(y, X_with_const)
logit_results = logit_model.fit(disp=0)

print("\nLogistic Regression (Logit) Coefficients:")
print(logit_results.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

# Probit
probit_model = sm.Probit(y, X_with_const)
probit_results = probit_model.fit(disp=0)

print("\nProbit Regression Coefficients:")
print(probit_results.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

# Marginal Effects
probit_margeff = probit_results.get_margeff(at='overall')
print("\nProbit Average Marginal Effects (AME):")
print(probit_margeff.summary())

# Testing for Multicolinearity (VIF)
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
#print(spf_data)
#print(f"SPF data shape: {spf_data.shape}")
#print(f"SPF unique values: {spf_data['SPF'].unique()}")
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
