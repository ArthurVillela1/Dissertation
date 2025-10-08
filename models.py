import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score
)

df = pd.read_excel("Data.xlsx")

data1 = df[['R_12M', 'R_6M', 'T10Y2Y', 'T10Y3M', 'BaaSpread', 'E-Rule', 'PERatioS&P']].dropna()
y = data1['R_12M']
X = data1[['T10Y2Y', 'BaaSpread']]


threshold = 0.5 # Threshold to convert predicted probabilities into binary predictions.
tscv = TimeSeriesSplit(n_splits=5) # Cross-validation strategy that creates 5 sequential train/test splits while respecting temporal order.

#Logit
clf = LogisticRegression(solver="liblinear", random_state=42) # "liblinear" is a solver that uses a coordinate descent algorithm

f1s, precs, recs, aucs, pr_aucs = [], [], [], [], []

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] # Splits feature matrix X into training and testing sets using the indices provided by TimeSeriesSplit. iloc gets rows from X using the row numbers in train_idx
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx] # Splits target vector y into training and testing sets using the indices provided by TimeSeriesSplit

    if len(np.unique(y_te)) < 2: # Skip folds that can't be properly evaluated because certain machine learning metrics become mathematically undefined or meaningless when only one class is present in the test set
        continue
        
    clf.fit(X_tr, y_tr) # Trains (fits) the logistic regression model on the training data.
    proba = clf.predict_proba(X_te)[:, 1] # Gets the recession probabilities for the test set from the trained model.
    
    preds = (proba >= threshold).astype(int) # Converts probabilities into binary classifications     

    f1s.append(f1_score(y_te, preds, zero_division=0)) # 'zero_division=0' -> Instead of crashing with a math error (dividing by zero), it returns 0
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    aucs.append(roc_auc_score(y_te, proba))      
    pr_aucs.append(average_precision_score(y_te, proba))

print("\n=== 5-fold Time Series CV @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(f1s),  np.std(f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(aucs),  np.std(aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs), np.std(pr_aucs)))

# Gradient Boosting Classifier
gb_f1s, gb_precs, gb_recs, gb_aucs, gb_pr_aucs = [], [], [], [], []

gb_clf = GradientBoostingClassifier(
    random_state=42,
    n_estimators=300,     
    learning_rate=0.05, # Controls how much each tree contributes to the final prediction
    max_depth=2 # How deep each individual tree can grow. Max_depth=2: Each tree can only make 2 levels of decisions (splits)
)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] 
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    gb_clf.fit(X_tr, y_tr)
    gb_proba = gb_clf.predict_proba(X_te)[:, 1]
    gb_preds = (gb_proba >= threshold).astype(int)

    gb_f1s.append(f1_score(y_te, gb_preds, zero_division=0))
    gb_precs.append(precision_score(y_te, gb_preds, zero_division=0))
    gb_recs.append(recall_score(y_te, gb_preds, zero_division=0))
    gb_aucs.append(roc_auc_score(y_te, gb_proba))
    gb_pr_aucs.append(average_precision_score(y_te, gb_proba))

print("\n=== 5-fold Time Series CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(gb_f1s),  np.std(gb_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(gb_aucs),  np.std(gb_aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs), np.std(gb_pr_aucs)))

# Random Forest Classifier
rf_f1s, rf_precs, rf_recs, rf_aucs, rf_pr_aucs = [], [], [], [], []

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1, # Minimum number of samples required at each leaf node
    random_state=42,
    n_jobs=-1 # Number of CPU cores to use for parallel processing. -1 means using all available cores
)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    rf_clf.fit(X_tr, y_tr)
    rf_proba = rf_clf.predict_proba(X_te)[:, 1]
    rf_preds = (rf_proba >= threshold).astype(int)

    rf_f1s.append(f1_score(y_te, rf_preds, zero_division=0))
    rf_precs.append(precision_score(y_te, rf_preds, zero_division=0))
    rf_recs.append(recall_score(y_te, rf_preds, zero_division=0))
    rf_aucs.append(roc_auc_score(y_te, rf_proba))
    rf_pr_aucs.append(average_precision_score(y_te, rf_proba)) 

print("\n=== 5-fold Time Series CV (Random Forest) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(rf_f1s),  np.std(rf_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(rf_aucs),  np.std(rf_aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs), np.std(rf_pr_aucs))) 

# Naive Bayes Classifier
nb_f1s, nb_precs, nb_recs, nb_aucs, nb_pr_aucs = [], [], [], [], []

nb_clf = GaussianNB()

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    nb_clf.fit(X_tr, y_tr)
    nb_proba = nb_clf.predict_proba(X_te)[:, 1]
    nb_preds = (nb_proba >= threshold).astype(int)

    nb_f1s.append(f1_score(y_te, nb_preds, zero_division=0))
    nb_precs.append(precision_score(y_te, nb_preds, zero_division=0))
    nb_recs.append(recall_score(y_te, nb_preds, zero_division=0))
    nb_aucs.append(roc_auc_score(y_te, nb_proba))
    nb_pr_aucs.append(average_precision_score(y_te, nb_proba))

print("\n=== 5-fold Time Series CV (Naive Bayes) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(nb_f1s),  np.std(nb_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(nb_precs), np.std(nb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(nb_recs),  np.std(nb_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(nb_aucs),  np.std(nb_aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(nb_pr_aucs), np.std(nb_pr_aucs)))

# Statistical significance analysis using statsmodels
print("\n=== Statistical Significance Analysis ===")
X_with_const = sm.add_constant(X, has_constant='add')
logit_model = sm.Logit(y, X_with_const)
logit_results = logit_model.fit(disp=0)

print("Logistic Regression Coefficients:")
print(logit_results.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

# Calculate odds ratios
odds_ratios = np.exp(logit_results.params)
print(f"\nOdds Ratios:")
for var, or_val in odds_ratios.items():
    if var != 'const':
        pval = logit_results.pvalues[var]
        significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{var}: {or_val:.3f} {significance}")

# Testing for multicolinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_with_const.values, i)
    for i in range(X_with_const.shape[1])
]

print("\n=== Variance Inflation Factors (VIF) ===")
print(vif_data[vif_data["feature"] != "const"])