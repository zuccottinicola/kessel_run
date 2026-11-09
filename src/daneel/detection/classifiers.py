"""Classifier interfaces reserved for the detection course exercises."""

import numpy as np

from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


class RandomForestClassifier:
    """Random Forest detector for TESS exoplanet transit candidates."""

    def __init__(self, n_estimators=500, max_depth=None,
                 min_samples_leaf=1, max_features='sqrt',
                 bootstrap=True, class_weight=None,
                 random_state=42, n_jobs=-1, oob_score=True):

        # Initialize the Random Forest using the hyperparameters
        # provided by the user
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=oob_score
        )

    def fit(self, X_train, y_train):
        """Train the Random Forest and display the OOB score."""

        # Train the classifier using labelled TESS light curves.
        self.model.fit(X_train, y_train)

        # Print the Out-of-Bag score if available.
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
            print(f"OOB Score: {self.model.oob_score_:.4f}")

        return self

    def evaluate_with_optimal_threshold(self, X_test, y_test):
        """Evaluate the classifier using Youden's J statistic."""

        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)

        # Extract the predicted probabilities for class 1 (planet).
        proba = self.model.predict_proba(X_test)[:, 1]

        # Compute the ROC curve and find the threshold maximizing
        # Youden's J statistic: J = TPR - FPR.
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        # Compare predictions at the default and optimal thresholds.
        y_pred_default = (proba >= 0.5).astype(int)
        y_pred_best = (proba >= best_thresh).astype(int)

        # Calculate ROC-AUC and classification accuracy.
        auc = roc_auc_score(y_test, proba)
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_best = accuracy_score(y_test, y_pred_best)

        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"  At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")

        # Display precision, recall and F1-score for each class.
        print("\nClassification report (optimal threshold):")
        print(classification_report(
            y_test,
            y_pred_best,
            target_names=['Non-Planet', 'Planet'],
            digits=4,
            zero_division=0
        ))

        # Display the confusion matrix: [[TN, FP], [FN, TP]].
        cm = confusion_matrix(y_test, y_pred_best)
        print("\nConfusion matrix:")
        print(cm)

        # Display the confusion matrix at threshold 0.5 for comparison 
        cm_default = confusion_matrix(y_test, y_pred_default)
        print("\nConfusion matrix at default threshold (0.5):")
        print(cm_default)

        # Return predictions and ROC data for further analysis.
        return y_pred_best, proba, best_thresh, (fpr, tpr, thresholds)

class CNNClassifier:
    """Placeholder for the CNN exercise."""

    def fit(self, X, y):
        raise NotImplementedError("CNN training is a course exercise")
