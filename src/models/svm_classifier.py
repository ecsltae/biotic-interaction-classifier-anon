#!/usr/bin/env python3
"""
Biotic Interaction Classifier - HIGH PRECISION VERSION
=======================================================
Optimized for precision over recall.
When in doubt, predict NEGATIVE (no interaction).

Strategies for high precision:
1. Higher classification thresholds
2. Precision-optimized hyperparameters
3. Class weights favoring precision
4. Threshold tuning on validation set
"""

import pandas as pd
import numpy as np
import pickle
import re
import time
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, 
                             precision_score, recall_score, precision_recall_curve,
                             make_scorer)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')


# =============================================================================
# CUSTOM FEATURE EXTRACTORS (same as before)
# =============================================================================

class InteractionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, interaction_terms=None):
        self.interaction_terms = interaction_terms or set()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_lower = text.lower()
            interaction_count = sum(1 for term in self.interaction_terms 
                                   if term in text_lower)
            has_predation = int(any(w in text_lower for w in 
                ['predator', 'prey', 'predation', 'hunt', 'consume', 'eat']))
            has_parasitism = int(any(w in text_lower for w in 
                ['parasite', 'parasitic', 'parasitism', 'host', 'infect']))
            has_symbiosis = int(any(w in text_lower for w in 
                ['symbiosis', 'symbiotic', 'mutualism', 'commensal']))
            has_pollination = int(any(w in text_lower for w in 
                ['pollinate', 'pollinator', 'pollination', 'flower']))
            features.append([interaction_count, has_predation, has_parasitism, 
                           has_symbiosis, has_pollination])
        return np.array(features)


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_len = char_count / max(word_count, 1)
            species_indicators = sum(1 for w in words if w in [
                'species', 'genus', 'strain', 'isolate', 'var', 'sp', 'spp'])
            relationship_verbs = sum(1 for w in words if w in [
                'infects', 'infected', 'transmits', 'transmitted', 'causes',
                'affects', 'colonizes', 'feeds', 'eats', 'consumes',
                'pollinates', 'disperses', 'attacks', 'kills', 'parasitizes'])
            features.append([word_count, avg_word_len, species_indicators, relationship_verbs])
        return np.array(features)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(csv_path):
    df = pd.read_csv(csv_path)
    sentences = df['passage'].tolist()
    labels = df['label'].values
    print(f"Training data: {len(sentences)} samples")
    print(f"  Label 0: {np.sum(labels == 0)}, Label 1: {np.sum(labels == 1)}")
    return sentences, labels


def load_evaluation_set(tsv_path, encoding='latin-1'):
    df = pd.read_csv(tsv_path, sep='\t', encoding=encoding)
    sentences = df['sentence'].apply(lambda x: x.lower().strip()).tolist()
    labels = df['evaluation_pair_interacting'].values
    print(f"External evaluation: {len(sentences)} samples")
    print(f"  Label 0: {np.sum(labels == 0)}, Label 1: {np.sum(labels == 1)}")
    return sentences, labels, df


def load_interaction_dict(csv_path):
    df = pd.read_csv(csv_path)
    interactions = set(df['interaction'].str.lower().str.strip().tolist())
    print(f"Loaded {len(interactions)} interaction terms")
    return interactions


# =============================================================================
# FEATURE BUILDING
# =============================================================================

def build_features(train_sentences, eval_sentences, interaction_set):
    print("\nBuilding features...")
    
    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    
    X_train_tfidf = tfidf.fit_transform(train_sentences)
    X_eval_tfidf = tfidf.transform(eval_sentences)
    
    interaction_ext = InteractionFeatureExtractor(interaction_set)
    linguistic_ext = LinguisticFeatureExtractor()
    
    X_train_int = interaction_ext.transform(train_sentences)
    X_eval_int = interaction_ext.transform(eval_sentences)
    X_train_ling = linguistic_ext.transform(train_sentences)
    X_eval_ling = linguistic_ext.transform(eval_sentences)
    
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_int), csr_matrix(X_train_ling)])
    X_eval = hstack([X_eval_tfidf, csr_matrix(X_eval_int), csr_matrix(X_eval_ling)])
    
    print(f"  Total features: {X_train.shape[1]}")
    return X_train, X_eval, tfidf, interaction_ext, linguistic_ext


# =============================================================================
# HIGH PRECISION MODELS
# =============================================================================

def get_high_precision_models():
    """
    Models configured for HIGH PRECISION.
    Key strategies:
    - Higher C values (less regularization, stricter decision boundaries)
    - Class weights penalizing false positives more
    - Conservative predictions
    """
    
    # Custom class weights: penalize false positives more than false negatives
    # {0: 1.0, 1: 0.5} means we care more about not misclassifying negatives as positives
    precision_weights = {0: 1.0, 1: 0.3}  # Heavy penalty for false positives
    
    models = {
        'SVM Linear (High Precision)': SVC(
            kernel='linear', C=0.1,  # Lower C = wider margin = more conservative
            class_weight=precision_weights,
            probability=True, random_state=42
        ),
        'SVM RBF (High Precision)': SVC(
            kernel='rbf', C=0.1, gamma='scale',
            class_weight=precision_weights,
            probability=True, random_state=42
        ),
        'Logistic Regression (High Precision)': LogisticRegression(
            C=0.1,  # More regularization
            class_weight=precision_weights,
            max_iter=1000, random_state=42
        ),
        'Random Forest (High Precision)': RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5,  # More conservative splits
            class_weight=precision_weights,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting (High Precision)': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05,  # Slower learning
            max_depth=3, min_samples_leaf=10,
            random_state=42
        ),
    }
    return models


# =============================================================================
# THRESHOLD OPTIMIZATION FOR PRECISION
# =============================================================================

def find_optimal_threshold(y_true, y_proba, target_precision=0.8):
    """
    Find the classification threshold that achieves target precision.
    Higher threshold = higher precision, lower recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold that gives us at least target_precision
    valid_idx = np.where(precisions >= target_precision)[0]
    
    if len(valid_idx) == 0:
        # Can't achieve target precision, return highest precision threshold
        best_idx = np.argmax(precisions[:-1])  # Exclude last element (always 1.0)
        return thresholds[best_idx], precisions[best_idx], recalls[best_idx]
    
    # Among valid thresholds, pick the one with highest recall
    best_idx = valid_idx[np.argmax(recalls[valid_idx])]
    
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx]


def predict_with_threshold(model, X, threshold=0.5):
    """Predict using custom threshold"""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int), proba
    else:
        return model.predict(X), None


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_with_threshold_tuning(X_train, y_train, X_test, y_test, models, 
                                    target_precision=0.7):
    """
    Train models and tune threshold for target precision.
    Uses a validation split from training data to find optimal threshold.
    """
    # Split training data for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    results = []
    trained_models = {}
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH THRESHOLD OPTIMIZATION (target precision: {target_precision})")
    print(f"{'='*70}")
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train on training subset
        model.fit(X_tr, y_tr)
        
        # Find optimal threshold on validation set
        if hasattr(model, 'predict_proba'):
            val_proba = model.predict_proba(X_val)[:, 1]
            opt_threshold, val_prec, val_rec = find_optimal_threshold(
                y_val, val_proba, target_precision
            )
            print(f"  Optimal threshold: {opt_threshold:.3f}")
            print(f"  Validation - Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
        else:
            opt_threshold = 0.5
            print("  Model doesn't support probability, using default threshold")
        
        # Retrain on full training data
        model.fit(X_train, y_train)
        trained_models[name] = {'model': model, 'threshold': opt_threshold}
        
        # Evaluate on test set with optimized threshold
        y_pred, y_proba = predict_with_threshold(model, X_test, opt_threshold)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n  TEST SET RESULTS (threshold={opt_threshold:.3f}):")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  PRECISION: {prec:.4f}  <-- optimized for this")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        results.append({
            'Model': name,
            'Threshold': opt_threshold,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'TP': cm[1,1], 'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0]
        })
    
    return pd.DataFrame(results), trained_models


def cross_validate_precision_focused(X, y, models, n_folds=5):
    """Cross-validation with precision as primary metric"""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    
    print(f"\n{'='*70}")
    print(f"5-FOLD CROSS-VALIDATION (Precision-Focused)")
    print(f"{'='*70}")
    
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'accuracy': 'accuracy'
    }
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        start = time.time()
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - start
        
        prec = cv_results['test_precision']
        rec = cv_results['test_recall']
        f1 = cv_results['test_f1']
        acc = cv_results['test_accuracy']
        
        print(f"  PRECISION: {prec.mean():.4f} (+/- {prec.std()*2:.4f})  <-- primary metric")
        print(f"  Recall:    {rec.mean():.4f} (+/- {rec.std()*2:.4f})")
        print(f"  F1 Score:  {f1.mean():.4f} (+/- {f1.std()*2:.4f})")
        print(f"  Accuracy:  {acc.mean():.4f} (+/- {acc.std()*2:.4f})")
        print(f"  Time: {elapsed:.2f}s")
        
        results.append({
            'Model': name,
            'CV_Precision': prec.mean(),
            'CV_Precision_Std': prec.std(),
            'CV_Recall': rec.mean(),
            'CV_F1': f1.mean(),
            'CV_Accuracy': acc.mean(),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("BIOTIC INTERACTION CLASSIFIER - HIGH PRECISION VERSION")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    train_sentences, train_labels = load_training_data('training_data_cleaned.csv')
    eval_sentences, eval_labels, _ = load_evaluation_set('eval_100.tsv')
    interaction_set = load_interaction_dict('interaction_dict.csv')
    
    # Build features
    print("\n[2] Building features...")
    X_train, X_eval, tfidf, int_ext, ling_ext = build_features(
        train_sentences, eval_sentences, interaction_set
    )
    y_train = np.array(train_labels)
    y_eval = np.array(eval_labels)
    
    # Cross-validation
    print("\n[3] Cross-validation...")
    models = get_high_precision_models()
    cv_results = cross_validate_precision_focused(X_train, y_train, models)
    
    # Evaluate with threshold tuning
    print("\n[4] Threshold-optimized evaluation on external set...")
    models = get_high_precision_models()  # Fresh models
    
    # Try different target precisions
    for target_prec in [0.6, 0.7, 0.8]:
        print(f"\n{'#'*70}")
        print(f"# TARGET PRECISION: {target_prec}")
        print(f"{'#'*70}")
        
        models_fresh = get_high_precision_models()
        ext_results, trained = evaluate_with_threshold_tuning(
            X_train, y_train, X_eval, y_eval, models_fresh, target_precision=target_prec
        )
        
        print(f"\nSummary for target precision {target_prec}:")
        print(ext_results[['Model', 'Threshold', 'Precision', 'Recall', 'F1']].to_string(index=False))
    
    # Save best model (highest precision with reasonable recall)
    print("\n[5] Saving results...")
    cv_results.to_csv('cv_results_high_precision.csv', index=False)
    
    # Final model with target precision 0.7
    models_final = get_high_precision_models()
    final_results, final_trained = evaluate_with_threshold_tuning(
        X_train, y_train, X_eval, y_eval, models_final, target_precision=0.7
    )
    final_results.to_csv('external_results_high_precision.csv', index=False)
    
    # Find best model by precision
    best_idx = final_results['Precision'].idxmax()
    best_name = final_results.loc[best_idx, 'Model']
    best_info = final_trained[best_name]
    
    print(f"\n  Best model: {best_name}")
    print(f"  Precision: {final_results.loc[best_idx, 'Precision']:.4f}")
    print(f"  Threshold: {best_info['threshold']:.3f}")
    
    with open('best_model_high_precision.pkl', 'wb') as f:
        pickle.dump({
            'model': best_info['model'],
            'threshold': best_info['threshold'],
            'model_name': best_name,
            'tfidf': tfidf,
            'interaction_extractor': int_ext,
            'linguistic_extractor': ling_ext,
        }, f)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()