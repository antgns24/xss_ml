#!/usr/bin/env python3
"""
Comprehensive XSS Detection System using Machine Learning
Author: AI Assistant
Description: XSS detection using XGBoost and LightGBM with SMOTE for imbalanced data handling
Dataset: Cross-Site Scripting (XSS) Dataset for Deep Learning from Kaggle
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import joblib

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

class XSSDetector:
    """
    Comprehensive XSS Detection System using Machine Learning
    """
    
    def __init__(self):
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load XSS dataset from CSV file
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def explore_data(self, df: pd.DataFrame) -> None:
        """
        Perform exploratory data analysis
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        
        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Class distribution
        if 'Label' in df.columns:
            print(f"\nClass distribution:")
            print(df['Label'].value_counts())
            print(f"\nClass distribution (percentage):")
            print(df['Label'].value_counts(normalize=True) * 100)
            
            # Visualize class distribution
            plt.figure(figsize=(10, 6))
            df['Label'].value_counts().plot(kind='bar')
            plt.title('Class Distribution')
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Sample data
        print(f"\nFirst 5 rows:")
        print(df.head())
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract comprehensive features from text
        """
        features = {}
        
        # Basic features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        
        # XSS-specific patterns
        xss_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>',
            r'<applet.*?>',
            r'<meta.*?>',
            r'<link.*?>',
            r'<img.*?>',
            r'<svg.*?>',
            r'eval\(',
            r'alert\(',
            r'confirm\(',
            r'prompt\(',
            r'document\.',
            r'window\.',
            r'location\.',
            r'cookie',
            r'innerHTML',
            r'outerHTML',
            r'fromCharCode',
            r'String\.fromCharCode',
            r'unescape',
            r'decodeURI',
            r'setTimeout',
            r'setInterval'
        ]
        
        for i, pattern in enumerate(xss_patterns):
            features[f'pattern_{i}'] = len(re.findall(pattern, text, re.IGNORECASE))
        
        # HTML tags count
        soup = BeautifulSoup(text, 'html.parser')
        features['html_tags'] = len(soup.find_all())
        
        # Special characters
        special_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}', '[', ']']
        for char in special_chars:
            features[f'char_{char}'] = text.count(char)
        
        # Entropy calculation
        def calculate_entropy(s):
            prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
            entropy = -sum([p * np.log2(p) for p in prob])
            return entropy
        
        features['entropy'] = calculate_entropy(text) if text else 0
        
        # URL-like patterns
        features['url_count'] = len(re.findall(r'https?://\S+', text))
        features['ip_count'] = len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text))
        
        # Suspicious keywords
        suspicious_keywords = [
            'script', 'alert', 'confirm', 'prompt', 'eval', 'javascript',
            'vbscript', 'onload', 'onerror', 'onclick', 'onmouseover',
            'document', 'window', 'cookie', 'location', 'href'
        ]
        
        for keyword in suspicious_keywords:
            features[f'keyword_{keyword}'] = text.lower().count(keyword)
        
        return features
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data and extract features
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Assuming the dataset has columns like 'Sentence' and 'Label'
        # Adjust column names based on actual dataset structure
        text_column = 'Sentence' if 'Sentence' in df.columns else df.columns[0]
        label_column = 'Label' if 'Label' in df.columns else df.columns[1]
        
        X_text = df[text_column].astype(str)
        y = df[label_column]
        
        # Extract manual features
        print("Extracting manual features...")
        manual_features = []
        for text in tqdm(X_text, desc="Processing texts"):
            features = self.extract_features(text)
            manual_features.append(features)
        
        manual_features_df = pd.DataFrame(manual_features)
        self.feature_names = list(manual_features_df.columns)
        
        # TF-IDF features
        print("Extracting TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_features = self.vectorizer.fit_transform(X_text)
        
        # Combine features
        manual_features_scaled = self.scaler.fit_transform(manual_features_df)
        tfidf_features_dense = tfidf_features.toarray()
        X_combined = np.hstack([manual_features_scaled, tfidf_features_dense])
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Final feature shape: {X_combined.shape}")
        print(f"Label distribution: {np.bincount(y_encoded)}")
        
        return X_combined, y_encoded
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train multiple models with SMOTE for handling imbalanced data
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE to handle imbalanced data
        print("Applying SMOTE for handling imbalanced data...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Original training set distribution: {np.bincount(y_train)}")
        print(f"Balanced training set distribution: {np.bincount(y_train_balanced)}")
        
        # Define models
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # ROC AUC for binary classification
            if len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self, results: Dict) -> None:
        """
        Comprehensive model evaluation with visualizations
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Performance comparison
        metrics_df = pd.DataFrame({
            name: {
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC AUC': result['roc_auc']
            }
            for name, result in results.items()
        }).T
        
        print("Performance Comparison:")
        print(metrics_df)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance metrics comparison
        metrics_df.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. ROC Curves (for binary classification)
        if len(np.unique(list(results.values())[0]['y_test'])) == 2:
            for name, result in results.items():
                fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
            
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curves')
            axes[0, 1].legend()
        
        # 3. Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_result = results[best_model_name]
        
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Feature importance for best model
        if hasattr(best_result['model'], 'feature_importances_'):
            feature_importance = best_result['model'].feature_importances_
            top_features_idx = np.argsort(feature_importance)[-20:]
            
            axes[1, 1].barh(range(len(top_features_idx)), feature_importance[top_features_idx])
            axes[1, 1].set_yticks(range(len(top_features_idx)))
            axes[1, 1].set_yticklabels([f'Feature_{i}' for i in top_features_idx])
            axes[1, 1].set_title(f'Top 20 Feature Importances - {best_model_name}')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(best_result['y_test'], best_result['y_pred']))
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning for XGBoost and LightGBM
        """
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        tuned_models = {}
        
        # XGBoost hyperparameter tuning
        print("Tuning XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='f1_weighted', n_jobs=-1)
        xgb_grid.fit(X_train_balanced, y_train_balanced)
        
        tuned_models['XGBoost_Tuned'] = xgb_grid.best_estimator_
        print(f"Best XGBoost params: {xgb_grid.best_params_}")
        
        # LightGBM hyperparameter tuning
        print("Tuning LightGBM...")
        lgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='f1_weighted', n_jobs=-1)
        lgb_grid.fit(X_train_balanced, y_train_balanced)
        
        tuned_models['LightGBM_Tuned'] = lgb_grid.best_estimator_
        print(f"Best LightGBM params: {lgb_grid.best_params_}")
        
        # Evaluate tuned models
        for name, model in tuned_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"\n{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        return tuned_models
    
    def save_models(self, models: Dict, filename: str = 'xss_models.pkl') -> None:
        """
        Save trained models to disk
        """
        model_data = {
            'models': models,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filename)
        print(f"Models saved to {filename}")
    
    def load_models(self, filename: str = 'xss_models.pkl') -> None:
        """
        Load trained models from disk
        """
        model_data = joblib.load(filename)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        
        print(f"Models loaded from {filename}")
    
    def predict(self, text: str, model_name: str = 'XGBoost') -> Dict:
        """
        Predict if a given text contains XSS
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Extract features
        manual_features = self.extract_features(text)
        manual_features_df = pd.DataFrame([manual_features])
        
        # Ensure all feature columns are present
        for feature in self.feature_names:
            if feature not in manual_features_df.columns:
                manual_features_df[feature] = 0
        
        manual_features_df = manual_features_df[self.feature_names]
        
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([text])
        
        # Combine features
        manual_features_scaled = self.scaler.transform(manual_features_df)
        tfidf_features_dense = tfidf_features.toarray()
        X_combined = np.hstack([manual_features_scaled, tfidf_features_dense])
        
        # Make prediction
        model = self.models[model_name]['model']
        prediction = model.predict(X_combined)[0]
        probability = model.predict_proba(X_combined)[0]
        
        # Decode prediction
        prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'prediction': prediction_label,
            'probability': probability,
            'confidence': max(probability)
        }

def main():
    """
    Main function to run the XSS detection system
    """
    print("XSS Detection System using Machine Learning")
    print("=" * 60)
    
    # Initialize detector
    detector = XSSDetector()
    
    # For demonstration, create sample data if dataset is not available
    # In practice, you would load the actual dataset from Kaggle
    sample_data = {
        'Sentence': [
            '<script>alert("XSS")</script>',
            'Hello world',
            'javascript:alert(1)',
            'This is a normal sentence',
            '<img src=x onerror=alert(1)>',
            'Regular text content',
            '<iframe src="javascript:alert(1)"></iframe>',
            'Just some text',
            'document.cookie',
            'Normal content here'
        ],
        'Label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for XSS, 0 for normal
    }
    
    df = pd.DataFrame(sample_data)
    
    # If you have the actual dataset, uncomment the line below:
    # df = detector.load_data('path_to_your_dataset.csv')
    
    if df is not None:
        # Explore data
        detector.explore_data(df)
        
        # Preprocess data
        X, y = detector.preprocess_data(df)
        
        # Train models
        results = detector.train_models(X, y)
        
        # Evaluate models
        detector.evaluate_models(results)
        
        # Hyperparameter tuning (optional, uncomment if needed)
        # tuned_models = detector.hyperparameter_tuning(X, y)
        
        # Save models
        detector.save_models(results)
        
        # Test prediction
        test_samples = [
            '<script>alert("XSS attack")</script>',
            'This is a normal message',
            'javascript:void(0)',
            '<img src=x onerror=alert(1)>'
        ]
        
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS")
        print("="*50)
        
        for sample in test_samples:
            try:
                prediction = detector.predict(sample, 'XGBoost')
                print(f"\nText: {sample}")
                print(f"Prediction: {prediction['prediction']}")
                print(f"Confidence: {prediction['confidence']:.4f}")
            except Exception as e:
                print(f"Error predicting for '{sample}': {e}")

if __name__ == "__main__":
    main()