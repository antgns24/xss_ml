#!/usr/bin/env python3
"""
Installation Test Script for XSS Detection System
This script verifies that all dependencies are installed and the system works correctly.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
        'imblearn', 'matplotlib', 'seaborn', 'nltk', 'bs4', 
        'wordcloud', 'tqdm', 'joblib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_basic_functionality():
    """Test basic functionality of the XSS detection system"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test data loader
        from data_loader import XSSDataLoader
        loader = XSSDataLoader()
        df = loader.create_sample_dataset()
        print(f"✅ Data loader: Created dataset with {len(df)} samples")
        
        # Test XSS detector initialization
        from xss_detection import XSSDetector
        detector = XSSDetector()
        print("✅ XSS detector: Initialized successfully")
        
        # Test feature extraction
        test_text = '<script>alert("test")</script>'
        features = detector.extract_features(test_text)
        print(f"✅ Feature extraction: Generated {len(features)} features")
        
        # Test preprocessing
        X, y = detector.preprocess_data(df)
        print(f"✅ Data preprocessing: Generated features shape {X.shape}")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        return False

def test_model_training():
    """Test model training with minimal data"""
    print("\n🎯 Testing model training...")
    
    try:
        from xss_detection import XSSDetector
        from data_loader import XSSDataLoader
        
        # Create minimal dataset
        loader = XSSDataLoader()
        df = loader.create_sample_dataset()
        
        # Initialize detector
        detector = XSSDetector()
        
        # Preprocess data
        X, y = detector.preprocess_data(df)
        
        # Train models (with reduced parameters for speed)
        print("Training models with minimal configuration...")
        
        # Override model parameters for quick testing
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE
        from sklearn.metrics import accuracy_score, f1_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Test XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        xgb_model.fit(X_train_balanced, y_train_balanced)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"✅ XGBoost: Accuracy = {xgb_accuracy:.3f}")
        
        # Test LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=42, verbose=-1)
        lgb_model.fit(X_train_balanced, y_train_balanced)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        print(f"✅ LightGBM: Accuracy = {lgb_accuracy:.3f}")
        
        print("\n✅ Model training test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model training test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\n🔮 Testing prediction functionality...")
    
    try:
        from xss_detection import XSSDetector
        from data_loader import XSSDataLoader
        
        # Create and train a minimal model
        loader = XSSDataLoader()
        df = loader.create_sample_dataset()
        
        detector = XSSDetector()
        X, y = detector.preprocess_data(df)
        
        # Train with minimal configuration
        results = detector.train_models(X, y)
        
        # Test predictions
        test_samples = [
            '<script>alert("XSS")</script>',
            'Hello world, this is normal text'
        ]
        
        for sample in test_samples:
            prediction = detector.predict(sample, 'XGBoost')
            print(f"✅ Prediction for '{sample[:30]}...': {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
        
        print("\n✅ Prediction test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Prediction test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("XSS DETECTION SYSTEM - INSTALLATION TEST")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_tests_passed = False
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Test 3: Model training
    if not test_model_training():
        all_tests_passed = False
    
    # Test 4: Prediction
    if not test_prediction():
        all_tests_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your XSS detection system is ready to use!")
        print("\nNext steps:")
        print("1. Run 'python run_xss_detection.py' for the complete workflow")
        print("2. Replace sample data with your Kaggle dataset")
        print("3. Explore the generated visualizations and reports")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above and fix the issues.")
    print("="*60)

if __name__ == "__main__":
    main()