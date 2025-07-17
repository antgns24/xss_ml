#!/usr/bin/env python3
"""
Complete XSS Detection System Example
This script demonstrates the full workflow of XSS detection using machine learning
"""

import os
import sys
import pandas as pd
import numpy as np
from xss_detection import XSSDetector
from data_loader import XSSDataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_complete_xss_detection():
    """
    Run the complete XSS detection pipeline
    """
    print("="*80)
    print("COMPREHENSIVE XSS DETECTION SYSTEM")
    print("Using XGBoost and LightGBM with SMOTE for Imbalanced Data")
    print("="*80)
    
    # Step 1: Load and explore data
    print("\n🔍 STEP 1: DATA LOADING AND EXPLORATION")
    print("-" * 50)
    
    data_loader = XSSDataLoader()
    
    # Try to load real dataset, fallback to sample data
    try:
        # Uncomment and modify the path below to use your actual Kaggle dataset
        # df = data_loader.load_kaggle_dataset('path_to_your_xss_dataset.csv')
        df = data_loader.create_sample_dataset()
        print("✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Explore the dataset
    data_loader.explore_dataset(df)
    
    # Step 2: Initialize XSS Detector
    print("\n🤖 STEP 2: INITIALIZING XSS DETECTOR")
    print("-" * 50)
    
    detector = XSSDetector()
    print("✅ XSS Detector initialized")
    
    # Step 3: Preprocess data
    print("\n⚙️ STEP 3: DATA PREPROCESSING")
    print("-" * 50)
    
    X, y = detector.preprocess_data(df)
    print("✅ Data preprocessing completed")
    
    # Step 4: Train models
    print("\n🎯 STEP 4: MODEL TRAINING")
    print("-" * 50)
    
    results = detector.train_models(X, y)
    print("✅ Model training completed")
    
    # Step 5: Evaluate models
    print("\n📊 STEP 5: MODEL EVALUATION")
    print("-" * 50)
    
    detector.evaluate_models(results)
    print("✅ Model evaluation completed")
    
    # Step 6: Compare model performance
    print("\n🏆 STEP 6: MODEL COMPARISON")
    print("-" * 50)
    
    # Create performance comparison table
    performance_data = []
    for model_name, result in results.items():
        performance_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'ROC AUC': f"{result['roc_auc']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    print("\nModel Performance Comparison:")
    print(performance_df.to_string(index=False))
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"\n🥇 Best performing model: {best_model}")
    print(f"   F1-Score: {results[best_model]['f1_score']:.4f}")
    
    # Step 7: Hyperparameter tuning (optional)
    print("\n🔧 STEP 7: HYPERPARAMETER TUNING (Optional)")
    print("-" * 50)
    
    user_input = input("Do you want to perform hyperparameter tuning? (y/n): ").lower()
    if user_input == 'y':
        tuned_models = detector.hyperparameter_tuning(X, y)
        print("✅ Hyperparameter tuning completed")
    else:
        print("⏭️ Skipping hyperparameter tuning")
    
    # Step 8: Save models
    print("\n💾 STEP 8: SAVING MODELS")
    print("-" * 50)
    
    detector.save_models(results, 'xss_detection_models.pkl')
    print("✅ Models saved successfully")
    
    # Step 9: Test predictions
    print("\n🧪 STEP 9: TESTING PREDICTIONS")
    print("-" * 50)
    
    test_samples = [
        # XSS examples
        '<script>alert("XSS attack!")</script>',
        'javascript:alert(document.cookie)',
        '<img src=x onerror=alert(1)>',
        '<iframe src="javascript:alert(1)"></iframe>',
        '<svg onload=alert(1)>',
        'eval(String.fromCharCode(97,108,101,114,116,40,49,41))',
        '<body onload=alert("XSS")>',
        'document.write("<script>alert(1)</script>")',
        
        # Normal examples
        'Welcome to our website!',
        'Please enter your username and password',
        'Thank you for your purchase',
        'Contact us at support@example.com',
        'Our services are available 24/7',
        'Free shipping on orders over $50',
        'Subscribe to our newsletter',
        'Follow us on social media'
    ]
    
    print("Testing predictions on sample data:")
    print("-" * 40)
    
    for i, sample in enumerate(test_samples, 1):
        try:
            # Test with XGBoost
            xgb_prediction = detector.predict(sample, 'XGBoost')
            
            # Test with LightGBM
            lgb_prediction = detector.predict(sample, 'LightGBM')
            
            print(f"\n{i}. Text: {sample[:50]}{'...' if len(sample) > 50 else ''}")
            print(f"   XGBoost: {xgb_prediction['prediction']} (confidence: {xgb_prediction['confidence']:.3f})")
            print(f"   LightGBM: {lgb_prediction['prediction']} (confidence: {lgb_prediction['confidence']:.3f})")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ Prediction testing completed")
    
    # Step 10: Generate summary report
    print("\n📋 STEP 10: SUMMARY REPORT")
    print("-" * 50)
    
    generate_summary_report(results, df, best_model)
    
    print("\n🎉 XSS DETECTION SYSTEM COMPLETED SUCCESSFULLY!")
    print("="*80)

def generate_summary_report(results, df, best_model):
    """
    Generate a comprehensive summary report
    """
    report = f"""
XSS DETECTION SYSTEM - SUMMARY REPORT
{'='*50}

DATASET INFORMATION:
- Total samples: {len(df)}
- XSS samples: {len(df[df['Label'] == 1])}
- Normal samples: {len(df[df['Label'] == 0])}
- Class balance: {len(df[df['Label'] == 1]) / len(df) * 100:.1f}% XSS, {len(df[df['Label'] == 0]) / len(df) * 100:.1f}% Normal

MODEL PERFORMANCE:
"""
    
    for model_name, result in results.items():
        report += f"""
{model_name}:
  - Accuracy: {result['accuracy']:.4f}
  - Precision: {result['precision']:.4f}
  - Recall: {result['recall']:.4f}
  - F1-Score: {result['f1_score']:.4f}
  - ROC AUC: {result['roc_auc']:.4f}
"""
    
    report += f"""
BEST MODEL: {best_model}
- Best F1-Score: {results[best_model]['f1_score']:.4f}

FEATURES USED:
- Manual feature engineering (XSS patterns, HTML tags, special characters, etc.)
- TF-IDF vectorization (up to 5000 features)
- SMOTE for handling imbalanced data

RECOMMENDATIONS:
1. The {best_model} model shows the best performance for XSS detection
2. Consider collecting more diverse XSS samples for better generalization
3. Implement real-time monitoring using the trained models
4. Regular model retraining with new XSS patterns is recommended
5. Consider ensemble methods for improved robustness

FILES GENERATED:
- xss_detection_models.pkl (trained models)
- class_distribution.png (data visualization)
- model_evaluation.png (model comparison)
- xss_data_analysis.png (data analysis)
- xss_wordclouds.png (word clouds)
"""
    
    print(report)
    
    # Save report to file
    with open('xss_detection_report.txt', 'w') as f:
        f.write(report)
    
    print("📄 Summary report saved to 'xss_detection_report.txt'")

def interactive_prediction_mode(detector):
    """
    Interactive mode for testing XSS predictions
    """
    print("\n🔮 INTERACTIVE PREDICTION MODE")
    print("-" * 50)
    print("Enter text to test for XSS (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        try:
            # Test with both models
            xgb_result = detector.predict(user_input, 'XGBoost')
            lgb_result = detector.predict(user_input, 'LightGBM')
            
            print(f"\nResults for: {user_input}")
            print(f"XGBoost  - Prediction: {xgb_result['prediction']}, Confidence: {xgb_result['confidence']:.3f}")
            print(f"LightGBM - Prediction: {lgb_result['prediction']}, Confidence: {lgb_result['confidence']:.3f}")
            
            # Provide interpretation
            if xgb_result['prediction'] == 1 or lgb_result['prediction'] == 1:
                print("⚠️  WARNING: Potential XSS detected!")
            else:
                print("✅ Text appears to be safe.")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """
    Main function with user interaction
    """
    try:
        # Run complete detection system
        run_complete_xss_detection()
        
        # Ask if user wants interactive mode
        print("\n" + "="*80)
        user_choice = input("Do you want to enter interactive prediction mode? (y/n): ").lower()
        
        if user_choice == 'y':
            # Load the saved models for interactive mode
            detector = XSSDetector()
            try:
                detector.load_models('xss_detection_models.pkl')
                interactive_prediction_mode(detector)
            except Exception as e:
                print(f"❌ Error loading models for interactive mode: {e}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()