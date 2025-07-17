# XSS Detection System using Machine Learning

A comprehensive Python program for detecting Cross-Site Scripting (XSS) attacks using machine learning models XGBoost and LightGBM with SMOTE for handling imbalanced data.

## 🎯 Overview

This project implements a robust XSS detection system that compares the performance of two powerful machine learning models:
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework optimized for speed and memory efficiency

The system uses SMOTE (Synthetic Minority Oversampling Technique) to handle imbalanced datasets and includes comprehensive feature engineering specifically designed for XSS detection.

## 📊 Dataset

The system is designed to work with the **Cross-Site Scripting (XSS) Dataset for Deep Learning** from Kaggle:
- **URL**: https://www.kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning
- **Format**: CSV with text samples and binary labels (XSS vs Normal)

## 🚀 Features

### Machine Learning Models
- **XGBoost Classifier**: High-performance gradient boosting
- **LightGBM Classifier**: Fast and memory-efficient gradient boosting
- **Random Forest**: Ensemble learning baseline
- **SMOTE Integration**: Handles imbalanced data effectively

### Feature Engineering
- **Manual Features**: XSS-specific patterns, HTML tags, special characters
- **Text Features**: TF-IDF vectorization with n-grams
- **Statistical Features**: Text length, word count, entropy
- **Security Features**: Suspicious keywords, URL patterns, IP addresses

### Comprehensive Analysis
- **Exploratory Data Analysis**: Data distribution, class balance, text statistics
- **Model Comparison**: Performance metrics across all models
- **Visualizations**: ROC curves, confusion matrices, feature importance
- **Hyperparameter Tuning**: Grid search optimization

## 📁 Project Structure

```
xss_ml/
├── requirements.txt          # Python dependencies
├── xss_detection.py         # Main XSS detection system
├── data_loader.py           # Data loading and preprocessing
├── run_xss_detection.py     # Complete workflow example
├── README.md                # This file
└── Generated Files:
    ├── xss_detection_models.pkl    # Trained models
    ├── class_distribution.png      # Data visualization
    ├── model_evaluation.png        # Model comparison
    ├── xss_data_analysis.png       # Data analysis plots
    ├── xss_wordclouds.png          # Word clouds
    └── xss_detection_report.txt    # Summary report
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd xss_ml
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (automatic on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🎮 Usage

### Quick Start

Run the complete XSS detection pipeline:
```bash
python run_xss_detection.py
```

### Using Your Own Dataset

1. **Download the Kaggle dataset**
2. **Update the data path** in `run_xss_detection.py`:
```python
df = data_loader.load_kaggle_dataset('path_to_your_dataset.csv')
```

### Individual Components

**Data Loading and Exploration**:
```python
from data_loader import XSSDataLoader

loader = XSSDataLoader()
df = loader.load_kaggle_dataset('dataset.csv')
loader.explore_dataset(df)
```

**XSS Detection**:
```python
from xss_detection import XSSDetector

detector = XSSDetector()
X, y = detector.preprocess_data(df)
results = detector.train_models(X, y)
detector.evaluate_models(results)
```

**Making Predictions**:
```python
# Load trained models
detector.load_models('xss_detection_models.pkl')

# Predict XSS
text = '<script>alert("XSS")</script>'
prediction = detector.predict(text, 'XGBoost')
print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

## 📈 Model Performance

The system evaluates models using multiple metrics:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

### Expected Performance
With the sample dataset, typical performance ranges:
- **XGBoost**: F1-Score ~0.95-0.98
- **LightGBM**: F1-Score ~0.94-0.97
- **Random Forest**: F1-Score ~0.92-0.95

## 🔧 Configuration

### Model Parameters

**XGBoost**:
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

**LightGBM**:
```python
lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

### Feature Engineering

**TF-IDF Parameters**:
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)
```

**SMOTE Parameters**:
```python
SMOTE(random_state=42)
```

## 📊 Visualizations

The system generates several visualizations:

1. **Class Distribution**: Pie chart showing XSS vs Normal samples
2. **Text Statistics**: Length and word count distributions
3. **Model Comparison**: Performance metrics comparison
4. **ROC Curves**: Model performance visualization
5. **Confusion Matrix**: Classification accuracy breakdown
6. **Feature Importance**: Most important features for detection
7. **Word Clouds**: Visual representation of XSS and normal text

## 🧪 Testing

### Sample XSS Patterns Detected
- `<script>alert("XSS")</script>`
- `javascript:alert(1)`
- `<img src=x onerror=alert(1)>`
- `<iframe src="javascript:alert(1)"></iframe>`
- `eval(String.fromCharCode(...))`
- `document.cookie`
- `window.location`

### Interactive Testing Mode
The system includes an interactive mode for real-time testing:
```bash
python run_xss_detection.py
# Choose 'y' when prompted for interactive mode
```

## 🔍 Feature Details

### Manual Features (40+ features)
- **XSS Patterns**: Script tags, JavaScript protocols, event handlers
- **HTML Elements**: Dangerous tags like iframe, object, embed
- **Special Characters**: Angle brackets, quotes, parentheses
- **Suspicious Keywords**: alert, eval, document, window
- **Statistical**: Text entropy, URL count, IP addresses

### TF-IDF Features (5000 features)
- **N-grams**: 1-gram, 2-gram, 3-gram combinations
- **Stop Words**: Removed for better signal
- **Frequency Filtering**: Min/max document frequency thresholds

## 🚨 Security Considerations

1. **False Positives**: May flag legitimate code snippets
2. **Evasion**: Sophisticated attacks may bypass detection
3. **Updates**: Regular retraining needed for new attack patterns
4. **Performance**: Real-time deployment considerations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Kaggle Dataset**: Cross-Site Scripting (XSS) Dataset for Deep Learning
- **XGBoost**: Extreme Gradient Boosting framework
- **LightGBM**: Light Gradient Boosting Machine
- **scikit-learn**: Machine learning library
- **SMOTE**: Synthetic Minority Oversampling Technique

## 📞 Support

For questions or issues:
1. Check the generated `xss_detection_report.txt` for detailed results
2. Review the visualization files for insights
3. Examine the console output for debugging information

## 🔄 Future Enhancements

- Deep learning models (LSTM, BERT)
- Real-time web API deployment
- Advanced feature engineering
- Ensemble methods
- Multi-language support
- Performance optimization 
