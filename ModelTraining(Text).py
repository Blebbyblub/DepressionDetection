import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DepressionSeverityPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_sequence_length = 200
        self.vocab_size = 10000
        self.embedding_dim = 100
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Clean text data
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Encode labels
        df['encoded_label'] = self.label_encoder.fit_transform(df['label'])
        
        print("Label distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove user mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            # Remove punctuation and extra spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""
    
    def prepare_text_data(self, texts):
        """Tokenize and pad text sequences"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        return padded_sequences
    
    def build_model(self, num_classes):
        """Build the deep learning model"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
            SpatialDropout1D(0.2),
            
            # Convolutional layers for feature extraction
            Conv1D(64, 5, activation='relu'),
            GlobalMaxPooling1D(),
            
            # LSTM layer for sequence understanding
            # LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, df, test_size=0.2, random_state=42, epochs=20, batch_size=32):
        """Train the model"""
        # Prepare features and labels
        X = self.prepare_text_data(df['cleaned_text'].values)
        y = df['encoded_label'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(num_classes)
        
        print("Model Architecture:")
        self.model.summary()
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def predict_severity(self, text):
        """Predict depression severity for new text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Clean and preprocess the text
        cleaned_text = self.clean_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
        
        # Make prediction
        prediction = self.model.predict(padded_sequence)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        # Get the label
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'severity': predicted_label,
            'confidence': float(confidence),
            'probabilities': {
                label: float(prob) for label, prob in 
                zip(self.label_encoder.classes_, prediction[0])
            }
        }
    
    def save_model(self, model_path='depression_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save the trained model and tokenizer"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        # For tokenizer, you might want to use pickle or joblib
        import pickle
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_path}")
        
        # Save label encoder
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("Label encoder saved to label_encoder.pkl")

def main():
    """Main function to run the depression severity prediction"""
    
    # Initialize the predictor
    predictor = DepressionSeverityPredictor()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = predictor.load_and_preprocess_data('Depression_Severity_Levels_Dataset.csv')
    
    # Train the model
    print("\nTraining the model...")
    history = predictor.train(df, epochs=25, batch_size=32)
    
    # Save the model
    predictor.save_model()
    
    # Test predictions with some examples
    print("\nTesting predictions...")
    test_texts = [
        "I feel so overwhelmed and anxious all the time, I can't sleep or focus on anything.",
        "I'm applying for jobs and looking forward to starting my new career.",
        "Everything feels hopeless and I don't know how to keep going anymore.",
        "I had a great day with friends and family, feeling very grateful.",
        "The constant fear and panic attacks are making it impossible to function normally."
    ]
    
    for text in test_texts:
        prediction = predictor.predict_severity(text)
        print(f"\nText: {text[:100]}...")
        print(f"Predicted Severity: {prediction['severity']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print("Probabilities:", prediction['probabilities'])

if __name__ == "__main__":
    main()