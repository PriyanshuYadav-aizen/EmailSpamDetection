# Model Card: Email Spam Ensemble

## Model Details
- **Type**: Ensemble Classifier (Rule Engine + TF-IDF/LogReg + Bi-LSTM)
- **Version**: 2.0.0
- **Task**: Binary Classification (SPAM vs. HAM)
- **License**: MIT

## Architecture
The model uses a weighted ensemble to combine three distinct signals:
1. **Rule Engine (50%)**: Keyword-based heuristics and URL detection.
2. **TF-IDF + Logistic Regression (25%)**: Classical feature-based classification.
3. **Bidirectional LSTM (25%)**: Sequence-aware deep learning model.

## Training Data
- **Primary Source**: Enron Email Dataset.
- **Augmentation**: Synthetic phishing templates focused on modern banking and "urgent" lures.

## Performance
- **Accuracy**: High (>95% on test set)
- **Strengths**: Professional email tone, classic spam keywords.
- **Weaknesses**: Highly novel phishing templates may require Rule Engine overrides.

## Ethical Considerations
- **Privacy**: The model is trained on public datasets. No private user data was used.
- **Bias**: May be biased toward the corporate tone typical of the Enron era.

## Usage
Intended for use in email clients or server-side filters to flag suspicious content for end-users.
