# Anti-Hater Filter for Social Networks

## Project Overview

This project implements a Deep Learning solution for the multi-label classification of toxic online comments. The goal is to identify and categorize various forms of toxicity (e.g., threats, obscenity, insults) to support automated content moderation systems.

The model is built from scratch using TensorFlow/Keras and features a Bidirectional LSTM (Bi-LSTM) architecture. A key focus of this project is the handling of severe class imbalance, ensuring the model can detect rare but critical classes like threats and hate speech.

## Objectives

- **Multi-Label Classification**: Classify comments into 6 non-mutually exclusive labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
- **Handle Imbalance**: Address the extreme scarcity of dangerous classes (e.g., `threat` represents < 0.3% of the dataset).
- **Optimize Metrics**: Maximize F1-Score and Recall on minority classes without collapsing Precision.

## Tech Stack

- **Language**: Python 3.x
- **Deep Learning Framework**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Text Processing**: Regular Expressions (Regex)
- **Visualization**: Matplotlib, Seaborn

## Dataset

The dataset consists of Wikipedia comments manually labeled for toxic behavior.

- **Total Samples**: ~160,000
- **Input**: Raw text comments.
- **Output**: 6 binary labels (0 or 1).
- **Challenge**: The dataset is highly unbalanced. The vast majority of comments are "clean", and among the toxic ones, `toxic` is dominant, while `threat` and `identity_hate` are extremely rare.

## Preprocessing Strategy: A Data-Centric Approach

We deliberately avoided aggressive text normalization to preserve semantic meaning and user intent.

### Text Cleaning:
- Removed IP addresses, URLs, and non-alphabetic characters.
- **Decision**: We did NOT remove Stop Words or apply Lemmatization.
- **Reasoning**: In toxicity detection, stopwords like "you" (target of attack) and "not" (negation) are crucial for context. Lemmatization was skipped to preserve the specific aggressive tone of certain word forms.

### Tokenization & Padding:
- **Vocabulary Size**: Top 20,000 most frequent words.
- **Sequence Length**: Fixed at 225 tokens (Padding applied `pre`).

## Model Architecture: Bidirectional LSTM

We utilized a Recurrent Neural Network (RNN) to capture sequential dependencies in the text.

| Layer | Specifications | Purpose |
|-------|---------------|---------|
| Embedding | Dim: 128 | Learns dense vector representations of words. |
| Bidirectional LSTM | Units: 64 | Processes text in both directions (Past <-> Future) to understand full context. |
| GlobalMaxPool1D | - | Extracts the most significant features (highest toxicity peaks) from the sequence. |
| Dense | Units: 64 | Fully connected layer for pattern combination. |
| Dropout | Rate: 0.2 | Regularization to prevent overfitting on the oversampled data. |
| Output | Units: 6, Act: Sigmoid | Independent probability (0-1) for each of the 6 classes. |

## Handling Class Imbalance

The primary challenge was the model ignoring rare classes (`threat`, `identity_hate`). We experimented with three strategies:

1. **Baseline**: Standard Cross-Entropy. Result: F1-Score = 0.00 on rare classes.
2. **Weighted Loss**: Applied calculated class weights (inverse frequency). Result: High Recall but extremely low Precision (Model became "paranoid").
3. **Targeted Random Oversampling (Final Choice)**:
   - We manually oversampled only the minority classes in the Training set.
   - `threat`: x50 duplication.
   - `identity_hate`: x15 duplication.
   - **Outcome**: This allowed the model to learn the actual linguistic patterns of threats without skewing the loss function artificially, resulting in the best balance between Precision and Recall.

## Results

The final model achieves robust performance across all classes.

- **Weighted ROC-AUC**: > 0.98
- **Test Accuracy**: ~99% (High accuracy is expected due to the "clean" majority).
- **Rare Class Performance**: Successfully unlocked detection capabilities for `threat` and `identity_hate`, achieving acceptable F1-Scores where previous iterations failed.

## Visualizations Included

The project notebook includes:
- **Learning Curves**: To monitor overfitting/underfitting.
- **Multi-Label Confusion Matrices**: 6 separate 2x2 matrices to analyze TP/FP/TN/FN for each label.
- **ROC Curves**: Per-class curves showing the trade-off between TPR and FPR.

## How to Run

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter Notebook `Toxic_Comment_Classification.ipynb`.

The notebook handles data loading, preprocessing, oversampling, training, and evaluation automatically.

## Future Improvements

- **Threshold Optimization**: Implement dynamic thresholding (moving the decision boundary from 0.5 to an optimized value per class) to further maximize F1-Score.
- **Transformer Models**: Fine-tune BERT or RoBERTa to better capture sarcasm and subtle toxicity.
- **Advanced Augmentation**: Use Back-Translation or Synonym Replacement instead of simple duplication for oversampling.
