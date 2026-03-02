# Characterize_GenAIModel
Multi-Class Text Classification: Identifying Generative AI Models from User Survey Responses
Classifying which generative AI tool (Claude, Gemini, or ChatGPT) a user interacts with, based on 800+ natural language survey responses combining free-text, Likert-scale, and multi-select features. Compares classical ML and deep learning approaches with a two-stage hyperparameter optimization pipeline.
Approach
Data: Mixed-type survey responses, including 3 free-text columns, 4 ordinal (Likert/frequency) ratings, and 2 multi-select categorical fields.
Preprocessing pipeline (scikit-learn):

Free-text features: Separate TF-IDF vectorizers per text column with independently tuned parameters (max_features, ngram_range, min_df, max_df), combined via FeatureUnion
Ordinal features: OrdinalEncoder with explicit category ordering for Likert and frequency scales
Categorical features: Custom MultiSelectBinarizerPerColumn transformer handling comma-separated multi-select responses with parenthesis-aware splitting and unicode normalization

Models:

XGBoost with bagging to reduce prediction variance
Feedforward neural network (PyTorch) with BatchNorm, dropout, and early stopping

Two-stage hyperparameter tuning:

TF-IDF parameters tuned via GridSearchCV with GroupKFold
Model architecture (hidden units, dropout, learning rate, batch size) tuned via Bayesian optimization with Weights & Biases, evaluated on 5-fold GroupKFold mean validation accuracy

Evaluation: GroupKFold cross-validation splitting by user ID to prevent data leakage from shared response patterns. Neural network outperformed XGBoost by 6.5%.
Repository Structure
- preprocess2.ipynb                           # Full pipeline: preprocessing, model training, hyperparameter tuning
- test_all_models.ipynb                       # Model comparison and evaluation across architectures
- pred.py                                     # Prediction script using trained pipeline artifacts
- training_data_clean.csv                     # Preprocessed survey response dataset
- report.pdf                                  # Detailed methodology and results report
- requirements.txt                            # Python dependencies

