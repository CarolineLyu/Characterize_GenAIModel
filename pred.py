import pickle
import numpy as np
import pandas as pd
import re

with open("model_bundle.pkl", "rb") as f:
    BUNDLE = pickle.load(f)

TFIDF_MODELS = BUNDLE["tfidf_models"]              # dict[col] -> model dict
TEXT_COLS = BUNDLE["text_cols"]                    # list of text column names

ORD_COLS = BUNDLE["ord_cols"]                      # list of ordinal column names
ORD_MAPPINGS = BUNDLE["ord_mappings"]              # dict[col] -> {category: index}
ORD_FILL_VALUES = BUNDLE["ord_fill_values"]        # dict[col] -> fill index

CAT_COLS = BUNDLE["cat_cols"]                      # list of multi-select categorical columns
CAT_MULTI_SELECT_CHOICES = BUNDLE["cat_multi_select_choices"]

UNIQUE_CLASSES = np.array(BUNDLE["unique_classes"])  # np.array of original string labels
NUM_CLASSES = int(BUNDLE["num_classes"])

ENSEMBLE_STATES = BUNDLE["ensemble_states"]        # list of parameter dicts for each model


# tfidf preprocessing

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at",
    "with", "is", "it", "this", "that", "as", "by", "be", "are", "was",
    "were", "from", "but", "if", "so", "not", "isn", "t"
}

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text, lower=True, remove_stopwords=True, stopwords=STOPWORDS):
    if not isinstance(text, str):
        return []
    if lower:
        text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def generate_ngrams(tokens, ngram_range=(1, 1)):
    min_n, max_n = ngram_range
    L = len(tokens)
    all_ngrams = []
    for n in range(min_n, max_n + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            all_ngrams.append(" ".join(tokens[i:i + n]))
    return all_ngrams


def transform_tfidf_column(text_series, model_cfg):
    vocab = model_cfg["vocab"]
    idf = model_cfg["idf"]
    ngram_range = model_cfg.get("ngram_range", (1, 1))
    remove_stopwords = model_cfg.get("remove_stopwords", True)

    N = len(text_series)
    V = len(vocab)
    X = np.zeros((N, V), dtype=np.float32)

    for i, raw in enumerate(text_series):
        tokens = tokenize(raw, lower=True, remove_stopwords=remove_stopwords)
        terms = generate_ngrams(tokens, ngram_range)
        if not terms:
            continue
        counts = {}
        for t in terms:
            if t in vocab:
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            continue
        doc_len = float(sum(counts.values()))
        for term, cnt in counts.items():
            j = vocab[term]
            tf = cnt / doc_len
            X[i, j] = tf * idf[j]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms

    return X


def transform_all_text(df):
    X_blocks = []
    for col in TEXT_COLS:
        model_cfg = TFIDF_MODELS[col]
        # Handle missing columns gracefully
        if col in df.columns:
            col_values = df[col].fillna("").astype(str)
        else:
            # If column not present, treat as empty strings
            col_values = pd.Series([""] * len(df), index=df.index)
        X_col = transform_tfidf_column(col_values, model_cfg)
        X_blocks.append(X_col)
    return np.hstack(X_blocks) if X_blocks else np.zeros((len(df), 0), dtype=np.float32)


# ordinal preprocessing

def transform_ordinal(df):
    """
    Transform ordinal columns using saved mappings and fill values.
    """
    N = len(df)
    K = len(ORD_COLS)
    X_ord = np.zeros((N, K), dtype=np.float32)

    for j, col in enumerate(ORD_COLS):
        mapping = ORD_MAPPINGS[col]
        fill_val = ORD_FILL_VALUES[col]

        col_vals = df[col] if col in df.columns else pd.Series([np.nan] * N, index=df.index)
        encoded_col = []
        for v in col_vals.tolist():
            if pd.isna(v) or v not in mapping:
                encoded_col.append(fill_val)
            else:
                encoded_col.append(mapping[v])

        X_ord[:, j] = np.array(encoded_col, dtype=np.float32)

    return X_ord


# categorical preprocessing

def parse_multiselect(raw_value, choice_list):
    if pd.isna(raw_value):
        return []
    txt = str(raw_value).strip()
    if txt == "":
        return []
    selections = []
    for choice in choice_list:
        if choice in txt:
            selections.append(choice)
    return selections


def transform_multiselect_categorical(df):
    N = len(df)
    M = len(CAT_MULTI_SELECT_CHOICES)
    total_features = len(CAT_COLS) * M
    X = np.zeros((N, total_features), dtype=np.float32)

    for i in range(N):
        offset = 0
        for col in CAT_COLS:
            if col in df.columns:
                selections = parse_multiselect(df.iloc[i][col], CAT_MULTI_SELECT_CHOICES)
            else:
                selections = []
            for j, choice in enumerate(CAT_MULTI_SELECT_CHOICES):
                X[i, offset + j] = 1.0 if choice in selections else 0.0
            offset += M

    return X

# nn forward pass for inference

def forward_one_model(X, state, eps=1e-5):
    """
        X -> fc1 -> batchnorm (running stats) -> ReLU -> fc2 -> logits
    """
    W1 = state["fc1_W"]
    b1 = state["fc1_b"]
    gamma = state["bn1_gamma"]
    beta = state["bn1_beta"]
    running_mean = state["bn1_running_mean"]
    running_var = state["bn1_running_var"]
    W2 = state["fc2_W"]
    b2 = state["fc2_b"]

    X = X.astype(np.float32, copy=False)

    z1 = X @ W1 + b1  # shape: (N, hidden_units)

    x_norm = (z1 - running_mean) / np.sqrt(running_var + eps)
    bn_out = gamma * x_norm + beta

    h1 = np.maximum(0.0, bn_out)

    logits = h1 @ W2 + b2  # shape: (N, num_classes)

    return logits


def ensemble_logits(X):
    """
    Compute averaged logits over the ensemble.
    """
    all_logits = []
    for state in ENSEMBLE_STATES:
        logits = forward_one_model(X, state)
        all_logits.append(logits)
    # Average over models
    avg_logits = np.mean(all_logits, axis=0)
    return avg_logits


# prepare data

RENAMED_COLUMNS_WITH_TARGET = [
    "id",
    "best_tasks_free",
    "acad_tasks_rating",
    "best_tasks_select",
    "subopt_freq_rating",
    "subopt_tasks_select",
    "subopt_tasks_free",
    "evidence_freq_rating",
    "verify_freq_rating",
    "verify_method_free",
    "target",
]

RENAMED_COLUMNS_NO_TARGET = RENAMED_COLUMNS_WITH_TARGET[:-1]

# assumes data is in same format as the training data. 
# Note* markus test seems to fail if target is dropped, this extra logic is to handels cases if there are 10 or 11 columns
# so we can also test with the training data
def _prepare_columns(df):
    df = df.copy()
    n_cols = df.shape[1]

    if n_cols == len(RENAMED_COLUMNS_WITH_TARGET):
        # 11-column file
        df.columns = RENAMED_COLUMNS_WITH_TARGET
        df = df.drop(columns=["target", "label"], errors="ignore")
    elif n_cols == len(RENAMED_COLUMNS_NO_TARGET):
        # 10-column test file
        df.columns = RENAMED_COLUMNS_NO_TARGET
    else:
        # fallback
        df = df.drop(columns=["target", "label"], errors="ignore")

    return df


def predict_all(csv_path):
    """
    Takes a CSV file path and returns predictions
    """
    # prepare data
    df = pd.read_csv(csv_path)
    df = _prepare_columns(df)

    # preprocessing
    X_text = transform_all_text(df)
    X_ord = transform_ordinal(df)
    X_cat = transform_multiselect_categorical(df)
    X_all = np.hstack([X_text, X_ord, X_cat]).astype(np.float32)

    # prediction
    logits = ensemble_logits(X_all)
    pred_indices = np.argmax(logits, axis=1)

    # mapping to original model labels
    preds = UNIQUE_CLASSES[pred_indices]

    return preds


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pred.py <input_csv>")
    else:
        input_csv = sys.argv[1]
        predictions = predict_all(input_csv)
        for p in predictions:
            print(p)