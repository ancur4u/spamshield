# 🛡️ SpamShield — Cost-Aware Spam Triage for an Enterprise Mail Gateway

**M.Tech (AIML) · Machine Learning · Assignment 2** · Classification models + Streamlit deployment

---

## a. Problem statement

An inbound mail gateway must decide, for every message, whether to **deliver**, **quarantine** or **block** it. The two errors are not equally expensive: blocking a legitimate customer invoice (false positive) hurts far more than letting one spam message reach an inbox (false negative). Accuracy alone therefore hides what matters operationally.

This project:

1. Trains and compares **five classifiers** (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on the same spam/legitimate-mail dataset using six metrics.
2. Wraps the frozen models in a **Streamlit console** that goes beyond metric display: it converts model probabilities into a **three-action gateway policy** (deliver / quarantine / block) by minimising expected cost, and it can **score a raw pasted email** and explain the verdict.

The central question: *which model, at which thresholds, gives the cheapest and safest filtering policy?*

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — **Spambase** (Hopkins, Reeber, Forman, Suermondt; Hewlett-Packard Labs, 1999). Loaded here from the PMLB mirror of the same data (`data/spambase.tsv.gz`). |
| Task | Binary classification — `is_spam` (1 = spam, 0 = legitimate) |
| Instances | **4,601** emails (1,813 spam = 39.4 %, 2,788 legitimate) |
| Features | **57** numeric (assignment minimum: 12) |
| Missing values | None |
| Split | Stratified 80 / 20 → 3,680 train, **921 test** (`test_data.csv`), `random_state=42` |

Feature groups: **48** `word_freq_*` (100 × occurrences of a word ÷ total words), **6** `char_freq_*` for `; ( [ ! $ #`, and **3** capital-letter run statistics (`average`, `longest`, `total`).

**Pre-processing:** `log1p` to damp heavy-tailed frequencies, then `StandardScaler` for Logistic Regression, KNN and SVM; `log1p` only for Gaussian Naive Bayes; raw values for the Decision Tree. Everything lives inside a scikit-learn `Pipeline`, so the saved model files contain their own pre-processing. Hyper-parameters were tuned with 5-fold stratified `GridSearchCV` (scoring = F1) on the training split only.

## c. GitHub repository link

**https://github.com/<your-username>/<your-repo-name>**   ← *replace with your repo URL before submitting*

```
spamshield/
├── app.py                    # Streamlit application (6 pages)
├── train_models.py           # trains + saves all 5 models, metrics, test_data.csv
├── spamshield_features.py    # feature schema + raw-email feature extractor
├── requirements.txt
├── README.md
├── test_data.csv             # 921-email hold-out set (57 features + is_spam)
├── data/
│   ├── spambase.tsv.gz       # source data
│   └── spambase_full.csv     # same data with named columns
└── model/
    ├── logistic_regression.joblib   decision_tree.joblib   knn.joblib
    ├── naive_bayes.joblib           support_vector_machine.joblib
    ├── metrics.json                 # stored metrics, CV scores, ROC points
    └── meta.json                    # feature importances, class means
```

Re-create everything: `pip install -r requirements.txt && python train_models.py && streamlit run app.py`

## d. Models used

All five models were trained on the **same** training split and evaluated on the **same** 921-email hold-out set at a 0.5 decision threshold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9359 | 0.9769 | 0.9222 | 0.9146 | 0.9184 | 0.8657 |
| Decision Tree | 0.9066 | 0.9469 | 0.8795 | 0.8843 | 0.8819 | 0.8047 |
| KNN | 0.9338 | 0.9777 | 0.9148 | **0.9174** | 0.9161 | 0.8614 |
| Naive Bayes (Gaussian) | 0.8686 | 0.9473 | 0.7867 | 0.9146 | 0.8459 | 0.7388 |
| Support Vector Machine (RBF) | **0.9370** | **0.9810** | **0.9320** | 0.9063 | **0.9190** | **0.8677** |

Tuned settings (5-fold CV F1 on training data in brackets): LR `C=1` (0.9243) · Decision Tree `max_depth=14, min_samples_leaf=5` (0.9030) · KNN `k=11, distance-weighted` (0.9252) · Naive Bayes `var_smoothing=0.1` (0.8488) · SVM `C=5, gamma='scale'` (0.9359).

Confusion matrices on the test set (`[[TN, FP], [FN, TP]]`): LR `[[530, 28], [31, 332]]` · DT `[[514, 44], [42, 321]]` · KNN `[[527, 31], [30, 333]]` · NB `[[468, 90], [31, 332]]` · SVM `[[534, 24], [34, 329]]`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for such a simple model (93.6 % accuracy, AUC 0.977) — after the `log1p` + scaling step, spam and ham are close to linearly separable in this feature space. Balanced errors (28 FP / 31 FN), fully interpretable coefficients, and the smallest model file (< 5 KB). |
| Decision Tree | Clearly the weakest of the discriminative models (AUC 0.947, 44 FP / 42 FN). The deepest useful tree (depth 14) is needed to reach this score, which signals high variance; leaf-proportion probabilities are also coarse, which limits how finely thresholds can be tuned. Its upside is human-readable rules. |
| KNN | Almost level with Logistic Regression (AUC 0.978) and has the best recall (0.917, fewest missed spam). The price is operational: the saved model is ~1.7 MB because it stores the training set, and every prediction scans it, so inference scales poorly with data size. |
| Naive Bayes | Lowest accuracy (86.9 %) and precision (0.787): it produces **90 false positives**, more than three times the SVM's. Word frequencies are strongly correlated and the independence assumption breaks, giving over-confident spam scores. Recall is respectable (0.915), but blocking this many legitimate emails makes it the least suitable for a gateway. It is, however, by far the fastest to train. |
| Support Vector Machine | Best on Accuracy, AUC, Precision, F1 and MCC, with the fewest false positives (24) and the best cross-validated F1 (0.936). The RBF kernel captures non-linear word-combination effects the linear model misses. Weak spot: slightly lower recall (0.906) and unreliable scores for inputs far outside the training range (very short emails with extreme word shares). |
| **Overall winner for this dataset?** | **Support Vector Machine** (highest MCC 0.8677, AUC 0.981, and fewest false positives, which matters most for a mail gateway). Honest caveat: the gap to Logistic Regression and KNN is only 1–3 emails on a 921-email test set, so those two are near-equivalent; SVM is preferred because AUC, precision and CV-F1 all consistently favour it. Decision Tree and Naive Bayes are clearly behind. |

## e. Streamlit application

The app **loads the saved `.joblib` pipelines and only runs inference — nothing is retrained on the uploaded data.** Upload a CSV in the sidebar (it must contain the 57 feature columns and `is_spam`; if nothing is uploaded, the bundled test set is used so the app is never empty).

| Page | What it does |
|---|---|
| **Mission Control** | Corpus facts, live leaderboard of all five models, pipeline overview. |
| **Model Evaluation** | *Model dropdown*, adjustable decision threshold, six metric cards, **confusion matrix**, **classification report**, ROC curve, score-distribution plot. |
| **Model Arena** | All models side by side: metric table, bar chart, zoomed ROC overlay, majority-vote ensemble, and a table of **contested emails** (where models disagree) to prioritise human review. |
| **Gateway Policy Simulator** | Enter the cost of a blocked legitimate mail, a delivered spam and a quarantine review, plus daily volume. The app searches a grid of *(deliver-below, block-above)* thresholds for the **cheapest three-action policy** and compares it with a plain 0.5 cut-off. |
| **Live Email Inspector** | Paste a raw email → the app derives the 57 Spambase features, scores it with all five models, shows a deliver/quarantine/block verdict, and explains it with a model-agnostic **occlusion test** (which words/characters pushed the score). Warns when an input is far outside the training range. |
| **Insights & Caveats** | Permutation feature importance, class-wise feature means, CV scores, and a frank list of limitations. |

**Example finding from the simulator** (SVM, default assumed costs ₹500 / ₹40 / ₹6, 50,000 mails/day, optimised on the test set): a plain 0.5 cut-off costs about ₹7.25 lakh/day, whereas the best two-threshold policy (deliver < 0.16, block ≥ 0.98) costs about ₹1.29 lakh/day (−82 %) by blocking almost nothing legitimate and sending the uncertain middle band to quarantine. The cost figures are illustrative assumptions, and thresholds tuned and evaluated on the same test set are optimistic.

## f. Limitations

- Spambase is from 1999 and one HP Labs mailbox: top "legitimate" signals (`george`, `hp`, `edu`, `650`) reflect that inbox, so a real deployment would retrain on the organisation's own mail.
- Only bag-of-words and punctuation statistics are used — no headers, SPF/DKIM, URLs or attachments — and modern, polite phishing is under-represented.
- The raw-email extractor in the Live Inspector is a faithful but approximate re-implementation of the original feature extraction.

## g. Run & deploy

```bash
pip install -r requirements.txt
python train_models.py        # optional: regenerates models/, metrics, test_data.csv
streamlit run app.py
```

Deploy: push to GitHub → https://streamlit.io/cloud → *New app* → pick repo, branch `main`, file `app.py` → *Deploy*.
> Saved `.joblib` files depend on the scikit-learn version. `requirements.txt` pins the versions used to train them; if you retrain in another environment, update `requirements.txt` with `pip freeze` versions from that environment.
