"""
train_xgboost.py
----------------
Radiomics ozellikleri + XGBoost ile 4-sinif subtype siniflandirma.

Kullanim:
  python scripts/train_xgboost.py `
    --radiomics_csv "C:/Users/PC/Desktop/data/radiomics_features.csv" `
    --output_dir models/xgboost/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radiomics_csv", required=True)
    p.add_argument("--output_dir", default="models/xgboost/")
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.radiomics_csv)
    feat_cols = [c for c in df.columns if c.startswith("original_")]
    print(f"Toplam hasta: {len(df)}")
    print(f"Ozellik sayisi: {len(feat_cols)}")

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    X_train = train_df[feat_cols].values
    y_train = train_df["label"].map(LABEL2IDX).values
    X_val   = val_df[feat_cols].values
    y_val   = val_df["label"].map(LABEL2IDX).values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["label"].map(LABEL2IDX).values

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # NaN imputation + scaling
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val   = scaler.transform(imputer.transform(X_val))
    X_test  = scaler.transform(imputer.transform(X_test))

    # Class weights
    counts = train_df["label"].value_counts()
    sample_weight = train_df["label"].map(
        lambda l: 1.0 / counts[l]).values

    # XGBoost
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        objective="multi:softprob",
        num_class=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("\nXGBoost egitiliyor...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Val sonucu
    val_probs = model.predict_proba(X_val)
    lb = label_binarize(y_val, classes=list(range(4)))
    val_auc = roc_auc_score(lb, val_probs, average="macro", multi_class="ovr")
    print(f"\nVal Macro AUC: {val_auc:.4f}")

    # Test sonucu
    test_probs = model.predict_proba(X_test)
    test_preds = test_probs.argmax(1)
    lb_test = label_binarize(y_test, classes=list(range(4)))
    test_auc = roc_auc_score(lb_test, test_probs, average="macro", multi_class="ovr")
    test_acc = (test_preds == y_test).mean()

    print(f"\n{'='*55}")
    print("TEST SONUCLARI")
    print(f"{'='*55}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Macro AUC: {test_auc:.4f}")

    idx2label = {v: k for k, v in LABEL2IDX.items()}
    pred_labels = [idx2label[p] for p in test_preds]
    true_labels = [idx2label[t] for t in y_test]
    print(f"\n{classification_report(true_labels, pred_labels, digits=3)}")

    print("Per-class AUC:")
    for i, lbl in enumerate(LABELS):
        try:
            auc_i = roc_auc_score(lb_test[:, i], test_probs[:, i])
            print(f"  {lbl}: {auc_i:.4f}")
        except Exception:
            print(f"  {lbl}: N/A")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    print(f"\nTop 10 ozellik:")
    print(importance.head(10).to_string(index=False))

    # Model kaydet
    model.save_model(str(out / "xgboost_model.json"))
    json.dump({
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
    }, open(out / "test_results.json", "w"), indent=2)

    print(f"\nModel: {out}/xgboost_model.json")
    print(f"Feature importance: {out}/feature_importance.csv")


if __name__ == "__main__":
    main()