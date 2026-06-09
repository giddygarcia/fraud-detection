# 🏦 Central Asia Fintech Fraud Detection Dataset

> **First publicly available fraud detection dataset for Central Asian mobile payments ecosystem**  
> Covers Uzbekistan's digital payment landscape — Click, Payme, Uzum Bank, and more.

---

## 📌 Overview

This dataset simulates the mobile payment transaction environment of Uzbekistan — one of Central Asia's fastest-growing fintech markets. It is designed for **fraud detection**, **anomaly detection**, **customer segmentation**, and **financial behavior analysis** tasks.

The dataset covers **2 years of transactions (2023–2024)**, reflecting real-world patterns: seasonal spending spikes, urban vs. rural usage differences, and fraud patterns specific to mobile-first markets.

---

## 📁 Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `transactions.csv` | 150,000 | 13 | Core transactions with fraud labels |
| `users.csv` | 12,000 | 13 | User demographics and financial profile |
| `merchants.csv` | 2,500 | 10 | Merchant registry with risk scores |

---

## 🗂️ Schema

### `transactions.csv`
| Column | Type | Description |
|--------|------|-------------|
| `tx_id` | string | Unique transaction ID |
| `timestamp` | datetime | Transaction datetime (UTC+5) |
| `user_id` | string | FK → users.csv |
| `merchant_id` | string | FK → merchants.csv |
| `amount_uzs` | int | Amount in Uzbek Som |
| `channel` | category | `app`, `web`, `pos`, `qr`, `ussd` |
| `hour_of_day` | int | 0–23 |
| `day_of_week` | int | 0=Mon, 6=Sun |
| `is_weekend` | bool | Weekend flag |
| `session_duration_sec` | int | Session length before tx |
| `login_attempts` | int | Login attempts in session |
| `is_cross_city_tx` | bool | User city ≠ merchant city |
| **`is_fraud`** | bool | **Target variable** — 1 = fraudulent |

### `users.csv`
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Unique user ID |
| `name` | string | Uzbek full name |
| `gender` | category | M / F |
| `age` | int | 18–72 |
| `city` | string | Home city (12 cities) |
| `region` | string | Administrative region |
| `registration_date` | date | Account opening date |
| `primary_app` | string | Preferred payment app |
| `preferred_device` | category | Android / iOS / Web / USSD |
| `credit_score` | int | 300–850 |
| `monthly_income_uzs` | int | Monthly income in UZS |
| `identity_verified` | bool | KYC verification status |
| `historical_tx_count` | int | All-time transaction count |

### `merchants.csv`
| Column | Type | Description |
|--------|------|-------------|
| `merchant_id` | string | Unique merchant ID |
| `merchant_name` | string | Business name |
| `category` | string | Business category (12 types) |
| `city` | string | Merchant location |
| `region` | string | Administrative region |
| `mcc_code` | int | Merchant Category Code |
| `risk_score` | float | Platform risk score (0–100) |
| `avg_transaction_uzs` | int | Average transaction size |
| `is_online` | bool | Online merchant flag |
| `years_registered` | int | Years on platform |

---

## 🎯 Tasks You Can Solve

1. **Fraud Detection** — Binary classification (`is_fraud`). Imbalanced dataset (~3.5% fraud rate), real-world class weights matter.
2. **Anomaly Detection** — Unsupervised approach with Isolation Forest, DBSCAN, or Autoencoders
3. **Customer Segmentation** — RFM analysis + clustering on user behavior
4. **Risk Scoring** — Build merchant risk models from transaction patterns
5. **Time Series Analysis** — Spending patterns, seasonality, daily/weekly cycles
6. **Churn Prediction** — Identify inactive users from transaction frequency

---

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| Total transactions | 150,000 |
| Fraud rate | ~3.47% |
| Date range | Jan 2023 – Dec 2024 |
| Cities covered | 12 (all major UZ cities) |
| Payment apps | 8 (Click, Payme, Uzum Bank, etc.) |
| Avg transaction | ~273,000 UZS (~$22 USD) |
| Max transaction | 50,000,000 UZS (~$3,937 USD) |

---

## 🔍 Fraud Signal Overview

Fraud transactions in this dataset are engineered with realistic signals:

- **High-value transfers** > 5M UZS show elevated fraud rates
- **Nighttime activity** (00:00–04:00) correlates with fraud
- **Cross-city transactions** (user ≠ merchant city) are riskier
- **Multiple login attempts** before transaction is a red flag
- **USSD channel** has higher fraud incidence than app/web
- **High merchant risk scores** correlate with fraud occurrence

---

## 🚀 Quick Start

```python
import pandas as pd

tx = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
users = pd.read_csv("users.csv", parse_dates=["registration_date"])
merchants = pd.read_csv("merchants.csv")

# Full enriched dataset
df = tx.merge(users, on="user_id").merge(merchants, on="merchant_id", suffixes=("_user","_merchant"))

print(f"Shape: {df.shape}")
print(f"Fraud rate: {df.is_fraud.mean():.2%}")
```

---

## 🏗️ Baseline Model

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

features = ["amount_uzs","hour_of_day","day_of_week","is_weekend",
            "session_duration_sec","login_attempts","is_cross_city_tx",
            "credit_score","monthly_income_uzs","identity_verified",
            "risk_score","is_online"]

df_merged = tx.merge(users, on="user_id").merge(merchants, on="merchant_id")

X = df_merged[features]
y = df_merged["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, scale_pos_weight=29, random_state=42)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f"Baseline AUC: {auc:.4f}")  # Expected: ~0.91+
```

---

## 🌍 Context: Uzbekistan Fintech

Uzbekistan's fintech sector has seen explosive growth since 2019:
- **Click** and **Payme** process millions of daily transactions
- **Uzum Bank** (formerly Apelsin) reached 10M+ users by 2024
- Mobile payment adoption grew 400%+ between 2020–2024
- Fraud detection remains a critical unsolved challenge for local banks

This dataset captures these dynamics: heavy Android usage, USSD still relevant in rural areas, and cross-city fraud patterns unique to the region.

---

## 📝 Citation

If you use this dataset in research or projects, please cite:
```
@dataset{uz_fintech_fraud_2025,
  title={Central Asia Fintech Fraud Detection Dataset},
  author={Safarbek},
  year={2025},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/safar1/central-asia-fintech-fraud}
}
```

---

## ⚠️ Disclaimer

This is a **synthetic dataset** generated to reflect realistic statistical properties of Central Asian mobile payment systems. No real personal or financial data is included. Data distributions are based on publicly available fintech market reports for Uzbekistan.

---

*Tags: fraud-detection, fintech, uzbekistan, central-asia, imbalanced-classification, anomaly-detection, tabular, time-series*
