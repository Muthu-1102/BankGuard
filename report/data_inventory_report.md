# BankGuard — Raw Data Inventory Report

- **Source Directory:** `D:\Projects\BankGurd\data\raw\ieee_cis`
- **Total Files:** 4
- **Total Size on Disk:** 1286.38 MB (~1.26 GB)

## 1. File Summary Table

| File Name | File Size (MB) | Row Count | Column Count | Est. In-Memory Size (MB) | Has Target (`isFraud`) | Status |
|---|---:|---:|---:|---:|:---:|:---:|
| `train_transaction.csv` | 651.69 | 590,540 | 394 | 2,084.9 | Yes | Verified |
| `train_identity.csv` | 25.3 | 144,233 | 41 | 159.63 | No | Verified |
| `test_transaction.csv` | 584.79 | 506,691 | 393 | 1,818.49 | No | Verified |
| `test_identity.csv` | 24.6 | 141,907 | 41 | 153.85 | No | Verified |

## 2. Detailed File Inspection

### `train_transaction.csv`

- **Path:** `D:\Projects\BankGurd\data\raw\ieee_cis\train_transaction.csv`
- **Disk Size:** 651.69 MB (683,351,067 bytes)
- **Exact Row Count:** 590,540
- **Column Count:** 394
- **Estimated DataFrame Memory:** 2084.9 MB
- **Data Types Breakdown:**
  - `float64`: 376 columns
  - `object`: 14 columns
  - `int64`: 4 columns
- **Column Family Counts:**
  - `TransactionID` present: True
  - `isFraud` present: True
  - `TransactionDT` present: True
  - `TransactionAmt` present: True
  - `ProductCD` present: True
  - `card_features`: 6
  - `addr_features`: 2
  - `dist_features`: 2
  - `email_features`: 2
  - `C_features`: 14
  - `D_features`: 15
  - `M_features`: 9
  - `V_features`: 339
- **First 5 Columns:** `TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD`
- **Last 5 Columns:** `V335, V336, V337, V338, V339`

### `train_identity.csv`

- **Path:** `D:\Projects\BankGurd\data\raw\ieee_cis\train_identity.csv`
- **Disk Size:** 25.3 MB (26,529,680 bytes)
- **Exact Row Count:** 144,233
- **Column Count:** 41
- **Estimated DataFrame Memory:** 159.63 MB
- **Data Types Breakdown:**
  - `float64`: 23 columns
  - `object`: 17 columns
  - `int64`: 1 columns
- **Column Family Counts:**
  - `TransactionID` present: True
  - `isFraud` present: False
  - `TransactionDT` present: False
  - `TransactionAmt` present: False
  - `ProductCD` present: False
  - `id_features`: 38
  - `device_features`: 2
- **First 5 Columns:** `TransactionID, id_01, id_02, id_03, id_04`
- **Last 5 Columns:** `id_36, id_37, id_38, DeviceType, DeviceInfo`

### `test_transaction.csv`

- **Path:** `D:\Projects\BankGurd\data\raw\ieee_cis\test_transaction.csv`
- **Disk Size:** 584.79 MB (613,194,934 bytes)
- **Exact Row Count:** 506,691
- **Column Count:** 393
- **Estimated DataFrame Memory:** 1818.49 MB
- **Data Types Breakdown:**
  - `float64`: 376 columns
  - `object`: 14 columns
  - `int64`: 3 columns
- **Column Family Counts:**
  - `TransactionID` present: True
  - `isFraud` present: False
  - `TransactionDT` present: True
  - `TransactionAmt` present: True
  - `ProductCD` present: True
  - `card_features`: 6
  - `addr_features`: 2
  - `dist_features`: 2
  - `email_features`: 2
  - `C_features`: 14
  - `D_features`: 15
  - `M_features`: 9
  - `V_features`: 339
- **First 5 Columns:** `TransactionID, TransactionDT, TransactionAmt, ProductCD, card1`
- **Last 5 Columns:** `V335, V336, V337, V338, V339`

### `test_identity.csv`

- **Path:** `D:\Projects\BankGurd\data\raw\ieee_cis\test_identity.csv`
- **Disk Size:** 24.6 MB (25,797,161 bytes)
- **Exact Row Count:** 141,907
- **Column Count:** 41
- **Estimated DataFrame Memory:** 153.85 MB
- **Data Types Breakdown:**
  - `float64`: 23 columns
  - `object`: 17 columns
  - `int64`: 1 columns
- **Column Family Counts:**
  - `TransactionID` present: True
  - `isFraud` present: False
  - `TransactionDT` present: False
  - `TransactionAmt` present: False
  - `ProductCD` present: False
  - `device_features`: 2
- **First 5 Columns:** `TransactionID, id-01, id-02, id-03, id-04`
- **Last 5 Columns:** `id-36, id-37, id-38, DeviceType, DeviceInfo`

## 3. Key Observations

1. **Identity Table Coverage:** Both `train_identity` and `test_identity` have significantly fewer rows than their transaction counterparts (~24% transaction coverage). A **left join** on `TransactionID` is mandatory to avoid dropping transaction rows.
2. **Supervised Target:** `isFraud` is present exclusively in `train_transaction.csv` and absent in `test_transaction.csv`, confirming proper supervised setup.
3. **High Feature Dimensionality:** `train_transaction.csv` contains 394 columns and `train_identity.csv` contains 41 columns (joining yields ~434 unique features), heavily dominated by anonymized `V` features (339 columns) and `id_` features (38 columns).
4. **Memory Considerations:** Uncompressed data requires ~1.5–2.0 GB in memory when fully loaded; chunked reading or downcasting float64/int64 dtypes will optimize memory footprint during feature engineering.
