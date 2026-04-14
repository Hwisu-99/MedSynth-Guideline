# Quantitative Guidelines for Anonymity Assessment of Synthetic Health Checkup Data

## 1. Background
* **Problem**: Current government guidelines provide evaluation metrics but lack **specific quantitative thresholds** for determining "anonymity."
* **Impact**: This ambiguity creates legal uncertainty, leading to a "bottleneck" in the medical data economy.
* **Goal**: To establish precise, numeric "Pass/Fail" criteria for synthetic medical data.

## 2. Methodology
### Generation Tools
* **ARX**: K-Anonymity based de-identification.
* **Synthpop**: ML-based statistical synthesis.
* **SDV (Synthetic Data Vault)**: Generative AI (GAN, VAE) models.
* **MASQ**: Privacy-preserving synthesis via Differential Privacy.

### Evaluation Metrics
* **Safety**: Re-identification Risk, Linkage Risk, and **DCR (Distance to Closest Record)**.
* **Utility**: Correlation Similarity and **TSTR (Train on Synthetic, Test on Real)**.

## 3. Proposed Anonymity Guidelines (Thresholds)

| Category | Metric | Proposed Threshold |
| :--- | :--- | :--- |
| **Safety** | DCR (Zero-distance Ratio) | **< 0.1%** |
| **Safety** | Mean DCR | **> 10% of Std. Dev.** |
| **Utility** | TSTR Performance Retention | **> 90%** |
| **Utility** | Correlation Delta | **< 0.05** |

## 4. Dataset
* **Source**: [National Health Insurance Service - Health Checkup Data](https://www.data.go.kr/data/15007122/fileData.do)
* **Provider**: National Health Insurance Service (via data.go.kr)
