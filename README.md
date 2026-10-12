# Breast Cancer Subtype Classifier (Computational Pathology)

An explainable machine learning pipeline built in Python to differentiate **Luminal** vs. aggressive **Triple-Negative Breast Cancer (TNBC)** using histopathology tissue features from a Nigerian patient cohort (29 patients, 5,283 patches). 

Optimized to run **100% offline** on low-specification local clinic desktops, bypassing power grid and data constraints in under-resourced West African clinical settings.

> 🔒 *Note: The raw underlying dataset contains confidential clinical records and is excluded from this repository to maintain ethical compliance and patient privacy.*

## 🧠 Key Methodology Adjustments
*   **Zero Data Leakage:** Replaced random patch shuffling with patient-grouped validation splits (**GroupKFold / Leave-One-Out CV**). Patches from the same patient slide are strictly locked together to guarantee the model validates only on entirely unseen human profiles.
*   **Resolved Pseudo-Replication:** Proved that running univariate significance checks on pooled patches falsely inflates significance (p < 10⁻²⁴). Aggregated rows honestly to the patient level (N=29) to eliminate micro-noise.
*   **Artifact-Blindness:** Conducted a Metadata Drop Test by stripping out 5 processing variables. Cross-validation performance increased from 0.7596 to 0.7885 ROC-AUC, proving the pipeline relies on pure tissue anatomy rather than microscope scanning artifacts.
*   **Permutation "Lie Detector":** Retrained the model 200 times on randomly scrambled labels. Meaningless noise models collapsed to a random 0.4559 AUC, validating our real model's **0.7885 AUC** as a genuine biological signal (p = 0.015).

## 📈 Operational Performance Metrics
*(Evaluated via un-leaked Leave-One-Out Cross-Validation on biological features)*
*   **Area Under Curve (ROC-AUC):** **0.7885** (p = 0.015)
*   **Recall (Sensitivity):** **0.8125** (Successfully catches 81.25% of true TNBC cases)
*   **Overall System Accuracy:** **0.7241**
*   **Precision Score:** **0.7222**
*   **F1-Score:** **0.7647**

## 🧬 Explainable AI (XAI) Discoveries
Extracted internal tree weights from the champion **XGBoost Classifier** to break open the black box:
*   **Dark Hotspots (`G_min`, `R_min`, `B_min` — 30.1% Importance):** Bypassed diluted feature averages to target absolute darkest thresholds where dense, crowded Luminal nuclei clusters heavily absorb purple stain.
*   **Nuclear Distortion (`eccentricity_mean` — 11.59% Importance):** Captures the extreme elongated stretching of nuclei as Luminal cell groups compress tightly inside branching duct loops.
*   **Sheet Streaming (`circularity_mean` — 7.41% & `texture_correlation` — 4.88% Importance):** Identifies aggressive TNBC spreading in uniform, continuous sheets. Explainability analysis showed TNBC patches had a **34.42% lower nuclei count**, consistent with tumor necrosis.

## 📁 Repository Structure
*   `pathology_analysis.py`: Master Python script housing the clean data preprocessing, cross-validation, modeling, and evaluation pipelines.

---
*Developed as part of a lecturer-led computational pathology research.*
