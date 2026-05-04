# TabRefDetect

**Numerical Discrepancy Detection in Cross-Document Table Citations**

TabRefDetect is an automated framework designed to detect whether a citing paper's table introduces a quotation error when referencing numerical values from a cited paper's table. It provides both a closed-source multimodal large language model (MLLM) reasoning strategy and a three-stage fine-tuning strategy based on open-source MLLMs.

---

##  Abstract

[**Objective**] The task of numerical discrepancy detection in cross-document table citations aims to determine whether a citing paper's table introduces a quotation error when referencing numerical values from a cited paper's table. To address the low efficiency and the oversight in manual verification, this study develops an automated detection strategy.

[**Methods**] First, we constructed a human-annotated dataset for this task, named **TabRefError**. This dataset contains numerical alignment samples across diverse table types. Second, we proposed a reasoning strategy based on closed-source multimodal large language models and a three-stage fine-tuning strategy based on open-source multimodal large language models.

[**Results**] Experimental results indicate that the closed-source model achieves a macro F1 score of 0.81. The fine-tuned open-source model achieves a macro F1 score of 0.78, performing comparably to the closed-source model. This validates the effectiveness of the fine-tuning strategy for this task.

[**Limitations**] Although this study achieves automatic detection of numerical discrepancies, discrepancies in experimental settings also should be concerned. The automatic classification of discrepancy causes has not been accomplished.

[**Conclusions**] This research provides a dataset and technical foundation for numerical discrepancy detection in cross-document table quotations. This research also contributes practical value for intelligent editorial review and research evaluation tasks. Future work will include contextual information to achieve automatic classification of the causes of numerical discrepancies.

---

##  Dataset: TabRefError

We introduce **TabRefError**, a human-annotated dataset for numerical discrepancy detection in cross-document table citations. It contains numerical alignment samples across diverse table types.

> ⚠️ **Note:** Since our human annotations cannot guarantee absolute correctness, we currently release only a **portion of the dataset** to ensure quality.

### How to Access

| Access Method | Description |
|---|---|
| **GitHub Release** | A partial subset of the TabRefError dataset is available for download via the [GitHub Releases](https://github.com/huabinW/TabRefDetect/releases) page. |
| **Full Dataset** | For the complete dataset, please contact the authors via email. |

📧 For full dataset access, please contact the corresponding author.

---

##  Getting Started

### 1. API-based Inference (Closed-Source MLLM)

If you wish to use the closed-source model reasoning strategy, please refer to the API client code:

 [`API_client/`](https://github.com/huabinW/TabRefDetect/tree/main/API_client)

This module contains scripts and instructions for calling closed-source multimodal large language models to perform numerical discrepancy detection.

### 2. Fine-Tuning (Open-Source MLLM)

If you wish to fine-tune an open-source multimodal large language model using our proposed three-stage fine-tuning strategy, please refer to:

 [`Code/`](https://github.com/huabinW/TabRefDetect/tree/main/Code)

This module contains the complete fine-tuning pipeline, training configurations, and evaluation scripts.

---

