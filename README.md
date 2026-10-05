# my_jev: Jev-style document decisions with Qwen + MLP

A document classifier that works like a decision model: every class is written as a hypothesis ("This is a legal document such as a contract, policy or terms."). Qwen reads the document together with one hypothesis, and a small MLP head outputs the probability that the hypothesis is true. The class with the highest probability wins.

Because the model scores hypotheses instead of fixed classes, it can also answer new yes/no questions and choose between new options without retraining (`decide()` in the notebook).

[Open in Colab](https://colab.research.google.com/github/Rozzeo/my_jev/blob/main/qwen_jev_pairs.ipynb)

## What's inside

| Path | What it is |
|---|---|
| `qwen_jev_pairs.ipynb` | Training and inference: data → pairs → Qwen + MLP with the Hugging Face `Trainer` → plots → test → `decide()` → `predictions.csv` |
| `data/train.csv`, `data/valid.csv`, `data/test.csv` | Demo dataset: 1,499 business texts labeled `marketing`, `legal`, `other` |
| `make_dataset.py` | Rebuilds `data/` from public sources |

## How the model works

- **Backbone:** `Qwen/Qwen3-0.6B` (Apache 2.0). The last 7 transformer layers are fine-tuned; the rest stays frozen.
- **Head:** LayerNorm → Linear(1024→256) → GELU → Dropout → Linear(256→1). It outputs one score per (document, hypothesis) pair.
- **Training data:** for each document, one pair per class (target 1 for the true class, 0 for the others), plus 2,000 public MNLI pairs. The MNLI pairs teach the general skill "the text supports this statement", which is what makes new questions work.
- **Loss:** binary cross-entropy with `pos_weight`, so 0.5 stays a fair yes/no threshold.
- **Learning rates:** 1e-3 for the new MLP head, 2e-5 for the Qwen layers. The best epoch is chosen by validation macro-F1.

## Run on Colab

1. Open the notebook with the link above. If the repository is private, Colab asks for GitHub access first.
2. *Runtime → Change runtime type → T4 GPU*.
3. Upload the `data` folder to Google Drive as `MyDrive/my_jev/data/`.
4. *Run all*, and allow Drive access when asked.

The notebook writes `predictions.csv` and `qwen_jev_pairs_weights.pt` (the trained layers and head, about 0.45 GB) next to the data on Drive. Git ignores both.

To change the classes, edit `classes` in the settings cell. Labels in the data must match its keys.

## Dataset

Columns: `text` (lowercased, at most 3,000 characters), `label` (`marketing`, `legal`, `other`) and `source` (origin and id, for example `ledgar:12`).

The split is 70 / 15 / 15 with the same class balance in every part: train 1,049, valid 225, test 225.

| Label | Rows | Source | What it is |
|---|---|---|---|
| legal | 300 | LEDGAR (LexGLUE) | Contract provisions from SEC filings |
| legal | 200 | UNFAIR-ToS (LexGLUE) | Sentences from online Terms of Service |
| marketing | 350 | Ads Creative Ad Copy Programmatic | Banner ad copy |
| marketing | 150 | Enron-Spam, filtered | Promotional business emails. Keyword rules drop pills, scams and stock spam; some noise remains |
| other | 499 | Enron-Spam, "ham" part | Internal Enron business emails |

Rebuild with `pip install datasets pandas scikit-learn` and `python make_dataset.py`. The build is deterministic (seed 42).

### Limitations

- Each class comes from different sources, so a model can partly learn the form of a text (banner, contract, email) instead of its meaning. Scores on this set will look better than on real documents, so test on real, labeled work documents.
- All text is lowercased. The Enron emails exist only in lowercase, and mixed case elsewhere would give the class away.
- English only. The emails date from 1999 to 2005.

### Sources and licenses

- **LEDGAR** and **UNFAIR-ToS**, via [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue), CC BY 4.0.
  Chalkidis et al., *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL 2022.
  Tuggener et al., *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*, LREC 2020.
  Lippi et al., *CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service*, Artificial Intelligence and Law, 2019.
- **[Ads Creative Ad Copy Programmatic](https://huggingface.co/datasets/PeterBrendan/Ads_Creative_Ad_Copy_Programmatic)**, MIT.
- **[Enron-Spam](https://huggingface.co/datasets/SetFit/enron_spam)**: Metsis, Androutsopoulos and Paliouras, *Spam Filtering with Naive Bayes – Which Naive Bayes?*, CEAS 2006. The dataset card states no license. The Enron emails were made public by the US Federal Energy Regulatory Commission.
- **[MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli)**, downloaded by the notebook at run time and not stored here. Williams et al., NAACL 2018.
