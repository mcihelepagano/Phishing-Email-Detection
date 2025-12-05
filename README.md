# Phishing Email Detection

A study about how ML and DL can help detecting phishing emails

## Project's Background

Phishing, as its still defined nowadays, exists as a concept since 1996. It's the result of social engineering attacks against America On-Line accounts. Attackers, (= fishers), used social engineered messages, (= baits) to steal personal informations of their victims (= fish). So called "fishing" became "phishing" after "ph" became a popular hacking replacement to "f" since the "Phone Phreaking" exploits.

Since then, phishing has evolved into one of the most widespread cyber threats worldwide. Attackers continuously adapt, leveraging increasingly sophisticated techniques to bypass traditional filters and exploit human trust. Email remains the most common delivery vector for phishing attempts due to its low cost and potential reach.

Traditional email filtering approaches such as rule-based systems, keyword detection, and blacklists have struggled to keep up with the creativity of attackers, who now craft highly personalized and context-aware messages. This arms race has positioned machine learning (ML) and deep learning (DL) as powerful allies in phishing detection, enabling automated systems to uncover hidden patterns, linguistic markers, and behavioral cues that are often invisible to static rule-based methods.

By leveraging large-scale datasets and advanced modeling techniques, ML and DL approaches offer the possibility of improving detection rates, reducing false positives, and adapting more quickly to new phishing strategies. This study explores how these methods can enhance the robustness of phishing email detection and what challenges remain to be solved.

## Research's Problem

Despite significant advances in ML and DL approaches for phishing detection, a core challenge remains: how can detection systems balance high accuracy with adaptability and interpretability when faced with evolving phishing tactics?

This study will aim to answer the following question:

**How effective are machine learning and deep learning approaches at detecting phishing emails, and what trade-offs exist between accuracy, adaptability, and interpretability in real-world deployment ?**

## State of the Art

Research on phishing detection highlights two complementary trends: the use of natural language processing (NLP) with traditional machine learning, and the application of deep learning to capture more complex linguistic patterns.

Survey work on phishing email detection emphasizes how textual features can be exploited to distinguish legitimate messages from malicious ones. Lexical choices, structural cues, and semantic signals are all important indicators. Classical ML models such as Support Vector Machines (SVM), Random Forests, or Decision Trees have demonstrated strong performance regarding phishing email identification. Their main limitation lies in the need for careful feature engineering and their reduced capacity to handle previously unseen, adaptive phishing strategies.

In parallel, a growing body of research focuses on deep learning. These methods; including recurrent architectures, convolutional networks, and transformers; excel at automatically extracting hierarchical features and modeling contextual dependencies within email text. Unlike traditional models, they reduce reliance on manual preprocessing and can generalize better across different phishing variants. However, they require large volumes of labeled data, significant computational resources, and often operate as "black boxes", raising challenges for interpretability and trust.

Current findings suggest that the most promising approaches often lie at the intersection of the two paradigms. Hybrid or ensemble techniques, which combine the efficiency and interpretability of traditional ML with the expressive power of DL, are emerging as strong candidates to address the evolving nature of phishing campaigns.

## Sources

### Papers

These are the papers I used as a foundation to my study.

[Phishing Email Detection Using Natural Language Processing Techniques: A Literature Survey](https://www.sciencedirect.com/science/article/pii/S1877050921011741) by Said Salloum, Tarek Gaber, Sunil Vadera and Khaled Shaalan

[Deep Learning for Phishing Detection: Taxonomy, Current Challenges and Future Directions](https://ieeexplore.ieee.org/abstract/document/9716113) by Nguyet Quang Do, Wilayah Persekutuan, Ali Selamat, Ondrej Krejcar, Enrique Herrera-Viedma and Hamido Fujita

### Dataset

To conduct my study, I'll use a dataset made by Advaith S. Rao.
His dataset is based on 3 distinct datasets brought together in a way that facilitates ML training.

- The **"Enron emails dataset"** containing emails recovered from the Enron company upon its bankrupcy.
- A **"Phishing emails dataset"** containing phishing emails.
- And a **"Social engineering dataset"** containing more phishing emails.

Rich of 32 columns and around 450k rows, this dataset will give me the required substance to answer this study's
interrogations.

More informations available on the kaggle page of the dataset: [DATASET ON KAGGLE](https://www.kaggle.com/datasets/advaithsrao/enron-fraud-email-dataset/data)

## How to run ?

- Clone the repo using `git clone` and navigate to the root of the project.
- After making sure you had python installed, run `python -m venv .venv` to create a venv and activate the venv.
    - `source .venv/bin/activate` on linux
    - `.venv/script/activate.ps1` on windows
- Once in your venv, run `pip install -r ./src/requirements.txt` to install the dependencies
- Finally, run the app using `python ./src`


## Deep Learning model

- Ensure dependencies are installed: `pip install -r ./src/requirements.txt` (adds torch + tqdm; transformers removed to stay lightweight).
- Launch the CLI with `python ./src` and select "Train a DL model". Choose between:
    - `Train lightweight TextCNN (cross-entropy)`
    - `Train lightweight TextCNN (focal loss + optional GA threshold)`
- Defaults aim to stay CPU-friendly: epochs=4, batch size=32, lr=1e-3, max tokens=200, vocab=20k, sample limit=80k emails. The model is a small 1D CNN over token embeddings; imbalance is handled with class weights and sampling.
- Saved artifacts live in `models/text_cnn_phishing/` (model.pt, vocab.json, label mapping, config, and optional GA-tuned threshold).
- To evaluate, pick `Evaluate saved lightweight DL model`; metrics include accuracy, precision, recall, F1, ROC-AUC, and class support. If a GA threshold was tuned, it is reused for binary problems.
- GPU is optional but will shorten training time if available.
