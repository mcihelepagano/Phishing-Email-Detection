
# Deep Learning Classifier

DistilBERT fine-tuning was dropped because it was too resource-intensive on typical laptops. The deep learning track now uses a lightweight TextCNN trained on in-house tokenization, with options to change the scoring function (cross-entropy vs. focal loss) and to run a tiny genetic algorithm to tune the decision threshold for binary problems.

## Implementation overview
- Dataset: uses `datamanager.df`; if `text_combined` is missing it is created from Subject + Body.
- Split: train/validation split (80/20, stratified when possible) with imbalance handled through class weights and a `WeightedRandomSampler`.
- Tokenization: regex-based word splits, capped/padded to `max_length` (default 200 tokens). Vocabulary is limited to the top `max_vocab_size` terms (default 20k) to stay memory friendly.
- Architecture: embedding layer feeding a 1D CNN with kernels (3, 4, 5), max-pooling, and a small MLP head.
- Scoring: choose `cross_entropy` or `focal` (gamma configurable). For binary runs, an optional GA searches thresholds that maximize F1 on the validation set.
- Training defaults: epochs=4, batch_size=32, lr=1e-3, focal_gamma=2.0, sample limit=80k rows to keep CPU runs feasible.
- Persistence: saves `model.pt`, `vocab.json`, `label_mapping.json`, `config.json`, and (optionally) `threshold.json` under `models/text_cnn_phishing/`.

## How to train
1. Install dependencies: `pip install -r ./src/requirements.txt`.
2. Run the CLI: `python ./src`.
3. Load the dataset, then choose `Train a DL model` and pick either:
   - `Train lightweight TextCNN (cross-entropy)`, or
   - `Train lightweight TextCNN (focal loss + optional GA threshold)`.
4. Accept defaults or set epochs/batch size/lr/max tokens/vocab/sample limit. If focal loss is selected, you can also set gamma and opt into GA-based threshold tuning (binary only).

## How to evaluate
- From the CLI, choose `Train a DL model` > `Evaluate saved lightweight DL model`. It reloads artifacts from `models/text_cnn_phishing/`, prints accuracy, precision, recall, F1, ROC-AUC, support, and applies the GA-tuned threshold if it exists for a binary setup.

## Repro tips
- Keep `train_sample_limit` modest (e.g., 50–100k) on CPU; set it to 0 to use the full dataset if you have more memory.
- Raising `max_length` or `max_vocab_size` improves coverage but increases RAM/VRAM usage.
- GPU is optional but will shorten training; CPU works with the defaults, just slower.
