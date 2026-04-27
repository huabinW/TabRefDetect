import json
import os
from collections import defaultdict, Counter
import numpy as np


def load_and_prepare_data(data_path):
    """Load data and group samples by article ID."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not all("label" in item and "paper_id" in item for item in data):
        raise ValueError("Missing 'label' or 'paper_id' in dataset.")

    article_to_samples = defaultdict(list)
    for idx, item in enumerate(data):
        # Extract base article ID by splitting the specific sample ID
        article_id = item["paper_id"].split('_')[0]
        article_to_samples[article_id].append((idx, item["label"]))

    article_info = []
    for article_id, samples in article_to_samples.items():
        labels = [x[1] for x in samples]
        label_dist = Counter(labels)

        article_info.append({
            'id': article_id,
            'samples': [x[0] for x in samples],
            'labels': labels,
            'label_dist': label_dist,
            'size': len(labels),
            'majority_label': label_dist.most_common(1)[0][0]
        })

    return data, article_info


def create_balanced_folds(article_info, n_splits=5, random_state=42):
    """Create balanced folds based on article size and majority label."""
    np.random.seed(random_state)
    all_folds = [[] for _ in range(n_splits)]

    # Group all articles by their majority label
    label_groups = defaultdict(list)
    for art in article_info:
        label_groups[art['majority_label']].append(art)

    # Distribute articles to folds
    for label, group in label_groups.items():
        # Sort by sample size descending to balance the total size across folds
        sorted_group = sorted(group, key=lambda x: x['size'], reverse=True)
        for i, art in enumerate(sorted_group):
            fold_idx = i % n_splits
            all_folds[fold_idx].append((art, art['size'], art['majority_label']))

    return all_folds


def main():
    # Relative paths
    data_path = "./biaoge.json"
    output_dir = "./folds"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare
    data, article_info = load_and_prepare_data(data_path)
    all_labels = [label for art in article_info for label in art['labels']]
    unique_labels = set(all_labels)

    # Generate folds
    all_folds = create_balanced_folds(article_info, n_splits=5, random_state=42)
    fold_info = []

    for fold in range(5):
        # Split train/val articles
        val_articles = [item[0] for item in all_folds[fold]]
        train_articles = []
        for i in range(5):
            if i != fold:
                train_articles.extend([item[0] for item in all_folds[i]])

        # Get sample indices
        train_indices = [idx for art in train_articles for idx in art['samples']]
        val_indices = [idx for art in val_articles for idx in art['samples']]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        # Calculate distributions
        train_dist = Counter([item["label"] for item in train_data])
        val_dist = Counter([item["label"] for item in val_data])

        train_ratio = {k: v / len(train_data) for k, v in train_dist.items()} if train_data else {}
        val_ratio = {k: v / len(val_data) for k, v in val_dist.items()} if val_data else {}

        fold_info.append({
            'fold': fold,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'train_label_ratio': train_ratio,
            'val_label_ratio': val_ratio
        })

        # Save split files
        with open(os.path.join(output_dir, f"fold_{fold}_train.json"), "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        with open(os.path.join(output_dir, f"fold_{fold}_val.json"), "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=4)

        print(f"Fold {fold}: Train={len(train_data)}, Val={len(val_data)}")

    # Save statistics
    stats_path = os.path.join(output_dir, "folds_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(fold_info, f, ensure_ascii=False, indent=2)

    # Print balance analysis
    print("\nLabel Ratio Balance Analysis (Validation Sets):")
    for label in unique_labels:
        ratios = [fi['val_label_ratio'].get(label, 0) for fi in fold_info]
        print(f"  '{label}': Mean={np.mean(ratios):.3f}, Std={np.std(ratios):.3f}")


if __name__ == "__main__":
    main()
