import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from data.dataset import MNISTDigitDataset
from experiments.test_anomalies import score_cfm_loss, score_likelihood, load_model
from config import SIGMA, BASE_CHANNELS, NUM_SAMPLES, NUM_STEPS, NUM_PROBES, N_NORMAL, N_ANOMALY_PER_DIGIT, CHECKPOINT_PATH, DIGIT, DIGITS_ANOMALY

def collect_scores(model, score_fn, digits_anomaly, n_normal, n_per_digit, **kwargs):
    #Load reference/ "good" dataset (digit 0)and compute scores shape (n_normal,)
    zero_dataset = MNISTDigitDataset(digit=DIGIT, train=False)
    x_normal = torch.stack([zero_dataset[i] for i in range(min(n_normal, len(zero_dataset)))])
    scores_normal = score_fn(model, x_normal, **kwargs)

    #Loop through anomaly digits and compute their scores shape (n_per_digit,) and collect in list
    scores_anomaly_all = []
    for digit in digits_anomaly:
        ds = MNISTDigitDataset(digit=digit, train=False)
        n = min(n_per_digit, len(ds))
        x = torch.stack([ds[i] for i in range(n)])
        scores_anomaly_all.append(score_fn(model, x, **kwargs))

    #Concatenate all  anomaly scores shape (n_per_digit * 9,)
    scores_anomaly = torch.cat(scores_anomaly_all)

    #Combine normal and anomaly scores into a single array
    all_scores = torch.cat([scores_normal, scores_anomaly]).numpy()

    #Introduce binary labels
    labels = torch.cat([
        torch.zeros(scores_normal.size(0)),   # 0 = normal
        torch.ones(scores_anomaly.size(0)),   # 1 = anomaly
    ]).numpy()
    return all_scores, labels


def evaluate_roc(
    digits_anomaly=DIGITS_ANOMALY,
    checkpoint_path=CHECKPOINT_PATH,
    base_channels=BASE_CHANNELS, sigma=SIGMA,
    n_normal=N_NORMAL, n_per_digit=N_ANOMALY_PER_DIGIT,
    num_samples=NUM_SAMPLES,
    num_steps=NUM_STEPS, num_probes=NUM_PROBES,
    with_likelihood=True,
):
    model = load_model(checkpoint_path, base_channels)

    fig, ax = plt.subplots(figsize=(7, 6))
    results = {}

    #Evaluate performance using CFM loss
    print("Scoring: CFM loss")
    scores, labels = collect_scores(
        model, score_cfm_loss, digits_anomaly, n_normal, n_per_digit,
        sigma=sigma, num_samples=num_samples,
    )
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    results["cfm"] = auc
    print(f"CFM loss: AUC = {auc:.4f}")
    ax.plot(fpr, tpr, label=f"CFM loss (AUC={auc:.3f})", linewidth=2)

    #Evaluate performance using negative log-likelihood (optionally)
    if with_likelihood:
        print("\nScoring: negative log-likelihood")
        scores, labels = collect_scores(
            model, score_likelihood, digits_anomaly, n_normal, n_per_digit,
            num_steps=num_steps, num_probes=num_probes,
        )
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        results["nll"] = auc
        print(f"Negative log-likelihood: AUC = {auc:.4f}")
        ax.plot(fpr, tpr, label=f"Neg. log-likelihood (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves: anomaly detection (digit {DIGIT} vs. rest)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    print("\nROC curve saved to roc_curve.png")

    return results


if __name__ == "__main__":
    evaluate_roc()