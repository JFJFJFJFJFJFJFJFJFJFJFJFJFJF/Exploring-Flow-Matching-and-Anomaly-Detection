import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from experiments.evaluate_roc import collect_scores
from experiments.test_anomalies import score_cfm_loss, load_model
from config import (SIGMA, BASE_CHANNELS, NUM_SAMPLES, N_NORMAL, N_ANOMALY_PER_DIGIT, CHECKPOINT_PATH, DIGITS_ANOMALY)

T_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0)
RESULTS_PATH = "results_time_sweep.json"


def _save(results):
    #Save results dictionary to JSON file
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def _auc_for(model, t_fixed, sigma, n_normal, n_per_digit, num_samples, seed):
    #Fix seed across evaluations
    torch.manual_seed(seed)

    #Collect scores and labels for current time t (or uniform distribution if t_fixed is None)
    scores, labels = collect_scores(
        model, score_cfm_loss, DIGITS_ANOMALY, n_normal, n_per_digit,
        sigma=sigma, num_samples=num_samples, t_fixed=t_fixed,
    )

    #Compute ROC-AUC
    return roc_auc_score(labels, scores)


def sweep_time(t_grid=T_GRID, checkpoint_path=CHECKPOINT_PATH, base_channels=BASE_CHANNELS, sigma=SIGMA, n_normal=N_NORMAL, n_per_digit=N_ANOMALY_PER_DIGIT, num_samples=NUM_SAMPLES, seed=0):

    #Load trained model
    model = load_model(checkpoint_path, base_channels)
    results = {}

    #Compute AUC for t~U(0,1)
    auc = _auc_for(model, None, sigma, n_normal, n_per_digit, num_samples, seed)
    results["averaged"] = auc
    print(f"t ~ U([0,1])  |  AUC = {auc:.4f}", flush=True)
    _save(results)

    #Evaluate AUC individually for each fixed time step in t_grid
    for t in t_grid:
        auc = _auc_for(model, t, sigma, n_normal, n_per_digit, num_samples, seed)
        results[f"{t:.2f}"] = auc
        print(f"t = {t:.2f}      |  AUC = {auc:.4f}", flush=True)
        _save(results)

    return results


def plot_sweep(results=None, save_path="time_sweep.png"):

    #Load results from JSON if not directly provided
    if results is None:
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    #Extract sorted time steps and corresponding AUC values (excluding t averaged)
    ts = sorted(float(k) for k in results if k != "averaged")
    aucs = [results[f"{t:.2f}"] for t in ts]

    #Plot AUC over time steps t and compare against t averaged
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ts, aucs, "o-", linewidth=2, label=r"score at a single fixed $t$")
    ax.axhline(results["averaged"], color="k", linestyle="--", linewidth=1,
               label=rf"$t \sim U([0,1])$ (AUC={results['averaged']:.3f})")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(r"Anomaly detection quality as a function of $t$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Sweep plot saved to {save_path}")


if __name__ == "__main__":
    results = sweep_time()
    plot_sweep(results)