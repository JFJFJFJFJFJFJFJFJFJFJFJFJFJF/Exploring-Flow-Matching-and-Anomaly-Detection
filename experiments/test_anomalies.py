import torch

from models.vector_field_unet import VectorFieldUNet
from flow.paths import sample_conditional_path, conditional_vector_field
from data.dataset import MNISTDigitDataset
from flow.solve import solve_backward
from config import DIGIT, SIGMA, DIM, BASE_CHANNELS, NUM_SAMPLES, NUM_STEPS, NUM_PROBES, N_PER_DIGIT, CHECKPOINT_PATH


def score_cfm_loss(model, x, sigma=SIGMA, num_samples=NUM_SAMPLES, t_fixed=None):
    #initialize loss tensor shape(n,)
    losses = torch.zeros(x.size(0))

    #Estimate CFM loss expectation (using Monte Carlo over num_samples), either for t~U(0,1) or conditional expectation for T=t_0 fixed
    for _ in range(num_samples):
        if t_fixed is None:
            t = torch.rand(x.size(0), 1)
        else:
            t = torch.full((x.size(0), 1), float(t_fixed))

        y, _ = sample_conditional_path(x, t, sigma)
        xi_target = conditional_vector_field(y, t, x, sigma)
        with torch.no_grad():
            xi_pred = model(y, t)

        #Square error and reduce over pixel (.mean(dim=1)) getting shape (n,)
        losses += ((xi_pred - xi_target) ** 2).mean(dim=1)
    return losses / num_samples

def load_model(checkpoint_path=CHECKPOINT_PATH, base_channels=BASE_CHANNELS):
    model = VectorFieldUNet(dim=DIM, base_channels=base_channels)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def evaluate_digit(model, digit, sigma=SIGMA, n=N_PER_DIGIT, num_samples=NUM_SAMPLES):

    #Load test data set for the specific digit
    dataset = MNISTDigitDataset(digit=digit, train=False)
    n = min(n, len(dataset))
    x = torch.stack([dataset[i] for i in range(n)])

    return {"digit": digit, "n": n,
            "cfm": score_cfm_loss(model, x, sigma=sigma, num_samples=num_samples)}


def print_summary(results):
    #Statistics (mean, median, min, max) of the CFM loss per digit
    print(f"\n=== CFM loss ===")
    print(f"{'Digit':>7} | {'mean':>12} | {'median':>12} | {'min':>12} | {'max':>12}")
    print("-" * 68)
    for r in results:
        v = r["cfm"] 
        print(f"{r['digit']:>7} | {v.mean():>12.5f} | {v.median():>12.5f} | "
              f"{v.min():>12.5f} | {v.max():>12.5f}")


def print_separation(results, quantiles=(1.0, 0.99)):
    #Extract losses for the reference/"good" digit (i.e. digit 0)
    zero = next((r for r in results if r["digit"] == DIGIT), None)
    if zero is None:
        return

    #Quantile tresholds 
    thresholds = {q: torch.quantile(zero["cfm"], q).item() for q in quantiles}

    print(f"\n--- CFM loss: separation against digit {DIGIT} ---")
    for q in quantiles:
        print(f"Digit {DIGIT}: q{q:.2f}={thresholds[q]:.5f}")
    header = " | ".join(f"below q{q:.2f}(0)" for q in quantiles)
    print(f"{'Digit':>7} | {header}")
    print("-" * (10 + 18 * len(quantiles)))

    totals = {q: 0 for q in quantiles}
    total = 0
    for r in results:
        if r["digit"] == DIGIT:
            continue
        v = r["cfm"]
        total += v.numel()
        row = []
        for q in quantiles:
            #Counting false negatives
            below = (v <= thresholds[q]).sum().item()
            totals[q] += below
            row.append(f"{below:>6}/{v.numel():<7}")
        print(f"{r['digit']:>7} | " + " | ".join(row))

    print("-" * (10 + 18 * len(quantiles)))
    for q in quantiles:
        print(f"False negatives (q={q:.2f}): {totals[q]/total:.1%}")


def test_anomalies(digits=tuple(range(10)),
                   checkpoint_path=CHECKPOINT_PATH,
                   base_channels=BASE_CHANNELS, sigma=SIGMA, n_per_digit=N_PER_DIGIT,
                   num_samples=NUM_SAMPLES):

    model = load_model(checkpoint_path, base_channels)

    results = []
    for digit in digits:
        print(f"Computing digit {digit}...", end=" ", flush=True)
        results.append(evaluate_digit(
            model, digit, sigma=sigma, n=n_per_digit, num_samples=num_samples
        ))
        print("done")

    return results



def score_likelihood(model, x, num_steps=NUM_STEPS, num_probes=NUM_PROBES):

    #Compute likelihood via backward ODE solver (initial anomaly score)
    _, log_p = solve_backward(model, x, num_steps=num_steps,
                              compute_likelihood=True, num_probes=num_probes)
    #negating for high = anomalous
    return -log_p


def score_cfm_map(model, x, sigma=SIGMA, num_samples=NUM_SAMPLES):
    #Function for heat map (similar to score_cfm_loss, but keep pixel information instead of averaging)
    maps = torch.zeros_like(x)
    for _ in range(num_samples):
        t = torch.rand(x.size(0), 1)
        y, _ = sample_conditional_path(x, t, sigma)
        xi_target = conditional_vector_field(y, t, x, sigma)
        with torch.no_grad():
            xi_pred = model(y, t)

        #Square error NOT reduced over pixel shape(n, d)
        maps += (xi_pred - xi_target) ** 2
    return maps / num_samples

if __name__ == "__main__":
    results = test_anomalies()
    print_summary(results)
    print_separation(results, quantiles=(1.0, 0.99, 0.95, 0.90))