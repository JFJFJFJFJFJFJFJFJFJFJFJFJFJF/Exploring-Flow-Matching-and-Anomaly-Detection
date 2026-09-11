import torch
import matplotlib.pyplot as plt
from config import DIM, SIGMA, NUM_SAMPLES, DIGIT


def plot_image_grid(images, titles=None, save_path="output.png", cols=8):

    if images.dim() == 2:
        images = images.view(images.size(0), 28, 28)

    n = images.size(0)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i].detach().numpy(), cmap="gray")
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Bilder gespeichert in {save_path}")


def plot_trajectory(model, num_samples=4, num_steps=100, save_path="trajectory.png"):

    h = 1.0 / num_steps
    y = torch.randn(num_samples, DIM)
    t = torch.zeros(num_samples, 1)
    checkpoints = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    snapshots = {}

    with torch.no_grad():
        for step in range(num_steps):
            dy = model(y, t)
            y = y + h * dy
            t = t + h
            current_t = (step + 1) / num_steps
            for cp in checkpoints:
                if abs(current_t - cp) < 1e-6:
                    snapshots[cp] = y.clone()

    fig, axes = plt.subplots(num_samples, len(snapshots), figsize=(len(snapshots)*1.5, num_samples*1.5))
    for col, (t_val, y_snap) in enumerate(sorted(snapshots.items())):
        imgs = y_snap.view(num_samples, 28, 28)
        for row in range(num_samples):
            axes[row, col].imshow(imgs[row].numpy(), cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t={t_val}")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Trajektorie gespeichert in {save_path}")

def plot_score_maps(model, images, titles=None, save_path="score_maps.png", sigma=SIGMA, num_samples=NUM_SAMPLES):

    from experiments.test_anomalies import score_cfm_map

    maps = score_cfm_map(model, images, sigma=sigma, num_samples=num_samples)
    n = images.size(0)

    # same scale across all maps
    vmin = maps.min().item()
    vmax = maps.max().item()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.8))
    for i in range(n):
        axes[0, i].imshow(images[i].view(28, 28).numpy(), cmap="gray")
        axes[0, i].axis("off")
        if titles is not None:
            axes[0, i].set_title(titles[i], fontsize=8)

        im = axes[1, i].imshow(maps[i].view(28, 28).numpy(), cmap="inferno",
                               vmin=vmin, vmax=vmax)
        axes[1, i].axis("off")

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Score maps saved to {save_path}")


if __name__ == "__main__":
    import torch
    from flow.solve import solve
    from models.vector_field_unet import VectorFieldUNet
    from config import DIM, BASE_CHANNELS, CHECKPOINT_PATH, NUM_STEPS

    model = VectorFieldUNet(dim=DIM, base_channels=BASE_CHANNELS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    # plot generated images
    z0 = torch.randn(16, DIM)
    x = solve(model, z0, num_steps=NUM_STEPS, compute_likelihood=False)
    plot_image_grid(x, save_path="generated_samples.png")

    # plot trajectory
    plot_trajectory(model, num_samples=4)

    # plot images with score
    z0_scored = torch.randn(16, DIM)
    x_scored, log_p = solve(model, z0_scored, num_steps=NUM_STEPS, compute_likelihood=True)
    titles = [f"S={log_p[i].item():.1f}" for i in range(x_scored.size(0))]
    plot_image_grid(x_scored, titles=titles, save_path="scored_samples.png")

    # plot heatmap
    from data.dataset import MNISTDigitDataset
    from experiments.test_anomalies import score_cfm_loss

    anomaly_digit = 4

    zeros = MNISTDigitDataset(digit=DIGIT, train=False)
    others = MNISTDigitDataset(digit=anomaly_digit, train=False)
    x_mix = torch.stack([zeros[i] for i in range(4)] + [others[i] for i in range(4)])

    scores = score_cfm_loss(model, x_mix, num_samples=NUM_SAMPLES)
    tau = 0.2104   # 0.95 quantile of the digit-0 test scores

    titles = []
    for i, s in enumerate(scores.tolist()):
        true_label = str(DIGIT) if i < 4 else str(anomaly_digit)
        predicted_anomaly = s > tau
        is_anomaly = i >= 4
        mark = "\u2713" if predicted_anomaly == is_anomaly else "\u2717"
        titles.append(f"{true_label}  S={s:.3f}  {mark}")

    plot_score_maps(model, x_mix, titles=titles)