import torch
from flow.paths import sample_conditional_path
from config import NUM_STEPS, NUM_PROBES


def hutchinson_divergence(model, y: torch.Tensor, t: torch.Tensor, num_probes: int = 1) -> torch.Tensor:
    y = y.detach().requires_grad_(True)
    div_estimate = torch.zeros(y.size(0))

    for _ in range(num_probes):
        eps = torch.randint(0, 2, y.shape).float() * 2 - 1

        xi = model(y, t)
        vjp = torch.autograd.grad(xi, y, grad_outputs=eps, create_graph=False)[0]
        div_estimate += (vjp * eps).sum(dim=1)

    return div_estimate / num_probes


def solve(model, z0: torch.Tensor, num_steps: int =NUM_STEPS, compute_likelihood: bool = False, num_probes: int = NUM_PROBES):
    h = 1.0 / num_steps
    y = z0.clone()
    t = torch.full((z0.size(0), 1), 0.0)
    div_sum = torch.zeros(z0.size(0)) if compute_likelihood else None

    for _ in range(num_steps):
        with torch.no_grad():
            k1 = model(y, t)
            y_mid = y + (h / 2) * k1
            t_mid = t + h / 2

        if compute_likelihood:
            div_mid = hutchinson_divergence(model, y_mid, t_mid, num_probes=num_probes)
            div_sum += div_mid * h

        with torch.no_grad():
            k2 = model(y_mid, t_mid)
            y = y + h * k2
        t = t + h

    x = y

    if not compute_likelihood:
        return x

    d = z0.size(1)
    log_p0 = -0.5 * d * torch.log(torch.tensor(2 * 3.14159265)) - 0.5 * (z0 ** 2).sum(dim=1)
    log_p_good = log_p0 - div_sum
    return x, log_p_good


def solve_backward(model, x1: torch.Tensor, num_steps: int = NUM_STEPS, compute_likelihood: bool = True, num_probes: int = NUM_PROBES):

    h = 1.0 / num_steps
    y = x1.clone()
    t = torch.full((x1.size(0), 1), 1.0)
    div_sum = torch.zeros(x1.size(0)) if compute_likelihood else None

    for _ in range(num_steps):
        with torch.no_grad():
            k1 = model(y, t)
            y_mid = y - (h / 2) * k1      
            t_mid = t - h / 2

        if compute_likelihood:
            div_mid = hutchinson_divergence(model, y_mid, t_mid, num_probes=num_probes)
            div_sum += div_mid * h        

        with torch.no_grad():
            k2 = model(y_mid, t_mid)
            y = y - h * k2
        t = t - h

    z = y

    if not compute_likelihood:
        return z

    d = x1.size(1)
    log_p0 = -0.5 * d * torch.log(torch.tensor(2 * 3.14159265)) - 0.5 * (z ** 2).sum(dim=1)
    log_p_good = log_p0 - div_sum
    return z, log_p_good


if __name__ == "__main__":
    import torch
    from models.vector_field_unet import VectorFieldUNet
    from config import DIM, BASE_CHANNELS, CHECKPOINT_PATH, NUM_STEPS

    model = VectorFieldUNet(dim=DIM, base_channels=BASE_CHANNELS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    z0 = torch.randn(16, DIM)
    x, log_p = solve(model, z0, num_steps=NUM_STEPS, compute_likelihood=True)

    print(f"x shape: {x.shape}")
    print(f"log p(x): min={log_p.min():.2f}, max={log_p.max():.2f}, mean={log_p.mean():.2f}")
    sorted_idx = torch.argsort(log_p, descending=True)
    for idx in sorted_idx:
        print(f"Sample {idx.item()}: S = {log_p[idx].item():.2f}")

