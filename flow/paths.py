import torch 

#sigma_t = (1-t) + t*sigma (standard deviation)
def sigma_t(t: torch.Tensor, sigma: float)->torch.Tensor:
    return (1-t)+t*sigma

#f_t(y|x) dy = N(t*x, sigma_t^2 * I) (conditional probability path)
def sample_conditional_path(x: torch.Tensor, t: torch.Tensor, sigma: float):
    epsilon=torch.randn_like(x) #epsilon ~ N(0,I)
    st = sigma_t(t, sigma)

    #sample by reparametrization
    y=t*x+st*epsilon

    return y, epsilon

#xi(y,t|x) = sigma-1 / ((1-t)+t*sigma) * (y - t*x) + x (conditional vector field)
def conditional_vector_field(y: torch.Tensor, t: torch.Tensor, x: torch.Tensor, sigma: float) -> torch.Tensor:
    st = sigma_t(t, sigma)
    return x + ((sigma - 1.0) / st) * (y - t * x)

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 8
    dim = 784
    sigma = 0.01

    x = torch.randn(batch_size, dim)
    t = torch.rand(batch_size, 1)

    y, epsilon = sample_conditional_path(x, t, sigma)
    xi = conditional_vector_field(y, t, x, sigma)

    # sanity check: at t=1, m_1(x)=x and sigma_1=sigma (small),
    # so y should lie close to x
    t_ones = torch.ones(batch_size, 1)
    y_at_one, _ = sample_conditional_path(x, t_ones, sigma)

    diff = (y_at_one - x).abs().max()
    print(f"y shape: {y.shape}")
    print(f"x shape: {x.shape}")
    print(f"t shape: {t.shape}")
    print(f"t values: {t.squeeze()}")
    print(f"Max deviation |y - x| at t=1")
