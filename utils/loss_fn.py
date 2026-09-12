import math

import torch


def get_statistics(prediction: torch.Tensor, dim: int = 1, mode: str = "ensemble", epsilon: float = 1e-9):
    if mode == "ensemble":
        mu, sigma = prediction.mean(dim=dim), prediction.std(dim=dim)
    elif mode == "parametric":
        mu, sigma = prediction.split(1, dim=dim)
        mu, sigma = mu.squeeze(dim=dim), sigma.squeeze(dim=dim)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    return mu, sigma + epsilon


def f_mse(observation: torch.Tensor, prediction: torch.Tensor, **kwargs) -> torch.Tensor:
    return (prediction - observation) ** 2


def f_mae(observation: torch.Tensor, prediction: torch.Tensor, **kwargs) -> torch.Tensor:
    return (prediction - observation).abs()


def f_gaussian_crps(observation: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
    sqrtPi, sqrtTwo = math.sqrt(math.pi), math.sqrt(2)
    sigma = sigma.clamp(min=1e-6)
    z = (observation - mu) / sigma
    phi = torch.exp(-z ** 2 / 2) / (sqrtTwo * sqrtPi)
    return sigma * (z * torch.erf(z / sqrtTwo) + 2 * phi - 1 / sqrtPi)


def f_kernel_crps(observation: torch.Tensor, ensemble: torch.Tensor, fair: bool = False, **kwargs) -> torch.Tensor:
    n_member = ensemble.shape[-1]
    coef = -1 / (n_member * (n_member - 1)) if fair else -1 / (n_member ** 2)
    absolute_error = torch.mean((ensemble - observation[..., None]).abs(), dim=-1)
    ens_var = torch.zeros(size=ensemble.shape[:-1], device=ensemble.device)
    for i in range(n_member):
        ens_var += torch.sum(torch.abs(ensemble[..., i, None] - ensemble[..., i + 1:]), dim=-1)
    return absolute_error + coef * ens_var


def _block_mean(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(x, kernel_size=factor, stride=factor)


def _expand(x: torch.Tensor, factor: int) -> torch.Tensor:
    # nearest neighbour, so a band is exactly the deviation from its block mean and no
    # interpolation crosses the periodic longitude seam
    return torch.nn.functional.interpolate(x, scale_factor=factor, mode="nearest")


def laplacian_pyramid(x: torch.Tensor, levels: int = 3) -> list:
    """Split a field into `levels` scale bands, finest first.

    Band i is the deviation of the field from its 2^(i+1) block mean, at resolution
    (h / 2^i, w / 2^i); the last entry is the remaining low-pass residual. Non-overlapping
    block means never straddle the longitude seam, so no wrap handling is needed.
    """
    bands, current = [], x
    for _ in range(levels - 1):
        coarse = _block_mean(current)
        bands.append(current - _expand(coarse, 2))
        current = coarse
    bands.append(current)
    return bands


def f_pyramid_mse(observation: torch.Tensor, prediction: torch.Tensor,
                  levels: int = 3, weights: list = None, **kwargs) -> torch.Tensor:
    """Squared error accumulated per scale band rather than per grid point.

    Plain MSE is minimised by predicting the conditional mean, so a model blurs away any
    structure it cannot predict; small scales carry little of the total variance and are the
    first to go. Scoring each band separately, with its own weight, makes that blurring
    expensive at the scales one cares about.

    Returns a field-shaped tensor, as the other losses here do: each band's squared error is
    expanded back to the full grid before the weighted sum, so the per-variable and area
    weights of the training step still apply.

    Note the bands are not orthogonal, so with uniform weights this is not equal to `f_mse`;
    the training curves of the two objectives are not comparable, only their validation MSE is.
    """
    weights = list(weights) if weights is not None else [1.0] * levels
    assert len(weights) == levels, f"{levels} levels need {levels} weights, got {len(weights)}"

    observed = laplacian_pyramid(observation, levels)
    predicted = laplacian_pyramid(prediction, levels)

    total = None
    for index, (o, p, weight) in enumerate(zip(observed, predicted, weights)):
        error = (p - o) ** 2
        if index:
            error = _expand(error, 2 ** index)
        error = weight * error
        total = error if total is None else total + error
    return total
