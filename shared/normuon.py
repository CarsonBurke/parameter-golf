"""
NorMuon: Muon with per-row adaptive learning rate normalization.
Based on https://github.com/zichongli5/NorMuon (arxiv 2510.05491)

Drop-in replacement for Muon. Adds post-Newton-Schulz per-row adaptive
scaling with norm preservation. One extra state buffer per param (fan_out, 1).
"""

import torch
import torch.distributed as dist
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def normuon_post_ns(g: Tensor, second_momentum: Tensor, beta2: float) -> Tensor:
    """Per-row adaptive scaling after Newton-Schulz, with norm preservation."""
    g32 = g.float()
    vnorm = g32.norm()
    v_mean = g32.mul(g32).mean(dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    g32.div_(second_momentum.sqrt().add_(1e-10))
    g32.mul_(vnorm / g32.norm().add_(1e-10))
    return g32.bfloat16()


class NorMuon(torch.optim.Optimizer):
    """Muon + per-row adaptive normalization. API-compatible with baseline Muon."""

    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, beta2: float = 0.95):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, beta2=beta2),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            beta2 = group["beta2"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["second_momentum"] = torch.zeros(
                            g.size(0), 1, device=g.device, dtype=torch.float32
                        )
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g = normuon_post_ns(g, state["second_momentum"], beta2)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss
