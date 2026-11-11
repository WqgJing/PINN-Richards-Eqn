"""Loss weighting: gradient-norm balancing and causal time weights.

Every loss term is first divided by a physical scale sigma (so it reads as a negative
log-likelihood and starts at O(1)). On top of that:

GradNormBalancer (Wang, Teng & Perdikaris 2021, "learning-rate annealing"):
    every `freq` steps, g_k = || grad_theta L_k || over the network parameters.
    The reference term (the PDE) keeps weight 1; every other term gets
    lambda_hat_k = g_ref / g_k, EMA-smoothed, step change capped at x`max_step`, clipped
    to [w_min, w_max]. (An earlier variant equalised all terms to the mean gradient; it
    pushed the PDE weight to 0.03 and the network fitted the data without the physics.)

CausalWeights (Wang, Sankaran & Perdikaris 2022):
    the window is cut into M time chunks; chunk i gets w_i = exp(-eps * sum_{j<i} L_j)
    (detached), so later times count only once earlier times are solved. The losses are
    sigma-normalised, so a chunk is "solved" when L_j ~ 1; eps ~ 1/M then gives weights
    near exp(-1) at the last chunk once everything is solved, and ~0 while early chunks
    are far from it. eps grows x10 when min(w) > 0.99, up to eps_max.
"""
import math
import torch


class GradNormBalancer:
    def __init__(self, names, freq=100, alpha=0.9, max_step=2.0, w_min=1e-2, w_max=1e2, fixed=False,
                 ref='pde'):
        self.names = list(names)
        self.w = {k: 1.0 for k in self.names}
        self.freq, self.alpha, self.max_step = int(freq), float(alpha), float(max_step)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.fixed = bool(fixed)
        self.ref = ref

    def maybe_update(self, step, losses, params):
        if self.fixed or step % self.freq != 0:
            return False
        g = {}
        for k in self.names:
            if k not in losses or not losses[k].requires_grad:
                continue
            grads = torch.autograd.grad(losses[k], params, retain_graph=True, allow_unused=True)
            g[k] = math.sqrt(sum(float((x.detach() ** 2).sum()) for x in grads if x is not None)) + 1e-30
        if len(g) < 2 or self.ref not in g:
            return False
        for k, gk in g.items():
            if k == self.ref:
                continue
            target = g[self.ref] / gk
            new = self.alpha * self.w[k] + (1.0 - self.alpha) * target
            new = min(max(new, self.w[k] / self.max_step), self.w[k] * self.max_step)
            self.w[k] = min(max(new, self.w_min), self.w_max)
        return True


class CausalWeights:
    def __init__(self, t_max_days, n_chunks=16, eps=1.0, eps_max=100.0, enabled=True):
        self.t_max = float(t_max_days)
        self.M = int(n_chunks)
        self.eps = float(eps)
        self.eps_max = float(eps_max)
        self.enabled = bool(enabled)
        self.last_min_w = 1.0

    def weighted_mean(self, r2, t_days):
        """r2: squared residuals (N,1). Returns causal-weighted mean."""
        if not self.enabled:
            return r2.mean()
        idx = torch.clamp((t_days.detach() / self.t_max * self.M).long(), 0, self.M - 1).reshape(-1)
        sums = torch.zeros(self.M, dtype=r2.dtype, device=r2.device).index_add(0, idx, r2.reshape(-1))
        cnt = torch.zeros(self.M, dtype=r2.dtype, device=r2.device).index_add(
            0, idx, torch.ones_like(r2.reshape(-1)))
        chunk = sums / torch.clamp(cnt, min=1.0)
        cum = torch.cat([torch.zeros(1, dtype=r2.dtype, device=r2.device), torch.cumsum(chunk.detach(), 0)[:-1]])
        w = torch.exp(-self.eps * cum)
        self.last_min_w = float(w.min())
        return (w * chunk).mean()

    def maybe_anneal(self):
        if self.enabled and self.last_min_w > 0.99 and self.eps < self.eps_max:
            self.eps = min(self.eps * 10.0, self.eps_max)
            return True
        return False


class DiscrepancyBalancer:
    """Morozov discrepancy principle for the data-type terms; the PDE is the objective (w = 1).

    The physics is exact up to the network's approximation error, the observations are not: each
    data-type term (theta data, surface flux) is sigma-normalised by its known noise level, so its
    noise-level misfit is L_k = 1. Fitting it more tightly than that is over-fitting, and the excess
    is paid for by a physics violation (seen in E2: theta RMSE 2e-5 << 0.005 with zb off by 9 cm).
    So each term is held at its noise level, L_k -> target_k (= 1), by dual ascent on its weight in
    log space (a weight is a Lagrange multiplier of the constraint L_k <= target_k):
        log w_k <- log w_k + eta * log(EMA(L_k) / target_k),   step capped at log(max_step)
    The PDE residual is minimised subject to these constraints (Engl, Hanke & Neubauer 1996, ch. 4;
    Morozov 1966). The only inputs are the physical noise levels sigma_k; eta and the update period
    are algorithm parameters of the dual ascent, not problem weights.
    """

    def __init__(self, names, freq=100, eta=0.5, beta=0.9, max_step=2.0, w_min=1e-4, w_max=1e6,
                 targets=None, ref='pde'):
        self.names = list(names)
        self.ref = ref
        self.w = {k: 1.0 for k in self.names}
        self.freq, self.eta, self.beta = int(freq), float(eta), float(beta)
        self.max_log = math.log(float(max_step))
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.targets = {k: 1.0 for k in self.names if k != ref}
        self.targets.update(targets or {})
        self.ema = {}
        self.fixed = False

    def maybe_update(self, step, losses, params=None):
        for k, v in losses.items():
            if k == self.ref or k not in self.targets:
                continue
            x = float(v.detach())
            self.ema[k] = x if k not in self.ema else self.beta * self.ema[k] + (1 - self.beta) * x
        if step % self.freq != 0 or step == 0:
            return False
        for k, x in self.ema.items():
            d = self.eta * math.log(max(x, 1e-300) / self.targets[k])
            d = max(-self.max_log, min(self.max_log, d))
            self.w[k] = min(max(self.w[k] * math.exp(d), self.w_min), self.w_max)
        return True
