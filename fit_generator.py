"""Fitting the quantile generator.

The default objective is the randomised pinball loss, which is a strictly
consistent scoring function for the conditional quantile pointwise in the
level, so minimising it over levels targets the whole conditional law with one
observation per conditioning window.  The alternative objective is an MMD
between path blocks, which is what the earlier architecture was trained on; it
is kept so that "same model class, same diagnostics, pinball against MMD" can
be run rather than argued about.  It is a PATH-LAW criterion and has no
conditional content, which is the point of the comparison.

The Lipschitz penalty
---------------------
`lip_mode="penalty"` adds lam * relu(sum_i sup|d qhat/d z_i| - L_max)^2,
estimated by autograd on the batch.  Three regions, and the choice matters more
than it looks:

  "box"   uniform windows over the whole state space.  The hinge is a maximum
          over the batch, so on uniform draws it fires on points the recursion
          never visits and, at a small training budget, drives the fitted map
          flatter than the truth.  That was the main cause of the tail problem:
          with L_max = 0.85 and lam = 5 the fitted sum of lag sensitivities came
          out at 0.62 against a target S of 0.705, and the generated tails and
          the volatility clustering were both short.
  "hood"  windows drawn near the data, data plus pen_sigma * N(0,I), clipped.
          The default.  It constrains the map where the recursion actually goes
          and costs nothing.
  "data"  the training windows only.

`lip_mode="project"` rescales G after each step using the analytic bound, which
is valid but very conservative.  `"none"` imposes nothing, and is forced when
the target itself has S >= 1.

Budget
------
The clustering diagnostic is budget-sensitive: at T = 60k the lag-1
autocorrelation of squares on ARCH reaches about 0.88 of the target's at 300
epochs and markedly less below that.  Epochs and path length are interchangeable
at equal sample-visits.
"""
import time

import numpy as np
import torch

from generators.quantile_generator import QuantileNet
from loss.pinball import pinball, sample_levels
from loss.mmd import mmd2_torch

__all__ = ["grad_lip", "fit", "unroll_blocks"]


def grad_lip(net, z, u):
    """Differentiable sum_i max_batch |d qhat / d z_i|.

    This is the quantity the uniqueness condition is about.  The closed-form
    bound sum_j |a_j| ||G_j||_1 bounds the same thing worst-case and is usually
    vacuous, which is why the penalty uses this instead.
    """
    z = z.detach().requires_grad_(True)
    out = net(u, z).sum()
    gr, = torch.autograd.grad(out, z, create_graph=True)
    return gr.abs().max(dim=0).values.sum()


def unroll_blocks(net, n_blocks, block_len, burn, gen, M=1.0):
    """Differentiably unroll `n_blocks` chains and return the last
    `block_len` states of each, shape (n_blocks, block_len).

    Used only by the MMD objective: to score the law the generator induces you
    have to run the closed recursion, and to train on that score you have to
    backpropagate through it.  Keep `burn` short; the cost is (burn+block_len)
    forward passes per optimiser step.
    """
    m = net.m
    z = torch.zeros(n_blocks, m)
    out = []
    for t in range(burn + block_len):
        u = torch.rand(n_blocks, generator=gen)
        x = net(u, z)
        if t >= burn:
            out.append(x)
        z = torch.cat([x[:, None], z[:, :-1]], dim=1)
    return torch.stack(out, dim=1)


def fit(Z, X, m, *, objective="pinball", width=256, act="softclip",
        seed_net=0, v_mode="u", v_clip=2.5, epochs=300, batch=512, lr=3e-3,
        lip_mode="penalty", pen_region="hood", pen_sigma=0.15, L_max=0.95,
        lam=1.0, tail_frac=0.2, edge=0.02, u_min=1e-4, M=1.0,
        mmd_block=4, mmd_batch=256, mmd_burn=32,
        verbose=False, log_every=25, n_monitor=512, net=None):
    """Fit a `QuantileNet` and return (net, history).

    `history` records, per epoch, the training loss and the measured
    sum_i Lip_i on the state space, which is the quantity the penalty targets
    and the one to compare with the target's S_m afterwards.
    """
    torch.manual_seed(seed_net)
    gen = torch.Generator().manual_seed(seed_net)
    Zt = torch.as_tensor(np.asarray(Z), dtype=torch.float32)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n = len(Xt)
    if net is None:
        net = QuantileNet(m, width=width, act=act, seed=seed_net,
                          v_mode=v_mode, v_clip=v_clip)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    hist = []
    t0 = time.time()

    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            z, x = Zt[idx], Xt[idx]
            u = sample_levels(len(idx), gen, tail_frac=tail_frac, edge=edge,
                              u_min=u_min)

            if objective == "pinball":
                loss = pinball(net(u, z), x, u)
            elif objective == "mmd":
                real = x.unfold(0, mmd_block, 1) if len(x) > mmd_block else None
                if real is None or len(real) < 8:
                    continue
                k = min(mmd_batch, len(real))
                sel = torch.randperm(len(real), generator=gen)[:k]
                fake = unroll_blocks(net, k, mmd_block, mmd_burn, gen, M=M)
                loss = mmd2_torch(real[sel], fake, generator=gen)
            else:
                raise ValueError(f"unknown objective {objective!r}")

            if lip_mode == "penalty":
                nb_ = len(idx)
                if pen_region == "box":
                    zp = torch.rand(nb_, net.m, generator=gen) * 2 * M - M
                elif pen_region == "hood":
                    j = torch.randint(0, n, (nb_,), generator=gen)
                    zp = (Zt[j] + pen_sigma
                          * torch.randn(nb_, net.m, generator=gen)).clamp(-M, M)
                elif pen_region == "data":
                    zp = z[:0]
                else:
                    raise ValueError(f"unknown pen_region {pen_region!r}")
                zc = torch.cat([z, zp])
                uc = torch.cat([u, sample_levels(len(zc) - len(u), gen,
                                                 tail_frac=tail_frac,
                                                 edge=edge, u_min=u_min)])
                loss = loss + lam * torch.relu(grad_lip(net, zc, uc) - L_max) ** 2

            opt.zero_grad()
            loss.backward()
            opt.step()

            if lip_mode == "project":
                with torch.no_grad():
                    B = float(net.lipschitz_bound())
                    if B > L_max:
                        net.rescale_G(L_max / B)

            tot += float(loss.item())
            nb += 1

        sched.step()
        zmon = torch.rand(n_monitor, net.m, generator=gen) * 2 * M - M
        umon = torch.rand(n_monitor, generator=gen)
        lip = float(grad_lip(net, zmon, umon).detach())
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "lip_box": lip,
                     "lip_analytic": float(net.lipschitz_bound().detach())})
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}   loss {tot/max(nb,1):.5f}   "
                  f"sum_i Lip_i on the state space {lip:.3f}")

    return net, {"history": hist,
                 "final_loss": hist[-1]["loss"] if hist else float("nan"),
                 "final_lip_box": hist[-1]["lip_box"] if hist else float("nan"),
                 "objective": objective, "epochs": int(epochs),
                 "lip_mode": lip_mode, "secs": time.time() - t0}
