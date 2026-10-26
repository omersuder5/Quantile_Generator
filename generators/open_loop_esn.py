"""An OPEN-LOOP echo state network, trained on a generator's own innovations.

This is the object of `conj:approx`, and it is a different object from
`esn_realization.ESNRealization`, so the difference is worth stating first.

    exact-shift realisation (prop:exactshift)     open-loop ESN (this module)
    ------------------------------------------     ---------------------------
    weights DERIVED from the fitted network        reservoir RANDOM, fixed
    part of the state is its own past outputs      state sees the innovations only
    closed loop: x_t fed back                      open loop: nothing fed back
    N = W + (m-1) + 1, fixed by the network        N free
    error 1e-16, by construction                   error is a genuine approximation
    needs sigma affine on an interval              any bounded nonconstant sigma

The open-loop network is

    h_t = sigma(A h_{t-1} + C v_t + zeta),     v_t = 2 u_t - 1,
    y_t = w' h_t + b,

driven by the innovations u_t alone.  It never sees x.  Only the readout (w, b)
is trained, by least squares against the generator's output

    xhat_t = Psi_qhat(u_t, u_{t-1}, ...),

which is well defined because the generator is contractive: the closed
recursion has a unique solution, and that solution is a measurable function of
the innovation history.  So the ESN is asked to learn a FILTER of the
innovations, not a recursion: exactly the setting of Gonon and Ortega's L^p
universality theorem, with H = Psi_qhat and Z = the uniform innovations.

One thing the open loop gives for free and one thing it does not.

  free     y_t is a function of u_{<=t}, so the ESN output is adapted to the
           innovations, and the synchronous coupling (same u for both) is
           causal from xhat to y: u_{<=t} is recoverable from xhat_{<=t}
           because qhat is increasing in u.
  not free the reverse.  Nothing makes the ESN's output invertible, so u_{<=t}
           need not be recoverable from y_{<=t}, and the synchronous coupling
           need not be bicausal.  The L^p error eps between xhat and y is
           therefore NOT by itself an adapted-Wasserstein bound.  What converts
           it is Lemma A in two steps: the shared coupling proves a KERNEL bound,
           kappa_p <= (1 + S') eps_p (a statement about laws), and the
           Knothe-Rosenblatt coupling through the ESN's own innovations, which
           is bicausal, costs at most kappa_p / (1 - S').  `utils.realisation`
           bounds kappa_1 from below with PIT residuals, so the first step can
           be checked from both sides.

Two reservoirs are offered, because conj:approx(ii) is a claim about which one
can reach which rate.

  "random"      A Gaussian, rescaled to spectral radius `rho`.  The standard
                ESN.  By the circular law its eigenvalues fill a disc; only
                about sqrt(2N/pi) of them are real, spread uniformly over
                [-rho, rho], and the rest oscillate.  So it has almost no slow,
                monotone modes, whatever rho: good at nonlinear, short-memory
                filters, and unable to carry a power-law memory (Appendix C.4:
                with rho = 0.9, 0.99 or 0.999 and either input scale it stays
                at 4e-2 to 8e-2 relative error at N = 512, where truncation
                reaches 4e-3; the multiscale reservoir is at 2.4e-4 from
                N = 128 on).
  "multiscale"  A diagonal, with time constants tau_k spaced GEOMETRICALLY from
                1 to `tau_max`, i.e. lambda_k = exp(-1/tau_k).  With softclip
                and the input scaled into its affine window the reservoir is
                exactly linear, and the readout is then a sum of N decaying
                exponentials with fixed geometric rates: a quadrature, in
                log-rate, of the Laplace representation of a power law.  At
                fixed tau_max the step is log(tau_max)/N, so the error falls
                like exp(-c N / log tau_max) down to a floor; exp(-c sqrt N) is
                the lower envelope over tau_max.  This is the reservoir
                conj:approx(ii) is about.  Keep tau_max well below the training
                length: a unit with time constant tau sees about T/tau
                effectively independent samples.
"""
import numpy as np

__all__ = ["OpenLoopESN"]


def _softclip(y, beta=0.5):
    a = np.abs(y)
    sat = 1.0 + beta * (1.0 - np.exp(-np.maximum(a - 1.0, 0.0) / beta))
    return np.where(a <= 1.0, y, np.sign(y) * sat)


_ACT = {"tanh": np.tanh, "softclip": _softclip}


class OpenLoopESN:
    """Open-loop ESN with a fixed reservoir and a ridge-regression readout.

    Parameters
    ----------
    N : int
        Reservoir dimension.
    reservoir : "random" | "multiscale"
    rho : float
        Spectral radius of A ("random" only).  < 1 for the echo state property.
    input_scale, bias_scale : float
        Scales of C and zeta ("random" only).
    act : "tanh" | "softclip" | None
        None picks tanh for "random" and softclip for "multiscale", where the
        scaling below keeps every pre-activation inside [-1, 1] so softclip is
        the identity and the reservoir is exactly linear.
    tau_max : float
        Longest time constant ("multiscale" only).
    seed : int
    """

    def __init__(self, N, reservoir="random", rho=0.9, input_scale=0.6,
                 bias_scale=0.3, act=None, tau_max=2000.0, seed=0):
        self.N, self.reservoir = int(N), reservoir
        rng = np.random.default_rng(seed)
        if reservoir == "random":
            A = rng.standard_normal((self.N, self.N)) / np.sqrt(self.N)
            r = float(np.max(np.abs(np.linalg.eigvals(A))))
            self.A = A * (float(rho) / r)
            self.C = rng.uniform(-1, 1, self.N) * float(input_scale)
            self.zeta = rng.uniform(-1, 1, self.N) * float(bias_scale)
            self.diag = None
            self.act_name = act or "tanh"
        elif reservoir == "multiscale":
            tau = np.geomspace(1.0, float(tau_max), self.N)
            lam = np.exp(-1.0 / tau)
            self.diag = lam
            self.A = None
            # |h_k| <= 0.9 for every input in [-1, 1]: a normalised leaky
            # integrator, so the activation never leaves its affine window
            self.C = 0.9 * (1.0 - lam)
            self.zeta = np.zeros(self.N)
            self.act_name = act or "softclip"
            self.tau = tau
        else:
            raise ValueError(f"unknown reservoir {reservoir!r}")
        self.sigma = _ACT[self.act_name]
        self.w = self.b = self.mu = self.sd = None
        self.ridge = None

    # -- the reservoir ---------------------------------------------------------
    def states(self, u, h0=None):
        """Reservoir states for an innovation stream u in [0,1], shape (T, N)."""
        v = 2.0 * np.asarray(u, dtype=float) - 1.0
        T = len(v)
        H = np.empty((T, self.N))
        h = np.zeros(self.N) if h0 is None else np.asarray(h0, float).copy()
        if self.diag is not None:
            lam, C, zt, sg = self.diag, self.C, self.zeta, self.sigma
            for t in range(T):
                h = sg(lam * h + C * v[t] + zt)
                H[t] = h
        else:
            A, C, zt, sg = self.A, self.C, self.zeta, self.sigma
            for t in range(T):
                h = sg(A @ h + C * v[t] + zt)
                H[t] = h
        return H

    def states_many(self, u, H0):
        """States for several initial conditions under ONE stream, (n, T, N).
        Used for the echo state property check."""
        v = 2.0 * np.asarray(u, dtype=float) - 1.0
        h = np.asarray(H0, float).copy()                # (n, N)
        out = np.empty((h.shape[0], len(v), self.N))
        for t in range(len(v)):
            pre = (h * self.diag if self.diag is not None else h @ self.A.T)
            h = self.sigma(pre + self.C[None, :] * v[t] + self.zeta[None, :])
            out[:, t] = h
        return out

    # -- the readout -------------------------------------------------------------
    def fit(self, u, x, washout=500, ridges=None, val_frac=0.2):
        """Ridge readout on standardised states, ridge chosen on a contiguous
        validation block at the end of the training stream, then refitted on
        all of it.  `x` is aligned with `u`; the first `washout` steps are
        dropped so the reservoir has forgotten its zero start.  `ridges=None`
        uses the grid of `fit_states`.  Returns self."""
        kw = {} if ridges is None else {"ridges": ridges}
        return self.fit_states(self.states(u)[washout:],
                               np.asarray(x, float)[washout:],
                               val_frac=val_frac, **kw)

    def fit_states(self, H, y, ridges=(0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6,
                                        1e-4, 1e-2), val_frac=0.2):
        """`fit` on precomputed states, so one reservoir run can serve the
        training block, the test block and the diagnostics.

        Solved through an SVD of the standardised state matrix, never through
        the normal equations.  The multiscale reservoir's neighbouring leaky
        integrators are nearly collinear, and H'H squares the condition number:
        the normal equations floored the error at a level set by float64, not
        by the approximation, identically for every tau_max.  The SVD route
        works at the square root of that condition number.
        """
        y = np.asarray(y, float)
        self.mu = H.mean(0)
        self.sd = H.std(0) + 1e-12
        Hs = (H - self.mu) / self.sd
        ym = float(y.mean())
        yc = y - ym
        n_tr = int(len(y) * (1 - val_frac))

        def ridge_path(Hm, yv):
            U, s, Vt = np.linalg.svd(Hm, full_matrices=False)
            Uty = U.T @ yv
            cut = s.max() * 1e-15 * max(Hm.shape)          # numerical rank
            def w(lam):
                f = np.where(s > cut, s / (s ** 2 + lam * len(yv)), 0.0)
                return Vt.T @ (f * Uty)
            return w

        w_tr = ridge_path(Hs[:n_tr], yc[:n_tr])
        best = None
        for lam in ridges:
            wl = w_tr(lam)
            err = float(np.sqrt(np.mean((Hs[n_tr:] @ wl - yc[n_tr:]) ** 2)))
            if best is None or err < best[0]:
                best = (err, lam)
        self.ridge = best[1]
        self.w = ridge_path(Hs, yc)(self.ridge)
        self.b = ym
        return self

    def predict(self, u, h0=None):
        H = self.states(u, h0=h0)
        return ((H - self.mu) / self.sd) @ self.w + self.b

    def predict_from_states(self, H):
        return ((H - self.mu) / self.sd) @ self.w + self.b

    def describe(self):
        d = {"N": self.N, "reservoir": self.reservoir, "act": self.act_name,
             "ridge": self.ridge}
        if self.reservoir == "random":
            d["spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(self.A))))
        else:
            d["tau_range"] = [float(self.tau[0]), float(self.tau[-1])]
        return d
