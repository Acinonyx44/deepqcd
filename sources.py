"""
The observation models used in the experiments, each bundled with the model-based detectors that know it.

A source is everything the paper's workflow needs about a data-generating process:

    training_set(n, T, tau)  labelled streams for the offline training phase (Alg. 1)
    sampler(n, tau)          a chunked generator of fresh streams for the real-time phase (Alg. 2)
    llr(x, x_prev)           log f1/f0, which only the model-based benchmarks are allowed to use
    benchmarks()             those benchmarks, with a threshold grid each

Sources are stateless apart from the samplers they hand out, and they never touch the global seed, so a
caller controls reproducibility by seeding once and then calling in a fixed order.

The three sources here all have a single persistent change at tau. The transient case (paper Sec. 5.3)
has two change-points and a different metric, so it stays in deepqcd_transient.py.
"""
import numpy as np

from qcd import Recursive, cusum, shiryaev, shiryaev_roberts


class IID:
    """Paper Sec. 5.1: f0 = N(0, I_p), f1 = N(1_p, I_p), independent over time."""

    name, label = 'iid', 'IID Gaussian, mean 0 -> 1'
    T_train, rho = 2000, 0.001
    unit = 'steps'

    def __init__(self, p=7):
        self.input_dim = p
        self.mu1 = np.ones(p)

    def observations(self, n, t, tau):
        """Observations at the (1-indexed) times t, plus the labels 1{t >= tau}."""
        post = t[None, :] >= tau[:, None]
        x = np.random.randn(n, len(t), self.input_dim) + post[..., None] * self.mu1
        return x.astype(np.float32), post.astype(np.float32)

    def training_set(self, n, T, tau):
        return self.observations(n, np.arange(1, T + 1), tau)

    def sampler(self, n, tau):
        tau = np.asarray(tau, dtype=float)
        return lambda t0, L: self.observations(n, np.arange(t0 + 1, t0 + L + 1), tau)[0]

    def llr(self, x, x_prev):
        return x @ self.mu1 - self.mu1 @ self.mu1 / 2

    def benchmarks(self):
        return [('Shiryaev', lambda h: Recursive(shiryaev(self.rho), self.llr, h), prob_grid()),
                ('CUSUM', lambda h: Recursive(cusum, self.llr, h), np.linspace(0.1, 16, 60)),
                ('Shiryaev-Roberts', lambda h: Recursive(shiryaev_roberts, self.llr, h),
                 np.logspace(0, 7, 60))]


class AR:
    """Paper Sec. 5.2: x_t = mu_i + lam_i x_{t-1} + N(0, 1), with both the drift and the correlation
    changing at tau. x_0 = 0. The benchmarks use the conditional LR of Eq. (10)."""

    name, label = 'ar', 'AR(1), drift 0 -> 1 and correlation -0.3 -> 0.2'
    T_train, rho = 2000, 0.001
    input_dim = 1
    unit = 'steps'

    def __init__(self, mu0=0.0, lam0=-0.3, mu1=1.0, lam1=0.2):
        self.mu0, self.lam0, self.mu1, self.lam1 = mu0, lam0, mu1, lam1

    def sampler(self, n, tau):
        tau = np.asarray(tau, dtype=float)
        x_prev = np.zeros(n)  # x_0 = 0

        def sample(t0, L):
            nonlocal x_prev
            x = np.empty((n, L))
            for i in range(L):
                post = t0 + i + 1 >= tau
                mu = np.where(post, self.mu1, self.mu0)
                lam = np.where(post, self.lam1, self.lam0)
                x_prev = mu + lam * x_prev + np.random.randn(n)
                x[:, i] = x_prev
            return x[..., None].astype(np.float32)

        return sample

    def training_set(self, n, T, tau):
        x = self.sampler(n, tau)(0, T)
        y = (np.arange(1, T + 1)[None, :] >= np.asarray(tau)[:, None]).astype(np.float32)
        return x, y

    def llr(self, x, x_prev):
        """Eq. (10): both conditional densities are Gaussian with unit variance and means
        mu_i + lam_i x_{t-1}, so the log-LR is (x - midpoint) * (difference of means)."""
        x = x[..., 0]
        xp = np.concatenate([x_prev, x[:, :-1]], axis=1)  # x_{t-1} for every t in the chunk
        return ((x - 0.5 * (xp * (self.lam0 + self.lam1) + self.mu0 + self.mu1))
                * (xp * (self.lam1 - self.lam0) + self.mu1 - self.mu0))

    def benchmarks(self):
        return [('mod. Shiryaev', lambda h: Recursive(shiryaev(self.rho), self.llr, h), prob_grid()),
                ('mod. CUSUM', lambda h: Recursive(cusum, self.llr, h), np.linspace(0.1, 14, 60)),
                ('mod. Shiryaev-Roberts', lambda h: Recursive(shiryaev_roberts, self.llr, h),
                 np.logspace(0, 7, 60))]


class GARCH:
    """Ours, not in the paper: daily returns with volatility clustering whose long-run volatility jumps
    from 1 to vol1 at tau,

        r_t = sigma_t z_t,   sigma_t^2 = omega_i + alpha r_{t-1}^2 + beta sigma_{t-1}^2,

    with omega_i set so the unconditional variance is 1 before the change and vol1^2 after. alpha and
    beta do not change, so the clustering looks the same in both regimes and an ordinary volatility
    cluster is exactly what a naive detector mistakes for a regime change.

    With sq=True the feature transformation (paper Sec. 4.1) hands the network [r_t, r_t^2].
    """

    name, label = 'garch', 'GARCH(1,1) returns, long-run vol 1 -> 2'
    T_train, rho = 2000, 0.001
    unit = 'days'

    def __init__(self, alpha=0.05, beta=0.90, vol1=2.0, sq=False, k_roll=20):
        self.alpha, self.beta, self.vol1, self.sq, self.k_roll = alpha, beta, vol1, sq, k_roll
        self.omega = np.array([1.0, vol1 ** 2]) * (1 - alpha - beta)
        self.input_dim = 2 if sq else 1
        if sq:
            self.name, self.label = 'garch-sq', self.label + ', features [r, r^2]'

    def features(self, r):
        f = np.stack([r, r ** 2], axis=-1) if self.sq else r[..., None]
        return f.astype(np.float32)

    def sampler(self, n, tau):
        tau = np.asarray(tau, dtype=float)
        r_prev, sig2 = np.zeros(n), np.ones(n)  # start at the calm regime's unconditional variance

        def sample(t0, L):
            nonlocal r_prev, sig2
            r = np.empty((n, L))
            for i in range(L):
                omega = np.where(t0 + i + 1 >= tau, self.omega[1], self.omega[0])
                sig2 = omega + self.alpha * r_prev ** 2 + self.beta * sig2
                r_prev = np.sqrt(sig2) * np.random.randn(n)
                r[:, i] = r_prev
            return self.features(r)

        return sample

    def training_set(self, n, T, tau):
        x = self.sampler(n, tau)(0, T)
        y = (np.arange(1, T + 1)[None, :] >= np.asarray(tau)[:, None]).astype(np.float32)
        return x, y

    def llr_iid(self, x, x_prev):
        """The practitioner's misspecified model: returns treated as IID N(0, 1) vs N(0, vol1^2).
        Right unconditional volatilities, wrong dynamics."""
        return 0.5 * x[..., 0] ** 2 * (1 - 1 / self.vol1 ** 2) - np.log(self.vol1)

    def llr(self, x, x_prev):
        """The GARCH-aware log-LR, as a stateful callable (see GARCHLLR)."""
        raise TypeError('use GARCHLLR(source): the GARCH log-LR carries state between chunks')

    def benchmarks(self):
        return [('GARCH Shiryaev', lambda h: Recursive(shiryaev(self.rho), GARCHLLR(self), h), prob_grid()),
                ('IID Shiryaev', lambda h: Recursive(shiryaev(self.rho), self.llr_iid, h), prob_grid()),
                ('GARCH CUSUM', lambda h: Recursive(cusum, GARCHLLR(self), h), np.linspace(0.2, 9, 60)),
                ('IID CUSUM', lambda h: Recursive(cusum, self.llr_iid, h), np.linspace(0.2, 16, 60)),
                (f'rolling var ({self.k_roll}d)', lambda h: RollingVariance(self.k_roll, h),
                 np.linspace(0.8, 4.0, 60))]


class GARCHLLR:
    """log N(r_t; 0, sigma_{1,t}^2) / N(r_t; 0, sigma_{0,t}^2), with each regime's variance recursion run
    in parallel from t = 1 and driven by the observed returns. This is the usual recursive approximation
    to the true change-point LR of a GARCH process (the exact one is not recursive, because the
    post-change variance depends on when the change happened). Stateful across chunks."""

    def __init__(self, source):
        self.src = source

    def reset(self, n):
        self.sig2 = np.ones((2, n))

    def __call__(self, x, x_prev):
        s = self.src
        r, out = x[..., 0], np.empty(x.shape[:2])
        r_prev = x_prev[:, 0]
        for t in range(r.shape[1]):
            self.sig2 = s.omega[:, None] + s.alpha * r_prev ** 2 + s.beta * self.sig2
            s0, s1 = self.sig2
            out[:, t] = 0.5 * r[:, t] ** 2 * (1 / s0 - 1 / s1) + 0.5 * np.log(s0 / s1)
            r_prev = r[:, t]
        return out


class RollingVariance:
    """d_t = mean of r^2 over the last K returns: window-limited and model-free, the kind of rule a desk
    actually runs. Fewer than K terms are available at the start, where the window is zero-padded."""

    def __init__(self, K, h):
        self.K, self.h = K, h

    def reset(self, n):
        self.tail = np.zeros((n, self.K - 1))

    def __call__(self, x):
        r2 = x[..., 0].astype(np.float64) ** 2
        ext = np.concatenate([self.tail, r2], axis=1)
        S = np.concatenate([np.zeros((len(r2), 1)), np.cumsum(ext, axis=1)], axis=1)
        end = np.arange(self.K, self.K + r2.shape[1])
        self.tail = ext[:, ext.shape[1] - (self.K - 1):]
        return (S[:, end] - S[:, end - self.K]) / self.K


def prob_grid(n=60):
    """Ascending thresholds in (0, 1) for detectors whose statistic is a posterior probability, spaced
    so that most of the points sit near 1, which is where the low-false-alarm operating points are."""
    return 1 - np.logspace(np.log10(0.99), -6, n)


SOURCES = {
    'iid': IID,
    'ar': AR,
    'garch': GARCH,
    'garch-sq': lambda: GARCH(sq=True),
}


def build(name):
    if name not in SOURCES:
        raise SystemExit(f'unknown source {name!r}; choose from {", ".join(SOURCES)}')
    return SOURCES[name]()
