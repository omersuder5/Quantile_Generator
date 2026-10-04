"""Where the tensors live, and in what precision.

One resolver, used by everything, so that the device and dtype are decided in
exactly one place and every module agrees.

    from backend import resolve, DEV
    dev, dt = resolve("auto", "float32")

WHAT A GPU ACTUALLY BUYS HERE
-----------------------------
There are two axes in generation and they behave oppositely, so the blanket
statement "generation cannot be parallelised" is wrong and is not made here.

* **along a chain, strictly sequential.**  x_t is a function of x_{t-1},...,
  x_{t-m}, so the T steps of one path must be taken in order.  Nothing can
  change this; it is the definition of the recursion.
* **across chains, embarrassingly parallel.**  B chains driven by independent
  (or, for the spread diagnostic, shared) innovations are B rows of one state
  matrix, advanced by one matrix product per time step.  `q_batch` does exactly
  that and `initial_spread` uses it.

So the right question is never "can generation be parallelised" but "does the
quantity wanted take the form of many chains, or of one long path", and both
occur here:

  many chains   the MMD objective (`unroll_blocks`) and the synchronisation
                spread (`initial_spread`).  Batched, in torch and numpy
                respectively.
  one path      the free-running evaluation path, the shared-innovation
                coupling, the autocorrelations and the Lyapunov average.
                These are estimates along a single stationary trajectory, and
                cutting it into B short pieces would change the estimand, not
                just the arithmetic: B burnt-in chains of length T/B give the
                marginal more cheaply but cannot give the lag-h dependence for
                h near T/B, and the whole point of the ACF panel is the
                dependence.  So a single path it is, for a statistical reason
                and not a technical one.

Measured on this package, width 256, numpy float64:

  one chain, scalar step               14.3 us/step, and the SAME 14.3 at m = 2
                                       and at m = 16, which says the cost is
                                       interpreter and call overhead, not the
                                       256 x m matrix product
  24 chains, batched                   59-109 us/step for all 24, i.e. 2.5-4.5
                                       us per chain-step: 3x to 5x the
                                       throughput of the loop
  1000 chains, batched                 3.3 us per chain-step at m = 2

The gain from batching therefore saturates around 4x, because once B x 256 is
large the elementwise activation is genuine arithmetic rather than overhead.
And 14 us/step on a CPU is the same order as a single GPU kernel launch, which
is why a sequential chain is not a GPU problem however it is written: the
batched step is where a GPU has anything to contribute, and that is the step
`unroll_blocks` already takes.

Expect the GPU to shorten the fit.  Whether it shortens it at all depends on
the width: at `width = 256`, `batch = 512` the matrices are small enough that
launch overhead can swallow the gain, and the Lipschitz penalty needs a second
backward pass through `autograd.grad`, which has its own fixed cost.  Measure
before believing; `width >= 1024` or a larger batch is where a GPU starts to
pay.  `ClosedQuantileRecursion` extracts the weights to numpy float64 at
construction and never touches torch again, which is a deliberate choice about
where the sequential work belongs and not a missing GPU path.

APPLE SILICON
-------------
On a Mac `torch.cuda.is_available()` is always False: the GPU is reached
through MPS, not CUDA.  And **MPS does not implement float64 at all**, so
`device="mps"` with `dtype="float64"` cannot work.  `resolve` reports that
rather than failing later inside an operator, and `auto` will not pick MPS when
float64 is asked for.

PRECISION
---------
float32 is the default and is what every number quoted in the README and in
the earlier study was produced with.  float64 costs roughly a factor of two on
a CPU; on an NVIDIA card it is half the float32 throughput on a datacentre
part (A100, H100, V100) and a thirty-second or a sixty-fourth of it on a
consumer one (RTX, GeForce), so there it is a real decision rather than a free
upgrade.  It is worth switching on when a diagnostic sits near the precision
floor, which here means the ESN realisation check and nothing else, and even
that already lands at about 4e-16 in float32 because the recursion side is
numpy float64 regardless.

**Changing the dtype does not change which fit you get.**  Every random draw
is made on the CPU through the seeded generator at the default float32 and
only then cast, so the stream of levels, permutations and penalty windows is
identical across devices and precisions.  A float64 run is the same fit
computed more precisely, agreeing with its float32 twin to about seven decimal
places on the loss, not a different experiment.  The same holds across
devices: a CUDA run reproduces a CPU run, because a CUDA generator would have
its own stream and is deliberately not used.
"""
import warnings

import torch

__all__ = ["resolve", "guard", "describe", "to_t", "to_np", "DTYPES"]

DTYPES = {"float32": torch.float32, "float64": torch.float64,
          "f32": torch.float32, "f64": torch.float64,
          "32": torch.float32, "64": torch.float64}


def _available():
    cuda = torch.cuda.is_available()
    mps = (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return cuda, bool(mps)


def resolve(device="auto", dtype="float32", verbose=False):
    """Return (torch.device, torch.dtype) from the config strings.

    `device` is "auto", "cpu", "cuda", "cuda:N", "mps", or a torch.device.
    "auto" prefers CUDA, then MPS, then CPU, but skips MPS when float64 is
    requested because MPS has no float64.

    Anything unavailable falls back to CPU with a warning rather than raising,
    so a sweep started on a machine without the requested accelerator still
    runs instead of dying on configuration number one.
    """
    if isinstance(dtype, torch.dtype):
        dt = dtype
    else:
        key = str(dtype).lower().replace("torch.", "")
        if key not in DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}; choose from "
                             f"{sorted(set(DTYPES))}")
        dt = DTYPES[key]

    cuda, mps = _available()

    if isinstance(device, torch.device):
        dev = device
    else:
        spec = str(device).lower()
        if spec == "auto":
            if cuda:
                dev = torch.device("cuda")
            elif mps and dt is not torch.float64:
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(spec)

    dev = _sanity(dev, dt, stacklevel=3)

    if verbose:
        print(describe(dev, dt))
    return dev, dt


def _sanity(dev, dt, stacklevel=2):
    """Replace an unavailable or incompatible device by the CPU, with a warning."""
    cuda, mps = _available()
    if dev.type == "cuda" and not cuda:
        warnings.warn("CUDA was requested but is not available"
                      + (" (this is a Mac: the GPU is reached through MPS, "
                         "never CUDA)" if mps else "")
                      + "; falling back to CPU.", RuntimeWarning,
                      stacklevel=stacklevel)
        return torch.device("cpu")
    if dev.type == "mps":
        if not mps:
            warnings.warn("MPS was requested but is not available; falling "
                          "back to CPU.", RuntimeWarning, stacklevel=stacklevel)
            return torch.device("cpu")
        if dt is torch.float64:
            warnings.warn("MPS does not implement float64. Running on the CPU "
                          "instead; set dtype='float32' to use the GPU.",
                          RuntimeWarning, stacklevel=stacklevel)
            return torch.device("cpu")
    return dev


def guard(device, dtype):
    """Correct an incompatible (device, dtype) pair without resolving "auto".

    Both arguments may be None, meaning "leave it as it is"; the pair is only
    ever weakened, never strengthened, so asking for float64 alone never moves
    anything onto a GPU.  The one combination that has to be corrected is MPS
    with float64, which no amount of care downstream can make work: MPS has no
    float64 kernels, and even a cast whose destination is the CPU is performed
    on the source device and so raises there.  Caught here, at construction,
    rather than ten minutes into a fit inside an operator.
    """
    dt = dtype
    if dt is not None and not isinstance(dt, torch.dtype):
        key = str(dt).lower().replace("torch.", "")
        if key not in DTYPES:
            raise ValueError(f"unknown dtype {dt!r}; choose from "
                             f"{sorted(set(DTYPES))}")
        dt = DTYPES[key]
    if device is None:
        return None, dt
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    return _sanity(dev, dt if dt is not None else torch.get_default_dtype(),
                   stacklevel=3), dt


def describe(dev, dt):
    cuda, mps = _available()
    bits = [f"device {dev}", f"dtype {str(dt).replace('torch.', '')}"]
    if dev.type == "cuda":
        bits.append(torch.cuda.get_device_name(dev))
    bits.append(f"(cuda available: {cuda}, mps available: {mps})")
    return "  ".join(bits)


def to_t(x, dev, dt):
    """numpy or tensor -> tensor on (dev, dt)."""
    if torch.is_tensor(x):
        return x.to(device=dev, dtype=dt)
    return torch.as_tensor(x, dtype=dt, device=dev)


def to_np(t):
    """tensor -> numpy, from any device, in float64.

    Every consumer of a fitted network on the numpy side wants float64: the
    closed recursion runs in it, and the diagnostics compare against analytic
    quantities computed in it.

    The order of the two steps matters and is not interchangeable.  `.to("cpu",
    torch.float64)` casts on the *source* device and then moves, so on MPS it
    raises `TypeError: Cannot convert a MPS Tensor to float64 dtype` even though
    the destination is the CPU, where float64 is perfectly ordinary.  Move
    first, cast second.  Every tensor-to-numpy conversion in this package goes
    through this function so that the order cannot be got wrong in one place
    and right in another.
    """
    return t.detach().cpu().to(torch.float64).numpy()
