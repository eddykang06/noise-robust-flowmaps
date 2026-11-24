"""Golden outputs of the official iMF training objective, for tests/test_imf_reference.py.

Runs the official JAX `iMeanFlow.forward` (upstream_imf_jax.py, an unchanged copy of
github.com/Lyy-iiis/imeanflow `imf.py` at bf60cd7) with a small closed-form network in
place of imfDiT and every random draw replayed from fixed arrays, then saves the loss,
its two terms and the parameter gradients. Needs jax and flax (tests/reference/
requirements-jax.txt), not torch:

    python tests/reference/make_imf_reference.py   # writes tests/reference/imf_reference.npz

The network, shared with the torch test, is
    f = tanh(x A + t b_t + h b_h + (1 - 1/omega) b_w + t_min b_s + t_max b_e + E[y])
    u = f B_u,  v = f B_v        (x flattened; u, v reshaped back)
It reads t as well as h, so the test also covers the JVP tangent in t.
"""

import sys
import types
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
N, D, H, K = 8, 4, 16, 3          # batch, flattened data dim, hidden width, classes (K = null)
SHAPE = (N, 2, 2, 1)              # upstream's (B, H, W, C) layout


def parameters(rng):
    p = {"A": rng.normal(size=(D, H)), "E": rng.normal(size=(K + 1, H)),
         "B_u": rng.normal(size=(H, D)), "B_v": rng.normal(size=(H, D))}
    p.update({k: rng.normal(size=H) for k in ["b_t", "b_h", "b_w", "b_s", "b_e"]})
    return {k: (0.5 * v).astype(np.float32) for k, v in p.items()}


def cases(rng):
    """(name, guided, norm_p, labels, draws in upstream's call order)."""
    normal = lambda shape: ("normal", rng.normal(size=shape).astype(np.float32))
    uniform = lambda: ("uniform", rng.uniform(size=N).astype(np.float32))
    labels = rng.integers(0, K, size=N)
    drop = ("uniform", np.array([0.05, 0.5, 0.02, 0.9, 0.3, 0.7, 0.6, 0.8], np.float32))  # 2 drops
    out = []
    for p in [1.0, 0.0]:
        base = [normal(N), normal(N), normal(SHAPE)]  # t, r, e
        out.append((f"guided-p{p:g}", True, p, labels, base + [uniform(), uniform(), uniform(), drop]))
        # omega = 1, interval [0, 1], null labels: the official objective our guidance=False runs
        unit = [("uniform", np.zeros(N, np.float32)), ("uniform", np.ones(N, np.float32)),
                ("uniform", np.zeros(N, np.float32)), drop]
        out.append((f"unguided-p{p:g}", False, p, np.full(N, K), base + unit))
    return out


class ToyNet(nn.Module):
    init_params: dict

    @nn.compact
    def __call__(self, x, t, h, omega, t_min, t_max, y):
        p = {k: self.param(k, lambda key, v=v: jnp.asarray(v)) for k, v in self.init_params.items()}
        f = jnp.tanh(x.reshape(len(x), -1) @ p["A"] + t[:, None] * p["b_t"] + h[:, None] * p["b_h"]
                     + (1 - 1 / omega)[:, None] * p["b_w"] + t_min[:, None] * p["b_s"]
                     + t_max[:, None] * p["b_e"] + p["E"][y])
        return (f @ p["B_u"]).reshape(x.shape), (f @ p["B_v"]).reshape(x.shape)


class Replay:
    """jax.random stand-in that returns the given draws in order, checking each call's kind."""

    split = staticmethod(jax.random.split)

    def __init__(self, draws):
        self.draws = list(draws)

    def _next(self, kind, shape):
        got, value = self.draws.pop(0)
        assert got == kind, f"upstream asked for a {kind} draw, next recorded draw is {got}"
        return jnp.asarray(value).reshape(shape)

    def normal(self, key, shape, dtype=jnp.float32):
        return self._next("normal", shape).astype(dtype)

    def uniform(self, key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0):
        return (minval + (maxval - minval) * self._next("uniform", shape)).astype(dtype)


def main():
    rng = np.random.default_rng(0)
    params = parameters(rng)
    x = rng.normal(size=SHAPE).astype(np.float32)

    stub = types.ModuleType("models.imfDiT")
    stub.toy = lambda name, num_classes, eval: ToyNet(params, name=name)
    sys.modules["models"] = types.ModuleType("models")
    sys.modules["models"].imfDiT = sys.modules["models.imfDiT"] = stub
    sys.path.insert(0, str(HERE))
    import upstream_imf_jax as up

    key = jax.random.PRNGKey(0)
    out = {"x": x, **{f"param/{k}": v for k, v in params.items()}}
    for name, guided, p, labels, draws in cases(rng):
        replay = Replay(draws)
        up.jax = types.SimpleNamespace(random=replay, jvp=jax.jvp, lax=jax.lax)
        model = up.iMeanFlow(model_str="toy", num_classes=K, norm_p=p)
        variables = model.init({"params": key}, jnp.asarray(x), jnp.ones(N), jnp.asarray(labels))

        def loss(variables):
            return model.apply(variables, jnp.asarray(x), jnp.asarray(labels),
                               method=model.forward, rngs={"gen": key})

        (value, logs), grads = jax.value_and_grad(loss, has_aux=True)(variables)
        assert not replay.draws, "upstream did not use every recorded draw"
        out[f"{name}/labels"] = labels
        for i, (kind, v) in enumerate(draws):
            out[f"{name}/draw{i}/{kind}"] = v
        out[f"{name}/loss"] = np.asarray(value)
        out[f"{name}/loss_u"] = np.asarray(logs["loss_u"])
        out[f"{name}/loss_v"] = np.asarray(logs["loss_v"])
        for k, g in grads["params"]["net"].items():
            out[f"{name}/grad/{k}"] = np.asarray(g)
    out["versions"] = np.array(f"jax {jax.__version__}, flax {sys.modules['flax'].__version__}")
    np.savez(HERE / "imf_reference.npz", **out)
    print("wrote", HERE / "imf_reference.npz", out["versions"])


if __name__ == "__main__":
    main()
