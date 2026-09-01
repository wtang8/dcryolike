"""A learned pose-proposal network, and the features it eats.

The network's job is NOT to estimate the pose -- the matched filter already does
that optimally. Its job is to produce a shortlist whose top-K contains the truth,
which the exact likelihood then adjudicates. So it is trained and scored on
recall@K, never on top-1 accuracy.

Input representation. In the polar Fourier grid an in-plane rotation is exactly a
cyclic shift along theta, so |FFT_theta(.)| is exactly in-plane invariant. We use
that as the feature: it removes a nuisance degree of freedom by construction
rather than asking the network to learn the invariance from data, which matters
when every image is noise-dominated. Only the lowest angular frequencies are
kept; the rest are noise at these SNRs.

Two towers (image, template) share nothing but the embedding space, so the
template tower can later be fed structures it was not trained on.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def inplane_invariant(x: torch.Tensor, n_freq: int = 24) -> torch.Tensor:
    """(..., S, P) complex -> (..., S, n_freq) real, invariant to shifts in theta."""
    f = torch.fft.fft(x, dim=-1)
    return f[..., :n_freq].abs()


def crossshell_phase(x, ctf=None, n_freq=24, max_off=6, eps=1e-8):
    """In-plane-invariant features that KEEP the antipodal distinction.

    C_{s1,s2}(k) = F_{s1}(k) conj(F_{s2}(k)) is in-plane invariant (both factors
    pick up the same phase under a shift in theta), but unlike |F| it retains the
    inter-shell phase.

    Why the magnitude alone fails: for a real density F(-k)=conj(F(k)), so the
    antipodal slice satisfies T_-n(theta) = conj(T_n(-theta)), giving
    FFT[T_-n] = conj(FFT[T_n]) and hence IDENTICAL |FFT|. Magnitude features are
    provably blind to n vs -n. Conjugation flips the sign of Im(C), so the phase
    is exactly where that information lives -- and it is swamped unless C is
    normalised to unit modulus.

    CTF: C -> ctf(s1) ctf(s2) C, and ctf is real, so multiplying by
    sign(ctf(s1)ctf(s2)) removes the CTF completely. The feature is then
    invariant to defocus as well as to in-plane rotation.

    Returns (B, n_pairs * n_freq * 3): phase real, phase imag, log-magnitude.
    """
    f = torch.fft.fft(x, dim=-1)[..., :n_freq]
    S = f.shape[-2]
    i, j = torch.triu_indices(S, S, offset=0, device=f.device)
    keep = (j - i) <= max_off
    i, j = i[keep], j[keep]
    C = f[..., i, :] * f[..., j, :].conj()
    if ctf is not None:
        sgn = torch.sign(ctf[..., i, :1] * ctf[..., j, :1])
        C = C * sgn
    mag = C.abs()
    ph = C / (mag + eps)
    lm = torch.log1p(mag / (mag.mean(dim=(-2, -1), keepdim=True) + eps))
    return torch.cat([ph.real, ph.imag, lm], dim=-1).flatten(1)


def image_features(images, ctf=None, n_freq=24, eps=1e-8):
    """Shift-invariant magnitude spectrum, per-shell whitened, plus CTF profile.

    Per-shell normalisation is the cheap stand-in for whitening by the noise
    power spectrum: it removes the radial amplitude envelope that otherwise
    dominates the representation and differs between synthetic and real data.
    """
    m = inplane_invariant(images, n_freq)                      # (B,S,F)
    m = m / (m.norm(dim=(-2, -1), keepdim=True) + eps)
    shell = m.norm(dim=-1, keepdim=True)
    m = torch.cat([m, shell], dim=-1)                          # (B,S,F+1)
    if ctf is not None:
        m = torch.cat([m, ctf[:, :, :1]], dim=-1)              # (B,S,F+2)
    return m.flatten(1)


class Tower(nn.Module):
    def __init__(self, n_in, dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


class ProposalNet(nn.Module):
    """Two-tower cosine-similarity scorer over a fixed bank of viewing directions."""

    def __init__(self, n_img_in, n_tpl_in, dim=128, hidden=512):
        super().__init__()
        self.img = Tower(n_img_in, dim, hidden)
        self.tpl = Tower(n_tpl_in, dim, hidden)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def scores(self, img_feat, tpl_emb):
        return self.img(img_feat) @ tpl_emb.T * self.logit_scale.exp().clamp(max=100.0)
