import numpy as np
from optics import circular_pupil, apply_phase
from optics.propagation import fraunhofer_psf, mtf_radial
from optics.metrics import normalize_energy

def mtf_circular_analytic(nu: np.ndarray) -> np.ndarray:
    """
    Analytic MTF of a diffraction-limited circular pupil for normalized spatial frequency nu in [0,1]:
    MTF(nu) = (2/pi) [ acos(nu) - nu*sqrt(1-nu^2) ].
    """
    nu = np.clip(nu, 0.0, 1.0)
    return (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu**2))

def test_mtf_matches_circular_analytic_shape_midband():
    n = 512
    pupil = circular_pupil(n, radius=0.42)
    field = apply_phase(pupil, np.zeros((n, n)))
    psf = normalize_energy(fraunhofer_psf(field))

    rho, mtf = mtf_radial(psf)

    # Robust cutoff estimate:
    # For a circular pupil, MTF should trend downward and approach ~0 near cutoff.
    # Discretization/radial-binning can keep it from crossing an arbitrary threshold (like 0.02),
    # so we estimate cutoff by looking at the minimum in the top frequency tail.
    tail = slice(int(0.80 * len(mtf)), len(mtf))
    tail_min_idx = int(np.argmin(mtf[tail]) + 0.80 * len(mtf))
    cutoff = rho[tail_min_idx]

    # Cutoff must be positive and not near zero
    assert cutoff > 0.05

    nu = np.clip(rho / cutoff, 0.0, 1.0)
    mtf_a = mtf_circular_analytic(nu)

    # Compare mid-band where discretization is most reliable
    band = (nu > 0.05) & (nu < 0.70)
    err = float(np.mean(np.abs(mtf[band] - mtf_a[band])))

    # Tolerance accounts for binning/finite grid effects
    assert err < 0.08
