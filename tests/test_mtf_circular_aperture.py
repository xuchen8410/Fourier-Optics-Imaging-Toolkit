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

    # Data-driven cutoff: first point where mtf ~ 0
    idx = np.where(mtf < 0.02)[0]
    assert len(idx) > 10
    cutoff = rho[idx[0]]
    nu = np.clip(rho / cutoff, 0.0, 1.0)

    mtf_a = mtf_circular_analytic(nu)

    # Compare mid-band where discretization is stable
    band = (nu > 0.05) & (nu < 0.75)
    err = float(np.mean(np.abs(mtf[band] - mtf_a[band])))
    assert err < 0.06
