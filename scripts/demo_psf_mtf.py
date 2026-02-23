import os
import numpy as np

from optics import (
    circular_pupil, zernike_defocus, apply_phase,
    fraunhofer_psf, mtf_radial, strehl_ratio, normalize_energy
)
from optics.plotting import save_psf_image, save_mtf_curve

def main():
    os.makedirs("results", exist_ok=True)

    n = 512
    pupil = circular_pupil(n, radius=0.42)

    # Ideal PSF
    pupil_ideal = apply_phase(pupil, phase_waves=np.zeros((n, n)))
    psf_ideal = normalize_energy(fraunhofer_psf(pupil_ideal))

    # Aberrated PSF: defocus in waves
    zdef = zernike_defocus(n)
    phase = 0.15 * zdef  # waves
    pupil_ab = apply_phase(pupil, phase_waves=phase)
    psf_ab = normalize_energy(fraunhofer_psf(pupil_ab))

    s = strehl_ratio(psf_ab, psf_ideal)
    print(f"Strehl ratio (defocus): {s:.4f}")

    rho_i, mtf_i = mtf_radial(psf_ideal)
    rho_a, mtf_a = mtf_radial(psf_ab)

    save_psf_image(psf_ideal, "results/psf_ideal.png", log=True)
    save_psf_image(psf_ab, "results/psf_defocus.png", log=True)
    save_mtf_curve(rho_i, mtf_i, "results/mtf_ideal.png")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rho_i, mtf_i, label="Ideal")
    plt.plot(rho_a, mtf_a, label="Defocus")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Normalized spatial frequency (arb.)")
    plt.ylabel("MTF")
    plt.title("MTF: Ideal vs Defocus")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mtf_compare.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    main()
