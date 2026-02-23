# Optical PSF/MTF Toolkit (Fraunhofer)

A compact computational optics toolkit that generates PSF/OTF/MTF from a pupil function, with **physics-informed pytest verification** and CI automation.

## What this demonstrates
- Fourier optics implementation (Fraunhofer PSF, OTF/MTF)
- Verification mindset:
  - Energy invariance under global phase
  - Strehl monotonic decrease vs defocus amplitude
  - Numerical radial MTF vs analytic circular-aperture MTF (mid-band)

## Run locally
```bash
pip install -r requirements.txt
pytest -q
python scripts/demo_psf_mtf.py
