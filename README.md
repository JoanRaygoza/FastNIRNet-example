# FastNIRNet

**Fast Near‑Infrared Spectra Synthesizer with Neural Networks**

FastNIRNet is a family of neural‑network models that generate high‑resolution NIR (near‑infrared) stellar spectra for three wavelength windows:

* **15 200 Å – 15 700 Å**
* **15 950 Å – 16 300 Å**
* **16 600 Å – 16 850 Å**

(all sampled every **0.2 Å**).

Given six astrophysical parameters—effective temperature, surface gravity, global metallicity, alpha enhancement, projected rotational velocity, and limb‑darkening coefficient—the network outputs a synthetic flux vector with 5 503 points. Four model sizes let you trade accuracy for speed and memory.

The core library is light‑weight; the first time you instantiate a model, its weights are fetched from **Google Drive** via `gdown` and cached locally under `~/FastNIRNet_models/`.

---

## Astrophysical parameter space

| Parameter              | Min   | Max   |
| ---------------------- | ----- | ----- |
| **Teff** (K)           | 3 000 | 8 000 |
| **log g** (dex)        | 1.5   | 5.5   |
| **\[M/H]** (dex)       | –2.5  | +1.0  |
| **\[α/M]** (dex)       | –0.75 | +1.0  |
| **v sin i** (km s⁻¹)   | 0     | 50    |
| **ε** (limb‑darkening) | 0.0   | 1.0   |

---

## Model variants

| Variant    | Parameters | Download size\* | Best suited for                |
| ---------- | ---------- | --------------- | ------------------------------ |
| **tiny**   | \~1 M      | ≈ 11 MB         | Edge devices & rapid scans     |
| **small**  | \~3 M      | ≈ 38 MB         | Laptops / notebooks            |
| **medium** | \~12 M     | ≈ 138 MB        | Workstations & small GPUs      |
| **large**  | \~46 M     | ≈ 528 MB        | Maximum fidelity (server GPUs) |

\*Sizes are approximate .h5 files downloaded on demand.

---

## Installation

> FastNIRNet is currently in private beta. 


## Quick start — single input

```python
import numpy as np
from fastnirnet import FastNIRNet

# Medium‑size network (downloads weights on first run)
net = FastNIRNet("medium")

# teff, logg, mh, am, vsini, epsilon
x = np.array([6500, 4.5, 0.0, 0.4, 15, 0.6])

spectrum = net.synthetize_spectra(x)
print(spectrum.shape)  # (1, 5503)
```

### Multiple inputs

```python
X = np.array([
    [6500, 4.0, 0.0, 0.4, 12, 0.6],
    [4200, 2.5, –0.5, 0.2,  8, 0.4],
    [7600, 3.8, 0.3, 0.8, 30, 0.7],
])

synth = net.synthetize_spectra(X)
print(synth.shape)  # (3, 5503)
```

### Important
If using the CPU, you can select the number of jobs to use. This configuration must be done before instantiating FastNIRNet.

```python
from fastnirnet import FastNIRNet, config
config(jobs=6)
net = FastNIRNet("medium")
...
```

---

## Extra utilities

```python
wl   = net.get_wavelength()  # ndarray of 5503 wavelengths (Å)
seg  = net.get_segments()    # list of segment IDs ("0", "1", "2") for every region of spectra mentioned above
```

---

## API (`fastnirnet.fastnirnet`)

```python
class FastNIRNet(model: str = "large"):
    synthetize_spectra(data, batch_size: int = 32) -> np.ndarray
    get_wavelength() -> np.ndarray
    get_segments() -> list[str]
```

* **model** — one of `"tiny"`, `"small"`, `"medium"`, `"large"`
* **data** — 1‑D or 2‑D array with 6 columns in the order *(Teff, log g, \[M/H], \[α/M], v sin i, ε)*.

---

## Troubleshooting

| Symptom                                          | Fix                                                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| *`ModuleNotFoundError: No module named 'gdown'`* | `pip install gdown`                                                                                                 |
| Slow download / blocked                          | Download the `.h5` manually from the Google Drive link in `_MODEL_TABLE` and place it under `~/FastNIRNet_models/`. |
| `ValueError: Model 'xyz' not exist.`             | Use one of the four valid model names.                                                                              |

---

## Contributing

Bug reports and feature requests are welcome on the *Issues* page.

---

## License

MIT © 2025 Joan Raygoza
 
