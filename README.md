# FastNIRNet

**Fast Near‑Infrared Spectra Synthesizer with Neural Networks**

FastNIRNet is a family of neural‑network models that generate high‑resolution NIR (near‑infrared) stellar spectra from **15 104.2 Å** to **17 000 Å** wavelength.

(all sampled every **0.2 Å**).

Given seven astrophysical parameters—effective temperature, surface gravity, global metallicity, alpha enhancement, microturbulent velocity, projected rotational velocity, and doppler shift—the network outputs a synthetic flux vector with 9480 points. Four model sizes let you trade accuracy for speed and memory.

The core library is light‑weight; the first time you instantiate a model, its weights are fetched from **Google Drive** via `gdown` and cached locally under `~/FastNIRNet_models/`.

---

## Astrophysical parameter space

| Parameter                  | Min   | Max   |
| -------------------------- | ----- | ----- |
| **Teff** (K)               | 3 000 | 8 000 |
| **log g** (dex)            | 1.5   | 5.5   |
| **\[M/H]** (dex)           | –2.5  | +1.0  |
| **\[α/M]** (dex)           | –0.75 | +1.0  |
| **vmic** (km s⁻¹)          | 0.3   | 4.8   |
| **v sin i** (km s⁻¹)       | 0     | 150   |
| **doppler shift** (km s⁻¹) | -400  | 400   |

---

## Model variants

| Variant    | Parameters | Download size\* | Best suited for                |
| ---------- | ---------- | --------------- | ------------------------------ |
| **tiny**   | \~790 k    | ≈ 8 MB          | Edge devices & rapid scans     |
| **small**  | \~3 M      | ≈ 35 MB         | Laptops / notebooks            |
| **medium** | \~15 M     | ≈ 170 MB        | Workstations & small GPUs      |
| **large**  | \~52 M     | ≈ 604 MB        | Maximum fidelity (server GPUs) |

\*Sizes are approximate .h5 files downloaded on demand.

---

## Installation

> FastNIRNet is currently in private beta.

---

## Quick start 
### Synthetize — Single input
```python
import numpy as np
import fastnirnet

# Medium‑size network (downloads weights on first run)
net = FastNIRNet("large")

# teff, logg, mh, am, vmic, vsini, doppler shift
x = np.array([6500, 4.5, 0.0, 0.4, 1.3, 15, -27])

spectrum = net.synthetize_spectra(x)
print(spectrum.shape)  # (1, 9480)
```

### Synthetize — Multiple inputs

```python
X = np.array([
    [3200, 1.5, -0.2, -0.4, 0.3, 5, 0.0],
    [6500, 3.2, 0.7, 0.4, 1.3, 100, -35.8],
    [7500, 4.5, 0.0, 0.0, 3.3, 70.5, 50.7],
])

synth = net.synthetize_spectra(X)
print(synth.shape)  # (3, 9480)
```

### Inversion - Estimate parameters from a spectrum
H-MagNet can also be used to invert spectra, estimating the seven astrophysical parameters (Teff, logg, [M/H], AM, vmic, vsini, doppler shift) from an observed flux vector. It uses Particle Swarm Optimization (PSO) to minimize the error between the observed spectrum and the network's prediction.

```python
solution, inv_spectra, fitness = net.inversion(
    y_obs=spectrum,         # Input flux (shape: [1, 9480] or [9480])
    n_particles=1024,       # Number of particles (poblation size)
    iters=10,               # Optimization iterations
    verbose=1               # Show progress
)
```
This returns:

* **solution**: best-fit astrophysical parameters found by the optimizer.
* **inv_spectra**: synthetic spectrum generated from the inferred parameters.
* **fitness**: final value of the objective function.

### Custom objective function
You can provide your own objective function to compare the observed and predicted spectra. It must accept two arguments: y_obs and y_pred.

Example using mean absolute error (per wavelength point):

```python
from sklearn.metrics import mean_absolute_error
def obj(y_obs, y_pred):
    return mean_absolute_error(y_obs.T, y_pred.T, multioutput='raw_values')
```
Then use it like this:
```python
solution, inv_spectra, fitness = net.inversion(
    y_obs=spectrum,
    n_particles=4096,
    iters=10,
    objective_function=obj,
    verbose=1
)
```

### Inversion with fixed parameters
You can fix specific astrophysical parameters during the inversion by using the corresponding keyword arguments:
* `fixed_teff`
* `fixed_logg`
* `fixed_mh`
* `fixed_am`
* `fixed_vmic`
* `fixed_vsini`
* `fixed_doppler_shift`

For example, the following call fixes `logg` and `vsini`:
```python
solution, inv_spectra, fitness = net.inversion(
    spectrum, 
    n_particles=4096, 
    iters=10, 
    fixed_logg=3.12, 
    fixed_vsini=9.8, 
    verbose=1)
```

### Inversion with parameter ranges
You can constrain parameters to a specific range using the following arguments:
* `teff_range`
* `logg_range`
* `mh_range`
* `am_range`
* `vmic_range`
* `vsini_range`
* `doppler_shift_range`

```python
solution, inv_spectra, fitness = net.inversion(
    spectrum, 
    n_particles=4096, 
    iters=10, 
    teff_range=(4000,6000), 
    vsini_range=(5,20), 
    verbose=1)
```
This limits `Teff` and `vsini` to specific ranges while leaving the other parameters free. You may also combine fixed values for some parameters with range limits for others.

---

### Important
Be sure to select the corresponding device for run the model, whether GPU or CPU.
If using the CPU, you can select the number of jobs to use. This configuration must be done before instantiating FastNIRNet.

```python
from fastnirnet import FastNIRNet, config
config(jobs=6)
net = FastNIRNet("large")
...
```

## Extra utilities

```python
wl   = net.get_wavelength()  # ndarray of 9480 wavelengths (Å)
```

---

## API (`fastnirnet`)
### `class FastNIRNet(model: str = "large", batch_size: int = 512)`

Main interface for spectral synthesis and inversion using neural network models.

The `model` argument can be one of:

* `"tiny"` — fastest, lower accuracy
* `"small"` — good trade-off
* `"medium"` — higher accuracy, slower
* `"large"` *(default)* — best accuracy, highest memory and compute cost

Model weights are downloaded on first use and cached locally in `~/FastNIRNet_models/`.

`batch_size` argument controls the amount of data to load on the model when it is predicting (synthetize and inversion).

---

### **Methods**

```python
synthetize_spectra(data: np.ndarray) -> np.ndarray
```

Generates synthetic spectra from stellar parameters.

* `data`: 1D or 2D NumPy array of shape `(7,)` or `(n_samples, 7)` in the order
  *(Teff, log g, \[M/H], AM, vmic, vsini, doppler)*.
* `batch_size`: number of inputs per batch (for efficiency on large inputs).

---

```python
get_wavelength() -> np.ndarray
```

Returns the wavelength grid (shape: `(9480,)`) used by the model.

---

```python
inversion(
    y_obs: np.ndarray,
    wl_obs: np.ndarray = None,
    n_particles: int = 4096,
    iters: int = 10,
    objective_function: Callable = default_objective,
    W: float = 0.7,
    C1: float = 1.0,
    C2: float = 1.0,
    fixed_teff: float | None = None,
    fixed_logg: float | None = None,
    fixed_mh: float | None = None,
    fixed_am: float | None = None,
    fixed_vmic: float | None = None,
    fixed_vsini: float | None = None,
    fixed_doppler_shift: float | None = None,
    teff_range: tuple[float, float] = (3000, 8000),
    logg_range: tuple[float, float] = (1.5, 5.5),
    mh_range: tuple[float, float] = (-2.5, 1.0),
    am_range: tuple[float, float] = (-0.75, 1.0),
    vmic_range: tuple[float, float] = (0.3, 4.8),
    vsini_range: tuple[float, float] = (0.0, 150.0),
    doppler_shift_range: tuple[float, float] = (-200.0, 200.0),
    tol: float = 0.5,
    min_wlp = int = 4,
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray, float]
```

Performs a global optimization using Particle Swarm Optimization (PSO) to infer the atmospheric parameters that best reproduce the observed spectrum `y_obs`.
You can:
* Fix any subset of parameters using the `fixed_*` arguments.
* Restrict the search space of free parameters using the corresponding `*_range` tuples.
* Combine fixed and ranged parameters as needed.
**Parameters**:
* y_obs (`np.ndarray`): The observed flux vector to be fitted.
* wl_obs (`np.ndarray`): The observed wavelength vector.
* n_particles (`int`): Number of particles used in the PSO swarm.
* iters (`int`): Number of optimization iterations.
* objective_function (`Callable`, optional): Objective function to minimize. Defaults to `default_objective`.
* W, C1, C2 (`float`): PSO hyperparameters controlling inertia and learning factors.
* fixed_teff, fixed_logg, fixed_mh, fixed_bfield, fixed_vsini (`float | None`): Values to fix specific parameters during the inversion. If `None`, the parameter is optimized.
* teff_range, logg_range, mh_range, bfield_range, vsini_range (`tuple[float, float]`): Search intervals for each parameter (used if not fixed).
* tol (`float`): Tolerance in difference of 2 consecutive wavelength points to consider in optimization process.
* min_wlp(`int`): Minimum number of wavelength points consecutives below the tolence to consider in optimization process.
* verbose (`int`): Verbosity level (0 = silent, 1 = progress info).
**Returns**:
* `solution` (`np.ndarray`): Best-fit parameter vector.
* `inv_spectra` (`np.ndarray`): Synthetic spectrum corresponding to the best solution.
* `fitness` (`float`): Final error value of the best-fit solution.

---

## Troubleshooting

| Symptom                                          | Fix                                                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| *`ModuleNotFoundError: No module named 'gdown'`* | `pip install gdown`                                                                                                 |
| Slow download / blocked                          | Download the `.h5` manually from the Google Drive link in `_MODEL_TABLE` and place it under `~/FastNIRNet_models/`. |
| `ValueError: Model 'xyz' not exist.`             | Use one of the four valid model names.                                                                              |

---

## Contributing

1. Fork this repository and create a feature or bug‑fix branch.
2. Run the unit tests (`pytest`).
3. Open a pull request.

Bug reports and feature requests are welcome on the *Issues* page.

---

## License

MIT © 2025 Joan Raygoza
