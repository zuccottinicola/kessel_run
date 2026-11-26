# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: <https://tiziano1590.github.io/comp_astro_26/>.

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_26.git
cd comp_astro_26
pip install .
```

### Development installation

```bash
pip install -e '.[dev,docs]'
```
### Optional dependencies

```bash
pip install '.[cnn]'  # CNN detection
pip install '.[gan]'  # GAN generation
```
## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --transit [--output lightcurve.png]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-d, --detection {rf,cnn}`: Run exoplanet detection with the chosen algorithm (`rf` = Random Forest; `cnn` = Convolutional Neural Network)
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for the transit plot (default: `lc.png`)
  
### Plot the light curve

Plot the transit light curve of the planet defined in a parameters file:

```bash
daneel -i path_to_the_parameters.yaml -t
```

Use `-o name.png` to choose the output file (default: `lc.png`).

### Random Forest detection (Assignment 2, Task G)

Train and test a Random Forest classifier on the TESS light curves:

```bash
daneel -i examples/parameters_taskG.yaml -d rf
```

The YAML file contains the path to the dataset and the Random Forest settings
(number of trees, maximum depth, ...). The dataset `tess_data.csv` is not in the
repository: put it in `../datasets/` and run the command from the repository root.

The command prints the OOB score, the AUC, the accuracy and the confusion matrix,
both at the default threshold (0.5) and at the optimal threshold.

### CNN detection (Assignment 2, Task H)

Train and evaluate a Convolutional Neural Network
on the TESS light curves:

```bash
daneel -i examples/parameters_taskH.yaml -d cnn
```

The YAML file contains the dataset path, the number
of bins and the selected CNN architecture.

```yaml
cnn:
  csv_path: "../datasets/tess_data.csv"   # path to the TESS dataset
  nbins: 1000                             # flux values per light curve
  model: 1                                # CNN architecture: 1, 2 or 3
```

The dataset `tess_data.csv` must be placed in
`../datasets/`. Run the command from the repository root.

Three architectures are available:

- Model 1: original CNN.
- Model 2: CNN with a smaller classification head.
- Model 3: CNN with two convolutional layers.

The detection is implemented by the class `CNNClassifier` in `daneel.detection.classifiers`(methods `fit` and
`evaluate_with_optimal_threshold`), which reuses the functions in `daneel/detection/cnn_classifier.py`. An invalid `model` value stops the
program with a `ValueError`.

The command prints the AUC, accuracy, precision,recall and optimal classification threshold.

The trained model, optimal threshold, confusion matrix, training history and sample light curves are saved in the current working directory.

The CNN (`-d cnn`) needs TensorFlow 

### Examples
```bash
# Plot a transit light curve
daneel -i examples/params.yaml --transit --output lightcurve.png

# Run Random Forest detection on the TESS dataset
daneel -i examples/parameters_taskG.yaml --detection rf

# Run CNN detection on the TESS dataset
daneel -i examples/parameters_taskH.yaml --detection cnn

# Run atmospheric characterisation (not implemented yet)
daneel -i examples/params.yaml --atmosphere
```
## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

## Development

Run the test suite and style checks with:

```bash
pytest
ruff check .
```

Build the documentation with `make -C docs html`.

## License

This project is licensed under the GNU General Public License v3.0.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
