## Overview
SFE-Finder is a quantum machine learning project that aims to predict Stacking Fault Energy (SFE) values for various elements using their physical properties. This project leverages quantum computing techniques, specifically using PennyLane, to create a hybrid quantum-classical model for SFE prediction.
## Features
  Utilizes quantum circuits for feature processing
  Implements a hybrid quantum-classical model
  Predicts SFE values based on electronegativity, bulk modulus, and atomic volume
  Provides data preprocessing and model training scripts
## Requirements
  Python 3.7+
  PennyLane
  NumPy
  Pandas
  Scikit-learn
## Data
The project uses a dataset (qml_training-validation-data.csv) containing the following features for various elements:
  Electronegativity (el_neg)
  Bulk modulus (B/GPa)
  Atomic volume (Volume/A^3)
  Stacking Fault Energy (SFE/mJm^-3) as the target variable
  Model Architecture
The quantum part of the model consists of:
  Input encoding using RY rotations
  Parameterized quantum circuits with RX, RY, and RZ rotations
  Measurement of expectation values of Pauli Z operators
The classical part includes:
  Data preprocessing and scaling
  Optimization using gradient descent
  Cost function based on Mean Absolute Error
## Results
The model's performance is evaluated using Mean Absolute Error (MAE) and R-squared score. Current results and any visualizations can be found in the notebook.
## Contributing
Contributions to improve the model's accuracy, efficiency, or to extend its capabilities are welcome. Please feel free to submit pull requests or open issues for discussion.
