import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
def circuit(weights, x):
    qml.templates.AngleEmbedding(x, wires = range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires = range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def variational_classifier(weights, bias, x):
    circuit_output = circuit(weights, x)
    mean_expectation = np.mean(qml.numpy.array(circuit_output))
    return mean_expectation, bias
def square_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def cost(weights, bias, X, y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(y, predictions)

data = pd.read_csv('qml_training-validation-data.csv')

target = data['SFE/mJm^-3']
features = data.drop(columns=['SFE/mJm^-3'])

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)


x_train, x_test, y_train, y_test  = model_selection.train_test_split(features_normalized,target,test_size=0.2, random_state=42)

n_qubits = x_train.shape[1]
dev = qml.device("default.qubit", wires = n_qubits)

np.random.seed(42)
weights = 0.01 * np.random.randn(3, n_qubits)
bias = qml.numpy.array(0.0, requires_grad = True)

opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

num_epochs = 50
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        weights, bias, _ = opt.step(lambda w, b: cost(w, b, x_batch, y_batch), weights, bias)

    train_cost = cost(weights, bias, x_train, y_train)
    val_cost = cost(weights, bias, x_test, y_test)
    print(f"Epoch {epoch + 1}: Training cost = {train_cost:.4f} | Validation cost = {val_cost:.4f}")

predictions = [variational_classifier(weights, bias, x) for x in x_test]

print(predictions)