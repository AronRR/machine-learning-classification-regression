import numpy as np
import matplotlib.pyplot as plt

# Ejecutar desde la carpeta que contiene Training Dataset.arff.
with open("Training Dataset.arff", encoding="utf-8") as file:
    for line in file:
        if line.strip().lower() == "@data":
            break
    data = np.loadtxt(file, delimiter=",", comments="%")

X = data[:, :-1]
y = (data[:, -1] == 1).astype(int)  # Result: -1 -> 0, 1 -> 1

# División manual: 60% Train, 20% Validation y 20% Test.
np.random.seed(42)
indices = np.arange(len(y))
np.random.shuffle(indices)
n_train = int(0.6 * len(y))
n_val = int(0.2 * len(y))
train, val, test = np.split(indices, [n_train, n_train + n_val])
X_train, y_train = X[train], y[train]
X_val, y_val = X[val], y[val]
X_test, y_test = X[test], y[test]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(y, probability):
    p = np.clip(probability, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# Batch Gradient Descent: todos los datos de Train en cada actualización.
learning_rate = 0.01
epochs = 3000
threshold = 0.5
weights = np.zeros(X_train.shape[1])
bias = 0.0
train_loss, val_loss = [], []

for epoch in range(epochs):
    probability = sigmoid(X_train @ weights + bias)
    error = probability - y_train
    dw = X_train.T @ error / len(y_train)
    db = np.mean(error)
    weights -= learning_rate * dw
    bias -= learning_rate * db
    train_loss.append(cost_function(y_train, sigmoid(X_train @ weights + bias)))
    val_loss.append(cost_function(y_val, sigmoid(X_val @ weights + bias)))

# Métricas manuales.
for name, X_set, y_set in [("Train", X_train, y_train),
                            ("Validation", X_val, y_val),
                            ("Test", X_test, y_test)]:
    probability = sigmoid(X_set @ weights + bias)
    prediction = (probability >= threshold).astype(int)
    accuracy = np.mean(prediction == y_set)
    tp = np.sum((y_set == 1) & (prediction == 1))
    fp = np.sum((y_set == 0) & (prediction == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    print(f"{name}: Loss={cost_function(y_set, probability):.4f}, "
          f"Accuracy={accuracy:.2%}, Precision={precision:.2%}")

test_probability = sigmoid(X_test @ weights + bias)
test_prediction = (test_probability >= threshold).astype(int)
tn = np.sum((y_test == 0) & (test_prediction == 0))
fp = np.sum((y_test == 0) & (test_prediction == 1))
fn = np.sum((y_test == 1) & (test_prediction == 0))
tp = np.sum((y_test == 1) & (test_prediction == 1))
matrix = np.array([[tn, fp], [fn, tp]])
print("Matriz Test (filas: real; columnas: predicción):\n", matrix)

plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), train_loss, label="Train")
plt.plot(range(1, epochs + 1), val_loss, label="Validation")
plt.xlabel("Época")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_train_validation.png", dpi=200)

plt.figure(figsize=(5, 4))
plt.imshow(matrix, cmap="Blues")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(matrix[i, j]), ha="center", va="center",
                 color="white" if matrix[i, j] > matrix.max() / 2 else "black")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])
plt.xlabel("Predicción")
plt.ylabel("Clase real")
plt.title("Matriz de confusión — Test")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=200)

plt.figure(figsize=(7, 4))
for label in [0, 1]:
    plt.hist(test_probability[y_test == label], bins=np.linspace(0, 1, 31),
             alpha=0.6, label=f"Clase real {label}")
plt.axvline(threshold, color="black", linestyle="--", label="Umbral 0.5")
plt.xlabel("Probabilidad de Result = 1")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades — Test")
plt.legend()
plt.tight_layout()
plt.savefig("probabilities_test.png", dpi=200)
plt.show()

print("\nEjemplos de predicciones:")

for i in range(5):
    print(
        f"Probabilidad: {test_probability[i]:.4f}, "
        f"Predicción: {test_prediction[i]}, "
        f"Real: {y_test[i]}"
    )