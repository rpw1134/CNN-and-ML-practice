"""
Test script for SoftmaxRegressionModel on a linearly separable multi-class dataset.
"""
import numpy as np
from src.ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_multiclass_linear_data
from src.ml_practice_and_neural_net.ml_practice.models.SoftmaxRegressionModel import SoftmaxRegressionModel


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy percentage."""
    return np.mean(y_true == y_pred) * 100


if __name__ == "__main__":
    print("="*60)
    print("Testing SoftmaxRegressionModel on Linearly Separable Data")
    print("="*60)

    # Generate a perfectly separable dataset
    print("\n1. Generating dataset...")
    X, y = generate_multiclass_linear_data(
        n_samples=600,
        n_features=2,
        n_classes=3,
        separation=8.0,  # Large separation for perfect separability
        noise=0.3,       # Low noise for clean separation
        random_state=42
    )

    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Classes: {np.unique(y)}")
    for i in np.unique(y):
        print(f"     Class {i}: {np.sum(y == i)} samples")

    # Train the model
    print("\n2. Training SoftmaxRegressionModel...")
    model = SoftmaxRegressionModel(
        learning_rate=0.1,
        num_training_iterations=2000
    )
    model.fit(X, y)
    print("   Training complete!")

    # Evaluate on training set (model internally splits into train/test)
    print("\n3. Evaluating model...")

    # Get predictions on full dataset
    y_pred = model.predict(X)
    accuracy = calculate_accuracy(y, y_pred)

    print(f"   Full dataset accuracy: {accuracy:.2f}%")

    # Per-class accuracy
    print("\n   Per-class accuracy:")
    for i in np.unique(y):
        mask = y == i
        class_accuracy = calculate_accuracy(y[mask], y_pred[mask])
        print(f"     Class {i}: {class_accuracy:.2f}%")

    # Loss values
    print("\n4. Loss values:")
    train_loss = model.evaluate_cce_training_loss()
    test_loss = model.evaluate_cce_testing_loss()
    print(f"   Training loss: {train_loss:.6f}")
    print(f"   Testing loss:  {test_loss:.6f}")

    # Show confusion matrix
    print("\n5. Confusion Matrix:")
    classes = np.unique(y)
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true_class in classes:
        for pred_class in classes:
            conf_matrix[true_class, pred_class] = np.sum((y == true_class) & (y_pred == pred_class))

    print("   Predicted →")
    print("   Actual ↓")
    header = "        " + "  ".join([f"Class {i}" for i in classes])
    print(header)
    for i, row in enumerate(conf_matrix):
        print(f"   Class {i}  " + "  ".join([f"{val:7d}" for val in row]))

    print("\n" + "="*60)
    if accuracy >= 99.0:
        print("✅ SUCCESS: Model achieved near-perfect accuracy!")
    elif accuracy >= 90.0:
        print("⚠️  GOOD: Model achieved high accuracy but not perfect.")
    else:
        print("❌ ISSUE: Model accuracy is lower than expected.")
    print("="*60)

