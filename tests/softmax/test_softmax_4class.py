"""
Additional test for SoftmaxRegressionModel with 4 classes and slightly more noise.
"""
import numpy as np
from src.ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_multiclass_linear_data
from src.ml_practice_and_neural_net.ml_practice.models.SoftmaxRegressionModel import SoftmaxRegressionModel


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy percentage."""
    return np.mean(y_true == y_pred) * 100


if __name__ == "__main__":
    print("="*60)
    print("Testing SoftmaxRegressionModel - 4 Classes")
    print("="*60)

    # Generate a 4-class dataset
    print("\n1. Generating 4-class dataset...")
    X, y = generate_multiclass_linear_data(
        n_samples=800,
        n_features=2,
        n_classes=4,
        separation=6.0,
        noise=0.4,
        random_state=123
    )

    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Classes: {np.unique(y)}")
    for i in np.unique(y):
        print(f"     Class {i}: {np.sum(y == i)} samples")

    # Train the model
    print("\n2. Training SoftmaxRegressionModel...")
    model = SoftmaxRegressionModel(
        learning_rate=0.1,
        num_training_iterations=3000
    )
    model.fit(X, y)
    print("   Training complete!")

    # Evaluate
    print("\n3. Evaluating model...")
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

    print("\n" + "="*60)
    if accuracy >= 99.0:
        print("✅ SUCCESS: Model achieved near-perfect accuracy!")
    elif accuracy >= 95.0:
        print("✅ GOOD: Model achieved excellent accuracy!")
    elif accuracy >= 90.0:
        print("⚠️  DECENT: Model achieved good accuracy.")
    else:
        print("❌ ISSUE: Model accuracy is lower than expected.")
    print("="*60)

