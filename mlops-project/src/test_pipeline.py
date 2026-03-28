import sys
import os

sys.path.append(os.path.abspath("."))

from data_ingestion import load_data
from data_preprocessing import split_data
from model_training import train_model
from evaluation import evaluate_model
from model_saver import save_model


# Step 1: Load data
X, y = load_data()
print("Data loaded:", X.shape)

# Step 2: Split data
X_train, X_test, y_train, y_test = split_data(X, y)
print("Data split done")

# Step 3: Train model
model = train_model(X_train, y_train)
print("Model trained")

# Step 4: Evaluate model
accuracy = evaluate_model(model, X_test, y_test)
print("Accuracy:", accuracy)

# Step 5: Save model
save_model(model)
print("Model saved successfully!")