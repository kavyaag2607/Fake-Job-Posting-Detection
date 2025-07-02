import torch
import network  # your network.py file
import joblib

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model (same structure as when you trained)
model = network.Network().to(device)

# Load the trained weights from model.pth
try:
    model.load_state_dict(torch.load("saved_model.pt", map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully and is ready for inference.")
except Exception as e:
    print("Error loading the model:", e)

vectorizers = joblib.load("vectorizers.pkl")

# Example: List the keys (feature names)
print("Loaded vectorizers:")
for key in vectorizers:
    print(f"- {key}: {type(vectorizers[key])}")
