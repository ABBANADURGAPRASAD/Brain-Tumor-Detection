import matplotlib.pyplot as plt
import pickle
import os

# Load training history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Create a directory for saving graphs
graph_dir = "graphs"
os.makedirs(graph_dir, exist_ok=True)

# Plot Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.savefig(os.path.join(graph_dir, "accuracy_graph.png"))  # Save graph
plt.close()

# Plot Loss Graph
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig(os.path.join(graph_dir, "loss_graph.png"))  # Save graph
plt.close()

print("Graphs saved successfully in the 'graphs' folder!")
