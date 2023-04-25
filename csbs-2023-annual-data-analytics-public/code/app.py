import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn

class NeuralNetworkMultiClassification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetworkMultiClassification, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.5)
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(self.lrelu(self.layer2(x)) + x)
        x = self.drop(x)
        x = self.layer3(x)
        x = self.bn2(self.lrelu(self.layer4(x)) + x)
        x = self.drop(x)
        x = self.lrelu(self.layer5(x))
        x = self.layer6(x)
        return nn.functional.softmax(x, dim=1)

input_size = 60
num_classes = 20
model = NeuralNetworkMultiClassification(input_size, num_classes)

# Load the Pickle model
def load_pickle_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# Function to get prediction results
def get_prediction(data, pickle_model, torch_model):
    # Use the Pickle model
    pickle_result = pickle_model.predict(data)

    # Use the PyTorch model
    tensor_data = torch.tensor(data, dtype=torch.float32)
    torch_result = torch_model(tensor_data).detach().numpy()

    # Combine the results
    combined_result = {"pickle_result": pickle_result, "torch_result": torch_result}

    return combined_result

# App layout
st.title("Simple Web App for ML Model Predictions")

# Load the models
pickle_model_path = "finalized_model_ML.sav"
torch_model_path = "my_classification_model2.pth"

pickle_model = load_pickle_model(pickle_model_path)
model.load_state_dict(torch.load(torch_model_path))
model.eval()

# Input the 15x4 data matrix
row_names = [
    "RCONB835", "RIAD4180", "RCON3505", "RCON2150", "RCON5400", "RCON3499",
    "RCON2930", "RCON6558", "RCONB528", "RCONB576", "RIAD4513", "RCON3495",
    "RCON2160", "RCON5369", "RIAD4301"
]
num_rows = len(row_names)
num_cols = 4
column_names = [f"Quarter {i+1}" for i in range(num_cols)]

input_data = np.empty((num_cols, num_rows))
data_valid = True
input_received = False

for i, row_name in enumerate(row_names):
    st.write(row_name)
    cols = st.columns(num_cols)
    for j in range(num_cols):
        with cols[j]:
            value = st.number_input(
                column_names[j], value=0.0, key=f"R{i+1}C{j+1}", step=None, format="%f"
            )
            if value != 0.0:
                input_received = True
            input_data[j][i] = value

# Get predictions
if st.button("Get Predictions"):
    if input_received:
        try:
            prediction_result = get_prediction(input_data.reshape((60, 1)), pickle_model, model)
            st.write("Pickle Model Result:", prediction_result["pickle_result"])
            st.write("Torch Model Result:", prediction_result["torch_result"])
        except Exception as e:
            st.write("Error:", str(e))
    else:
        st.write("Please provide valid float values for all input boxes.")
