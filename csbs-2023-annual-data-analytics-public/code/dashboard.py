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


# Function to get prediction results
def get_prediction(data, torch_model):

    # Use the PyTorch model
    tensor_data = torch.tensor(data, dtype=torch.float32)
    torch_result = torch_model(tensor_data).detach()

    # Combine the results
    combined_result = {"torch_result": torch.argmax(torch_result) + 1}

    return combined_result

# App layout
st.title("Simple Web App for ML Model Predictions")

# Load the models
torch_model_path = "my_classification_model2.pth"
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
            if isinstance(value, (int, float)):
                input_received = True
            input_data[j][i] = value

# Get predictions
if st.button("Get Predictions"):
    if input_received:
        try:
            prediction_result = get_prediction(input_data.reshape((1, 60)), model)
            st.markdown(f'<h2 style="font-size: 24px;">Neural Network Model Result:</h2>', unsafe_allow_html=True)
            st.write(f"This Community bank can survive {prediction_result['torch_result'].item()} quarters")
        except Exception as e:
            st.write("Error:", str(e))
    else:
        st.write("Please provide valid float values for all input boxes.")
