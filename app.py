import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

st.set_page_config(layout="wide", page_title="Activation functions in Neural Networks")

# User inputs
st.sidebar.markdown("# Parameters of Neural network")
num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
hidden_dim = st.sidebar.slider("Neurons per Layer", 8, 256, 64, step=8)
epochs = st.sidebar.slider(
    "Training Epochs",
    500,
    5000,
    1000,
    step=500,
    help="Hyperparameter; it controls how many times the entire training dataset is passed through the model during training",
)

# Activation function options
ACTIVATION_FUNCS = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "Linear": nn.Identity(),
}

activation_name = st.sidebar.selectbox(
    "Choose Activation Function", list(ACTIVATION_FUNCS.keys())
)

# Define equations
functions = {
    "func1": r"$f(x) = x^3 +x^2 - 3x + 2$",
    "func2": r"$f(x) = \sin(2 \pi x) + 0.2*x^3$",
    "func3": r"$f(x) = e^x + \sin(2 \pi x)$",
}

# Dropdown selection (LaTeX won't render here)
selected = st.sidebar.selectbox("Choose a function", list(functions.keys()))

# Define neural network class
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, activation_fn, num_layers=1, hidden_dim=64):
        super(SimpleNet, self).__init__()
        layers = [nn.Linear(1, hidden_dim), activation_fn]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)

        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Define a target function to model
def target_function(x):
    """Target function based on user selection.

    Args:
        x (float): Input value or array of values.

    Returns:
        float: value of the target function at x.
    """
    if selected == "func1":
        return x**3 + x**2 - 3 * x + 2
    elif selected == "func2":
        return np.sin(2 * np.pi * x) + 0.2 * x**3
    elif selected == "func3":
        return np.exp(x) + np.sin(2 * np.pi * x)


st.title("Modeling a Nonlinear Function with Neural Networks")
st.markdown(
    """
    This app demonstrates how a neural network can approximate a nonlinear function. You can adjust the parameters of the neural network and see how it affects the approximation. A neural network doesn’t need to be deep to be powerful. Even a network with just one hidden layer (but enough neurons) can closely mimic any smooth function, as long as:

    - The activation function is non-linear (like Sigmoid, Tanh, or ReLU)  
    - The number of neurons in the hidden layer is sufficiently large 

    """
)

st.markdown(
    "*If you choose a linear activation function, the network will behave like a linear model, regardless of the number of layers or neurons.*"
)


st.markdown(
    f"<span style='color:red'>{functions[selected]}</span>", unsafe_allow_html=True
)


# Generate synthetic data
x = np.linspace(-2, 2, 200).reshape(-1, 1).astype(np.float32)
y = target_function(x)

# Convert to torch tensors
x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Model training
model = SimpleNet(
    ACTIVATION_FUNCS[activation_name], num_layers=num_layers, hidden_dim=hidden_dim
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(x_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    optimizer.step()

# Prediction
model.eval()
y_pred = model(x_tensor).detach().numpy()

col1, col2 = st.columns([2, 1])

with col1:
    # Plot using Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x.flatten(),
            y=y.flatten(),
            mode="lines",
            name="Target Function",
            line=dict(color="black", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x.flatten(),
            y=y_pred.flatten(),
            mode="lines",
            name="NN Prediction",
            line=dict(color="blue"),
        )
    )

    fig.update_layout(
        title=f"NN Approximation using {activation_name} ",
        xaxis_title="x",
        yaxis_title="f(x)",
        legend=dict(x=0.01, y=0.99),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


st.markdown("""### 🧠 Conclusion

This simple app illustrates how a neural network can approximate a nonlinear function, demonstrating the power of neural networks.

The same principles apply to more complex problems. For instance, in image classification tasks like distinguishing cats from dogs, neural networks approximate intricate mappings from pixel data to class labels. Early layers detect edges and textures, while deeper layers learn shapes and patterns.

In essence, the neural network's ability to approximate nonlinear relationships makes it a powerful tool not just for academic functions, but for solving real-world problems across:
- 📸 **Image recognition**
- 🧾 **Natural language processing**
- 📈 **Time-series forecasting**
- 🧬 **Medical diagnosis**
- 🤖 **Autonomous systems**

By exploring neural networks in this simplified setting, we gain foundational insight into how they power many of today’s most advanced AI systems.
""")
