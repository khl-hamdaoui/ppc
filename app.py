"""import streamlit as st
import numpy as np
from catboost import CatBoostRegressor
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


# Load your trained model
@st.cache_resource
def load_model():
    try:
        with open("cat_best_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None


# Load the saved StandardScaler
@st.cache_resource
def load_scaler():
    try:
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        st.error(f"Failed to load the scaler: {e}")
        return None


cat_model = load_model()
scaler = load_scaler()

# Check if model and scaler are loaded successfully
if cat_model is None or scaler is None:
    st.stop()

# Sidebar for input parameters
st.sidebar.title("Input Parameters")
gamma_d = st.sidebar.number_input(
    "γd [kN.m-3]", min_value=0.0, max_value=30.0, step=0.1, value=16.0
)
wi = st.sidebar.number_input(
    "wi [%]", min_value=0.0, max_value=1000.0, step=0.1, value=20.0
)
LL = st.sidebar.number_input(
    "LL [%]", min_value=0.0, max_value=1000.0, step=0.1, value=102.0
)
Sand = st.sidebar.number_input(
    "Sand [%]", min_value=0.0, max_value=1000.0, step=0.1, value=2.0
)
PI = st.sidebar.number_input(
    "PI [%]", min_value=0.0, max_value=1000.0, step=0.1, value=82.0
)
C = st.sidebar.number_input(
    "C [%]", min_value=0.0, max_value=1000.0, step=0.1, value=85.0
)
A = st.sidebar.number_input("A", min_value=0.0, max_value=1000.0, step=0.1, value=0.96)
LP = st.sidebar.number_input(
    "LP [%]", min_value=0.0, max_value=1000.0, step=0.1, value=20.0
)

if st.sidebar.button("Predict Swelling Pressure"):
    if None in [A, Sand, C, LL, LP, PI, gamma_d, wi]:
        st.sidebar.error("Please fill in all fields.")
    else:
        input_data = np.array([[A, Sand, C, LL, LP, PI, gamma_d, wi]])
        try:
            input_data_scaled = scaler.transform(input_data)
            prediction = cat_model.predict(input_data_scaled)
            st.sidebar.success(
                f"Predicted Swelling Pressure (Pr [MPa]): {prediction[0]:.4f}"
            )
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")

# Main content
st.title("Swelling Pressure Prediction App")

st.header("Parametric Study")
params = {
    "γd [kN.m-3]": gamma_d,
    "wi [%]": wi,
    "LL [%]": LL,
    "Sand [%]": Sand,
    "PI [%]": PI,
    "C [%]": C,
    "A": A,
    "LP [%]": LP,
}

col1, col2 = st.columns([1, 3])

with col1:
    param_choice = st.selectbox("Select parameter:", list(params.keys()))
    param_min = st.number_input(
        f"Minimum {param_choice}", value=params[param_choice] - 1.0
    )
    param_max = st.number_input(
        f"Maximum {param_choice}", value=params[param_choice] + 1.0
    )
    num_points = st.slider("Number of points", min_value=10, max_value=100, value=50)
    poly_degree = st.slider("Polynomial degree", min_value=1, max_value=10, value=3)

    perform_study = st.button("Perform Study", use_container_width=True)

with col2:
    if perform_study:
        try:
            param_values = np.linspace(param_min, param_max, num_points)
            predictions = []
            fixed_values = params.copy()
            feature_order = [
                "A",
                "Sand [%]",
                "C [%]",
                "LL [%]",
                "LP [%]",
                "PI [%]",
                "γd [kN.m-3]",
                "wi [%]",
            ]
            for value in param_values:
                input_data = [fixed_values[feature] for feature in feature_order]
                input_data[feature_order.index(param_choice)] = value
                input_data = np.array([input_data])
                input_data_scaled = scaler.transform(input_data)
                prediction = cat_model.predict(input_data_scaled)
                predictions.append(prediction[0])

            coeffs = np.polyfit(param_values, predictions, poly_degree)
            poly = np.poly1d(coeffs)

            x_smooth = np.linspace(param_min, param_max, 200)
            y_smooth = poly(x_smooth)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                x_smooth,
                y_smooth,
                label=f"Polynomial fit (degree {poly_degree})",
                color="#FF4B4B",
                linewidth=2,
            )
            ax.set_xlabel(param_choice, fontsize=12)
            ax.set_ylabel("Predicted Swelling Pressure (Pr [MPa])", fontsize=12)
            ax.set_title(
                f"Parametric Study: {param_choice} vs Predicted Swelling Pressure",
                fontsize=14,
            )
            ax.grid(True)
            ax.legend(fontsize=10)
            ax.tick_params(axis="both", which="major", labelsize=10)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Parametric study failed: {e}")
    else:
        st.info("Click 'Perform Study' to see the results.")
"""

import streamlit as st
import numpy as np
from catboost import CatBoostRegressor
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


# Load your trained model
@st.cache_resource
def load_model():
    try:
        with open("cat_best_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None


# Load the saved StandardScaler
@st.cache_resource
def load_scaler():
    try:
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        st.error(f"Failed to load the scaler: {e}")
        return None


cat_model = load_model()
scaler = load_scaler()

# Check if model and scaler are loaded successfully
if cat_model is None or scaler is None:
    st.stop()

# Sidebar for input parameters
st.sidebar.title("Input Parameters")
gamma_d = st.sidebar.number_input(
    "γd [kN.m-3]", min_value=0.0, max_value=30.0, step=0.1, value=16.0
)
wi = st.sidebar.number_input(
    "wi [%]", min_value=0.0, max_value=1000.0, step=0.1, value=20.0
)
LL = st.sidebar.number_input(
    "LL [%]", min_value=0.0, max_value=1000.0, step=0.1, value=102.0
)
Sand = st.sidebar.number_input(
    "Sand [%]", min_value=0.0, max_value=1000.0, step=0.1, value=2.0
)
PI = st.sidebar.number_input(
    "PI [%]", min_value=0.0, max_value=1000.0, step=0.1, value=82.0
)
C = st.sidebar.number_input(
    "C [%]", min_value=0.0, max_value=1000.0, step=0.1, value=85.0
)
A = st.sidebar.number_input("A", min_value=0.0, max_value=1000.0, step=0.1, value=0.96)
LP = st.sidebar.number_input(
    "LP [%]", min_value=0.0, max_value=1000.0, step=0.1, value=20.0
)

if st.sidebar.button("Predict Swelling Pressure"):
    if None in [A, Sand, C, LL, LP, PI, gamma_d, wi]:
        st.sidebar.error("Please fill in all fields.")
    else:
        input_data = np.array([[A, Sand, C, LL, LP, PI, gamma_d, wi]])
        try:
            input_data_scaled = scaler.transform(input_data)
            prediction = cat_model.predict(input_data_scaled)
            st.sidebar.success(
                f"Predicted Swelling Pressure (Pr [MPa]): {prediction[0]:.4f}"
            )
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")

# Main content
st.title("Swelling Pressure Prediction App")

# Display model information
st.header("Model and Data Information")
st.markdown("""
- **Model**: CatBoost Regressor
- **Algorithm**: Gradient Boosting
- **Dataset Size**: 288 samples
- **Training/Testing Split**: 70% training, 30% testing

### Model Accuracy
- **Training Performance**:
  - Average RMSE: 0.0000
  - Average R² Score: 0.9995
  - Average MAE: 0.0023

- **Testing Performance**:
  - Average RMSE: 0.0009
  - Average R² Score: 0.9659
  - Average MAE: 0.0183

### Geotechnical Relevance
Swelling pressure is the potential pressure exerted by clayey soils, especially when they absorb water and undergo volumetric expansion. Accurate prediction of this parameter is essential in geotechnical engineering for designing foundations, retaining structures, and other civil engineering applications involving expansive soils.
""")

st.header("Parametric Study")
params = {
    "γd [kN.m-3]": gamma_d,
    "wi [%]": wi,
    "LL [%]": LL,
    "Sand [%]": Sand,
    "PI [%]": PI,
    "C [%]": C,
    "A": A,
    "LP [%]": LP,
}

col1, col2 = st.columns([1, 3])

with col1:
    param_choice = st.selectbox("Select parameter:", list(params.keys()))
    param_min = st.number_input(
        f"Minimum {param_choice}", value=params[param_choice] - 1.0
    )
    param_max = st.number_input(
        f"Maximum {param_choice}", value=params[param_choice] + 1.0
    )
    num_points = st.slider("Number of points", min_value=10, max_value=100, value=50)
    poly_degree = st.slider("Polynomial degree", min_value=1, max_value=10, value=3)

    perform_study = st.button("Perform Study", use_container_width=True)

with col2:
    if perform_study:
        try:
            param_values = np.linspace(param_min, param_max, num_points)
            predictions = []
            fixed_values = params.copy()
            feature_order = [
                "A",
                "Sand [%]",
                "C [%]",
                "LL [%]",
                "LP [%]",
                "PI [%]",
                "γd [kN.m-3]",
                "wi [%]",
            ]
            for value in param_values:
                input_data = [fixed_values[feature] for feature in feature_order]
                input_data[feature_order.index(param_choice)] = value
                input_data = np.array([input_data])
                input_data_scaled = scaler.transform(input_data)
                prediction = cat_model.predict(input_data_scaled)
                predictions.append(prediction[0])

            coeffs = np.polyfit(param_values, predictions, poly_degree)
            poly = np.poly1d(coeffs)

            x_smooth = np.linspace(param_min, param_max, 200)
            y_smooth = poly(x_smooth)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(
                x_smooth,
                y_smooth,
                label=f"Polynomial fit (degree {poly_degree})",
                color="#FF4B4B",
                linewidth=2,
            )
            ax.set_xlabel(param_choice, fontsize=12)
            ax.set_ylabel("Predicted Swelling Pressure (Pr [MPa])", fontsize=12)
            ax.set_title(
                f"Parametric Study: {param_choice} vs Predicted Swelling Pressure",
                fontsize=14,
            )
            ax.grid(True)
            ax.legend(fontsize=10)
            ax.tick_params(axis="both", which="major", labelsize=10)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Parametric study failed: {e}")
    else:
        st.info("Click 'Perform Study' to see the results.")
