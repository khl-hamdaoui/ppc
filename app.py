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
# Display model information
st.header("Model and Data Information")
st.markdown("""
### Model Details
- **Model Type**: The model used in this application is a **CatBoost Regressor**, which is an implementation of gradient boosting on decision trees. It is well-suited for regression tasks, especially with tabular data, where relationships between features may be complex.
- **Algorithm**: The underlying algorithm, **Gradient Boosting**, iteratively builds an ensemble of decision trees, with each tree trained to correct the errors made by the previous ones. This approach enhances the model's accuracy and generalization ability.

### Dataset Information
- **Dataset Size**: The dataset consists of **288 samples**, which include a variety of soil properties relevant to predicting swelling pressure.
- **Features Used**: The model utilizes 9 input features that describe the characteristics of the soil, such as water content, liquid limit, plasticity index, sand and clay content, among others.
- **Training/Testing Split**: The dataset was split into **70% for training** (202 samples) and **30% for testing** (86 samples). The training set is used to fit the model, while the testing set evaluates its performance on unseen data.

### Model Performance Metrics
#### Training Performance
- **Root Mean Square Error (RMSE)**: **0.0000**. This indicates that the model fits the training data almost perfectly, with negligible prediction errors.
- **R² Score**: **0.9995**. The high R² value suggests that the model explains nearly all the variability in the training data.
- **Mean Absolute Error (MAE)**: **0.0023**. This low value confirms the model's accuracy in predicting the swelling pressure on the training set.

#### Testing Performance
- **Root Mean Square Error (RMSE)**: **0.0009**. The small RMSE on the testing set indicates that the model maintains high accuracy when predicting on new data.
- **R² Score**: **0.9659**. This R² value implies that the model still captures a substantial portion of the variability in the testing data, although there is a slight reduction compared to the training set.
- **Mean Absolute Error (MAE)**: **0.0183**. The low MAE value for testing indicates that the model’s predictions remain close to the actual values, even for new data.

### Geotechnical Relevance of Swelling Pressure Prediction
- **Swelling Pressure**: This is a critical parameter for clayey soils, which tend to swell when they absorb water. Swelling pressure can exert significant forces on structures, leading to potential damage if not properly accounted for.
- **Importance in Engineering**: Predicting swelling pressure is essential for **foundation design**, **retaining wall stability**, **road construction**, and other civil engineering projects where expansive soils may be present.
- **Influencing Factors**: Several soil properties, including **water content (wi)**, **dry density (γd)**, **liquid limit (LL)**, **plasticity index (PI)**, **sand content**, and **clay content (C)**, impact the swelling behavior. The model uses these features to make accurate predictions, aiding in risk assessment and engineering decision-making.
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
# Credit Section
st.header("Credits")
st.markdown("""
This application was developed by **PhD student Hamdaoui Khaled** from the **Lab of Geomaterials, University of Chlef, Algeria**.
""")
