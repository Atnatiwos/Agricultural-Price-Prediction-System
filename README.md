# Agricultural Price Prediction System

## Project Overview
This project develops a machine learning-based system to predict agricultural commodity prices in Ethiopian markets. It leverages historical price data, seasonal factors, and other relevant features to provide accurate price forecasts, aiding farmers, traders, and policymakers in making informed decisions.

## Features
- Data Preprocessing: Handles raw WFP food prices data, including unit normalization and date conversions.
- Feature Engineering: Creates time-based features (Year, Month), seasonal indicators (Ethiopian seasons, rainfall index), and lagged price features (Previous Price, 3-Month Moving Average).
- Model Training: Implements and evaluates two regression models: Random Forest Regressor and XGBoost Regressor.
- Model Evaluation: Uses metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2) to assess model performance.
- Interactive Web Application: A Streamlit application for users to interact with the trained model and get price predictions.
- Google Drive Integration: Saves and loads models and feature lists directly from Google Drive for persistent storage.

## Data Source
The primary dataset used is the WFP Food Prices Data for Ethiopia, available on the Humanitarian Data Exchange:
- [WFP Food Prices Data](https://data.humdata.org/dataset/wfp-food-prices-for-ethiopia)

## Installation
To set up the project locally, follow these steps:

1.  Clone the repository:
    ```bash
•	    git clone <https://github.com/Atnatiwos/Agricultural-Price-Prediction-System
 >
    cd agricultural-price-prediction
    ```

2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib streamlit pyngrok
    ```

3.  Google Drive Setup (for Colab):
    If running in Google Colab, ensure your Google Drive is mounted and the `agriculture_project` folder structure is set up as in the notebook for model and data paths.

4.  Ngrok Authentication (for Streamlit):
    For external access to the Streamlit app, you'll need an ngrok auth token. Sign up at [ngrok.com](https://ngrok.com), get your auth token, and set it in your environment or directly in the Colab notebook:
    ```python
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
    ```

## Usage

### Running the Notebook
Open the provided `.ipynb` file in Google Colab or Jupyter Notebook and run all cells sequentially to execute the data processing, model training, and evaluation steps.

### Running the Streamlit App
Once the model is trained and saved, you can run the Streamlit application:

1.  Navigate to the `App` directory (or ensure `app.py` is in your current working directory if not using Colab's file structure):
    ```bash
    streamlit run app.py
    ```
    If running in Colab and exposing via ngrok:
    ```python
    !streamlit run /content/drive/MyDrive/Colab/agriculture_project/App/app.py &>/content/logs.txt &
    public_url = ngrok.connect(8501)
    print(public_url)
    ```
    This will provide a public URL to access the Streamlit application in your browser.

## Models

### Random Forest Regressor
- MAE: 2.005
- RMSE: 4.614
- R2 Score: 0.970

### XGBoost Regressor
- MAE: 1.861
- RMSE: 4.087
- R2 Score: 0.976

## Contributing
Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
