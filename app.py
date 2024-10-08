
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
from flask import Flask, render_template, request, flash, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau #type: ignore
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.losses import MeanSquaredError
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_log_error

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB file size limit

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Handle file size limit error
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash('File size exceeds the allowed limit of 100 MB.')
    app.logger.warning("Attempted file upload exceeds limit.")
    return redirect(url_for('index'))

# Allowed file extensions check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)

                # Read file based on extension
                if filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='ISO-8859-1')  # Fallback encoding
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)

                # Preprocess the data
                df_cleaned = preprocess_data(df)
                # print('propna method data', df_cleaned)
                # Apply Min-Max normalization
                df_normalized = apply_min_max_normalization(df_cleaned)
                # print('data frame after apply min max', df_normalized)
                # Perform correlation analysis
                correlation_results = perform_correlation(df_normalized, target_column='windmill_generated_power(kW/h)')
                
                if correlation_matrix.empty:
                    flash('No numerical columns for correlation analysis.')
                    return redirect(request.url)
                
                # Filter columns by positive correlation
                # filtered_columns,max_included_value,excluded_values_dict, excluded_columns,excluded_max_value = filter_columns_by_correlation(correlation_matrix)
                
                # print("exclude values",excluded_values_dict)
                
                # # Check if there are filtered columns
                # if not filtered_columns:
                #     flash('No columns found with significant positive correlation.')
                #     return redirect(request.url)

                # session['filtered_columns'] = filtered_columns
                # session['max_included_value'] = max_included_value
                # session['excluded_values_dict'] = excluded_values_dict  # Updated key
                # session['excluded_columns'] = excluded_columns
                # session['max_excluded_value'] = excluded_max_value
                # session['datafile_path'] = filepath  # Save file path instead of dataframe
                session['columns'] = df_cleaned.select_dtypes(include=['number']).columns.tolist()
                session['correlation_results'] = correlation_results.to_dict()  # Convert to dictionary
                
                session['datafile_path'] = filepath  # Save file path instead of dataframe
                flash('File uploaded and analyzed. Highly correlated columns selected with threshold > 0.2. Contains only positive related columns.')
                return redirect(url_for('index'))

            except pd.errors.EmptyDataError:
                flash("Uploaded file is empty or invalid.")
            except pd.errors.ParserError:
                flash("Error parsing the file. Ensure the file format is correct.")
            except Exception as e:
                flash(f"Error reading file: {e}")
                app.logger.error(f"Error occurred while processing the file: {e}", exc_info=True)
                return redirect(request.url)

    # Only show filtered (positively correlated) columns
    columns = session.get('columns',[])
    correlation_data = session.get('correlation_results',{})
    
    return render_template('index.html', columns = columns, correlation_data = correlation_data)

@app.route('/select_variables', methods=['POST'])
def select_variables():
    
    selected_columns = request.form.getlist('columns')
    decomposition_method = request.form.get('decomposition')  # Get the selected decomposition method
    session['selected_columns'] = selected_columns
    session['decomposition_method'] = decomposition_method  # Store the selected method in the session

    # Capture dynamic parameters based on selected method
    if decomposition_method in ['wavelet_decomposition', 'wavelet_packet_decomposition']:
        wavelet_type = request.form.get('wavelet')  # Get wavelet type
        level = int(request.form.get('level'))  # Get decomposition level
        session['wavelet_type'] = wavelet_type
        session['level'] = level

    elif decomposition_method == 'variational_mode_decomposition':
        alpha = float(request.form.get('alpha'))  # Get alpha value
        tau = float(request.form.get('tau'))  # Get tau value
        K = int(request.form.get('K'))  # Get number of modes (K)
        DC = int(request.form.get('DC'))  # Get DC value (0 or 1)
        init = int(request.form.get('init'))  # Get initialization value
        # tol = float(request.form.get('tol'))  # Get tolerance value

        # Store all VMD parameters in the session
        session['alpha'] = alpha
        session['tau'] = tau
        session['K'] = K
        session['DC'] = DC
        session['init'] = init
        # session['tol'] = tol
        
    
    
    return redirect(url_for('decomposition'))
    # return redirect(url_for('decomposition'))

@app.route('/decomposition')
def decomposition():
    selected_columns = session.get('selected_columns', [])
    decomposition_method = session.get('decomposition_method', None)
    datafile_path = session.get('datafile_path', None)

    print('selected column',selected_columns)
    # Capture and convert additional parameters
    wavelet_type = session.get('wavelet_type', None)
    level = session.get('level', None)
    alpha = session.get('alpha', None)
    
    # Variational Mode Decomposition (VMD) parameters
    tau = session.get('tau', None)
    K = session.get('K', None)
    DC = session.get('DC', None)
    init = session.get('init', None)
    # tol = session.get('tol', None)

    # Convert level and alpha to integers or floats
    if level is not None:
        level = int(level)  # Convert level to an integer
    if alpha is not None:
        alpha = float(alpha)  # Convert alpha to a float
    if tau is not None:
        tau = float(tau)  # Convert tau to a float
    if K is not None:
        K = int(K)  # Convert K (number of modes) to an integer
    if DC is not None:
        DC = int(DC)  # Convert DC to an integer
    if init is not None:
        init = int(init)  # Convert init to an integer
    # if tol is not None:
    #     tol = float(tol)  # Convert tolerance to a float

    # Continue with the rest of your logic...
    if not selected_columns:
        flash('No variables selected for decomposition.')
        return redirect(url_for('index'))

    if decomposition_method is None:
        flash('No decomposition method selected.')
        return redirect(url_for('index'))

    if datafile_path is None:
        flash('No data found for decomposition.')
        return redirect(url_for('index'))

    # Reload the DataFrame from the file
    if datafile_path.endswith('.csv'):
        df = pd.read_csv(datafile_path)
    elif datafile_path.endswith('.xlsx'):
        df = pd.read_excel(datafile_path)

    # Extract the selected columns for decomposition
    data = df[selected_columns].values

    print('data of selected column', data)
    
    # Perform the decomposition
    if decomposition_method == 'variational_mode_decomposition':
        theoretical_result, plot_url = perform_decomposition(
            data, decomposition_method, alpha=alpha, tau=tau, K=K, DC=DC, init=init
        )
    else:
        # Handle wavelet-based decompositions
        theoretical_result, plot_url = perform_decomposition(
            data, decomposition_method, wavelet_type=wavelet_type, level=level
        )
    return render_template(
        'decomposition.html',
        selected_columns=selected_columns,
        decomposition_method=decomposition_method,
        theoretical_result=theoretical_result,
        plot_url=plot_url
    )





def perform_decomposition(data, method, wavelet_type=None, level=None, alpha=None, tau=None, K=None, DC=None, init=None, tol=None):
    
    tau = float(tau) if tau is not None else 0.0
    K = int(K) if K is not None else 3
    DC = int(DC) if DC is not None else 0
    init = int(init) if init is not None else 1
    tol = 1e-6
    # Ensure data is 1-dimensional
    if len(data.shape) > 1:
        data = data.flatten()

    # Handle NaN values by replacing them with the mean of the data
    if np.isnan(data).any():
        data_mean = np.nanmean(data)
        data = np.where(np.isnan(data), data_mean, data)

    # Normalize the data (scaling between 0 and 1)
    data_min, data_max = np.min(data), np.max(data)
    if data_min == data_max:
        data_normalized = np.zeros_like(data)
    else:
        data_normalized = (data - data_min) / (data_max - data_min)

    # Decompose the data based on the selected method
    if method == 'wavelet_decomposition':
        import pywt
        coeffs = pywt.wavedec(data_normalized, wavelet_type, level=level)
        decomposition_result = coeffs[0]  # Approximation coefficients

    elif method == 'wavelet_packet_decomposition':
        import pywt
        wp = pywt.WaveletPacket(data_normalized, wavelet_type, maxlevel=level)
        decomposition_result = wp['a' * level].data  # Approximation coefficients at the selected level
        
    elif method == 'emd':
        from PyEMD import EMD
        emd = EMD()
        IMFs = emd.emd(data_normalized)
        decomposition_result = IMFs[0]  # First IMF
        
    elif method == 'variational_mode_decomposition':
        import vmdpy
        decomposition_result, _, _ = vmdpy.VMD(data_normalized, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
        decomposition_result = decomposition_result[0]  # First decomposed mode

    else:
        raise ValueError("Unknown decomposition method")

    # Prepare the data for CNN prediction (reshaping into 3D array for CNN)
    X = np.arange(len(decomposition_result)).reshape(-1, 1)  # Time steps
    y = decomposition_result  # The decomposed result as target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input data for CNN (samples, timesteps, features)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build a simple CNN for regression
    # cnn_model = Sequential()

    # # Add convolutional layer
    # cnn_model.add(Conv1D(filters=32, kernel_size=1, activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], 1)))

    # # Add max pooling layer
    # cnn_model.add(MaxPooling1D(pool_size=1))

    # # # Optionally, you can add more convolution and pooling layers for more complex patterns:
    # # cnn_model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    # # cnn_model.add(MaxPooling1D(pool_size=2))

    # # Flatten the output
    # cnn_model.add(Flatten())

    # # Add fully connected layers for regression
    # cnn_model.add(Dense(100, activation='relu'))
    # cnn_model.add(Dense(1))  # Output layer for regression (one continuous output)

    # # Compile the model
    # cnn_model.compile(optimizer='adam', loss='mean_squared_error')

    # # Train the CNN model
    # cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=20, batch_size=32, verbose=1)

    # Build a revised CNN for regression
    cnn_model = Sequential()

    # Convolutional layers
    cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], 1)))
    if X_train_cnn.shape[1] > 1:
        cnn_model.add(MaxPooling1D(pool_size=2))  # Only use pooling if time_steps > 1
    else:
        print("Warning: Not enough time steps for pooling. Skipping MaxPooling1D.")
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    if X_train_cnn.shape[1] > 1:
        cnn_model.add(MaxPooling1D(pool_size=2))  # Only use pooling if time_steps > 1
    else:
        print("Warning: Not enough time steps for pooling. Skipping MaxPooling1D.")
    cnn_model.add(Dropout(0.5))

    # Flatten the output
    cnn_model.add(Flatten())

    # Fully connected layers
    cnn_model.add(Dense(100, activation='relu'))
    cnn_model.add(Dense(1))  # Output layer for regression

    # Compile the model
    cnn_model.compile(optimizer='adam', loss='mean_squared_error')

    
    # Define the learning rate reduction callback
    lr_reduction = ReduceLROnPlateau(
        monitor='val_loss',    # Monitor the validation loss
        factor=0.5,            # Reduce the learning rate by 50%
        patience=5,            # Wait for 5 epochs before reducing the learning rate
        min_lr=1e-6            # Set a minimum learning rate
    )
    
    # Train the model with callbacks
    cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=20, batch_size=32, verbose=1, callbacks=[lr_reduction])
    
    # Make predictions using CNN
    y_pred_cnn = cnn_model.predict(X_test_cnn)

    # Calculate evaluation metrics
    mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
    rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
    epsilon = 1e-10
    mape_cnn = np.mean(np.abs((y_test - y_pred_cnn.flatten()) / (y_test + epsilon))) * 100
    mse = mean_squared_error(y_test, y_pred_cnn)

    # Plot the original data vs CNN prediction
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_test, label='True Data', marker='o')
    plt.plot(X_test, y_pred_cnn, label='CNN Predicted Data', linestyle='--', marker='x')
    plt.title(f'CNN Prediction after {method.capitalize()}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a BytesIO object and encode it in base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    # Return evaluation metrics and the plot
    metrics_result_cnn = f"""
    CNN MAE: {mae_cnn:.4f}
    CNN RMSE: {rmse_cnn:.4f}
    CNN MSE: {mse:.4f}
    """

    return metrics_result_cnn, f"data:image/png;base64,{plot_url}"



def preprocess_data(df):
    # Optionally: Add more preprocessing steps like scaling, outlier removal, etc.
    df_cleaned = df.dropna()  # Simple method, you can expand this
    return df_cleaned

def perform_correlation(df, target_column='windmill_generated_power(kW/h)'):
    print("Data passed to perform_correlation (df):")
    print(df.head())  # Print the data passed to the function to verify it's normalized
    # Select numeric columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Ensure that the target column exists in the dataframe
    if target_column not in numerical_columns:
        flash(f"Target column {target_column} not found in the dataset.")
        return pd.DataFrame()

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_columns].corr(method='pearson')

    # Get correlations for the target column with all other numeric columns
    correlation_with_target = correlation_matrix[target_column].drop(target_column)  # Exclude self-correlation

    # Return a dictionary of correlation values for the target column
    return correlation_with_target.sort_values(ascending=False)

def filter_columns_by_correlation(correlation_matrix, threshold=0.2):
    included_columns = []  # To store columns that meet the threshold
    max_included_value = []  # To store max correlation values for included columns
    excluded_columns = []  # To store columns that don't meet the threshold (only column names)
    excluded_max_value = []  # To store max correlation for excluded columns
    excluded_values_dict = {}  # Dictionary to store excluded values per column

    for column in correlation_matrix.columns:
        # Find all positive correlations excluding self-correlation
        positive_correlations = correlation_matrix[column][(correlation_matrix[column] > 0) & (correlation_matrix[column] < 1)]

        # Find the max positive correlation
        max_correlation = positive_correlations.max()

        # Separate excluded correlations (those below threshold)
        excluded_values = positive_correlations[positive_correlations <= threshold].tolist()

        if max_correlation and max_correlation > threshold:
            # Include the column with its max correlation and excluded values
            included_columns.append(column)
            max_included_value.append(max_correlation)
            
            excluded_values_dict[column] = excluded_values
        else:
            # Exclude the column by just appending its name
            excluded_columns.append(column)
            excluded_max_value.append(max_correlation)
            # excluded_values_dict[column] = excluded_values

    return included_columns, max_included_value, excluded_values_dict, excluded_columns, excluded_max_value



@app.route('/clear_session', methods=['POST'])
def clear_session():
    # Clear specific session keys or all session data
    session.pop('filtered_columns', None)  # Clear the selected columns from session
    session.pop('max_included_value',None)
    session.pop('excluded_values_dict',None)
    session.pop('excluded_columns',None)
    session.pop('max_excluded_value',None)

    # You can also clear other session data if needed
    return '', 204  # Return a successful response with no content


def apply_min_max_normalization(df):
    numerical_columns = df.select_dtypes(include=['number']).columns  # Only normalize numerical columns
    scaler = MinMaxScaler()  # Initialize the scaler
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])  # Fit and transform the data
    return df
    
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
@app.template_filter('rounding')
def rounding(value, precision=2):
    return round(value, precision)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
