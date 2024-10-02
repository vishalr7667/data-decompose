# import os
# import pandas as pd
# from flask import Flask, render_template, request, flash, session, redirect, url_for
# from werkzeug.utils import secure_filename
# from werkzeug.exceptions import RequestEntityTooLarge

# app = Flask(__name__)
# app.secret_key = "supersecretkey"
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB file size limit

# # Ensure the upload folder exists
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# # Handle file size limit error
# @app.errorhandler(RequestEntityTooLarge)
# def handle_file_too_large(error):
#     flash('File size exceeds the allowed limit of 100 MB.')
#     return redirect(url_for('index'))

# # Allowed file extensions check
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Handle file upload
#         if 'file' not in request.files:
#             flash('No file part in the request.')
#             return redirect(request.url)

#         file = request.files['file']
#         if file.filename == '':
#             flash('No file selected')
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             try:
#                 file.save(filepath)

#                 # Read file based on extension
#                 if filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(filepath, encoding='utf-8')
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(filepath, encoding='ISO-8859-1')  # Fallback encoding
#                 elif filename.endswith('.xlsx'):
#                     df = pd.read_excel(filepath)

#                 # Preprocess the data
#                 df_cleaned = preprocess_data(df)
                
#                 # Perform correlation analysis
#                 correlation_matrix = perform_correlation(df_cleaned)
#                 if correlation_matrix.empty:
#                     flash('No numerical columns for correlation analysis.')
#                     return redirect(request.url)
                
#                 # Filter columns by positive correlation
#                 filtered_columns = filter_columns_by_correlation(correlation_matrix)
                
#                 # Debug: Print the filtered columns to verify filtering
#                 print(f"Filtered columns (positive correlation): {filtered_columns}")  # Debug line

#                 # Check if there are filtered columns
#                 if not filtered_columns:
#                     flash('No columns found with significant positive correlation.')
#                     return redirect(request.url)

#                 session['filtered_columns'] = filtered_columns

#                 flash('File uploaded and analyzed. Highly correlated columns selected.')
#                 return redirect(url_for('index'))

#             except Exception as e:
#                 flash(f"Error reading file: {e}")
#                 app.logger.error(f"Error occurred while processing the file: {e}", exc_info=True)
#                 return redirect(request.url)

#     # Only show filtered (positively correlated) columns
#     columns = session.get('filtered_columns', [])
#     return render_template('index.html', columns=columns)

# @app.route('/select_variables', methods=['POST'])
# def select_variables():
#     selected_columns = request.form.getlist('columns')
#     session['selected_columns'] = selected_columns
#     return redirect(url_for('decomposition'))

# @app.route('/decomposition')
# def decomposition():
#     selected_columns = session.get('selected_columns', [])
#     if not selected_columns:
#         flash('No variables selected for decomposition.')
#         return redirect(url_for('index'))
#     return render_template('decomposition.html', selected_columns=selected_columns)

# def preprocess_data(df):
#     # Optionally: Add more preprocessing steps like scaling, outlier removal, etc.
#     df_cleaned = df.dropna()  # Simple method, you can expand this
#     return df_cleaned

# def perform_correlation(df):
#     numerical_columns = df.select_dtypes(include=['number']).columns
#     if numerical_columns.empty:
#         flash("No numerical columns available for correlation analysis.")
#         return pd.DataFrame()
    
#     correlation_matrix = df[numerical_columns].corr(method='pearson')
    
#     # Debug: Print the correlation matrix to verify correctness
#     print(f"Correlation matrix:\n{correlation_matrix}")  # Debug line
    
#     return correlation_matrix

# def filter_columns_by_correlation(correlation_matrix, threshold=0.5):
#     """
#     Filter only columns that have positive correlation greater than the specified threshold.
#     """
#     positive_correlation_columns = []
    
#     # Iterate over each column and check the correlation values
#     for column in correlation_matrix.columns:
#         # Find the maximum positive correlation for each column (ignoring self-correlation which is always 1)
#         max_correlation = correlation_matrix[column][correlation_matrix[column] < 1].max()  # Exclude self-correlation
        
#         # Debug: Print the max correlation value for each column
#         print(f"Column '{column}' max positive correlation: {max_correlation}")  # Debug line
        
#         # Include only columns with positive correlation higher than the threshold
#         if max_correlation > threshold:
#             positive_correlation_columns.append(column)
    
#     # Debug: Print the final list of columns with positive correlation
#     print(f"Final list of positively correlated columns: {positive_correlation_columns}")  # Debug line
    
#     return positive_correlation_columns

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))



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
                
                # Perform correlation analysis
                correlation_matrix = perform_correlation(df_cleaned)
                if correlation_matrix.empty:
                    flash('No numerical columns for correlation analysis.')
                    return redirect(request.url)
                
                # Filter columns by positive correlation
                filtered_columns = filter_columns_by_correlation(correlation_matrix)
                
                # Check if there are filtered columns
                if not filtered_columns:
                    flash('No columns found with significant positive correlation.')
                    return redirect(request.url)

                session['filtered_columns'] = filtered_columns
                session['datafile_path'] = filepath  # Save file path instead of dataframe

                flash('File uploaded and analyzed. Highly correlated columns selected with threshold > 0.5. Contains only positive related columns.')
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
    columns = session.get('filtered_columns', [])
    return render_template('index.html', columns=columns)

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

# def decomposition():
#     selected_columns = session.get('selected_columns', [])
#     decomposition_method = session.get('decomposition_method', None)
#     datafile_path = session.get('datafile_path', None)

#     # Capture and convert additional parameters
#     wavelet_type = session.get('wavelet_type', None)
#     level = session.get('level', None)
#     alpha = session.get('alpha', None)

#     # Convert level and alpha to integers or floats
#     # if level is not None:
#     #     level = int(level)  # Convert level to an integer
        

#     # if alpha is not None:
#     #     alpha = float(alpha)  # Convert alpha to a float
        

#     # Continue with the rest of your logic...
#     if not selected_columns:
#         flash('No variables selected for decomposition.')
#         return redirect(url_for('index'))

#     if decomposition_method is None:
#         flash('No decomposition method selected.')
#         return redirect(url_for('index'))

#     if datafile_path is None:
#         flash('No data found for decomposition.')
#         return redirect(url_for('index'))

#     # Reload the DataFrame from the file
#     if datafile_path.endswith('.csv'):
#         df = pd.read_csv(datafile_path)
#     elif datafile_path.endswith('.xlsx'):
#         df = pd.read_excel(datafile_path)

#     # Extract the selected columns for decomposition
#     data = df[selected_columns].values

#     # Perform the decomposition
#     theoretical_result, plot_url = perform_decomposition(data, decomposition_method, wavelet_type, level, alpha)

#     return render_template(
#         'decomposition.html',
#         selected_columns=selected_columns,
#         decomposition_method=decomposition_method,
#         theoretical_result=theoretical_result,
#         plot_url=plot_url
#     )


# def perform_decomposition(data, method, wavelet_type=None, level=None, alpha=None):
#     print('Data for selected column',data)
#     theoretical_result = ""
#     level = int(level)
#     coeffs = []
#     # Perform the selected decomposition method
#     if method == 'wavelet_decomposition':
#         import pywt  # Importing wavelet library
#         coeffs = pywt.wavedec(data, wavelet_type, level=level)
#         theoretical_result = f"Wavelet Decomposition Result with {wavelet_type} at level {level}."

#         # Plot the approximation and details
#         # plt.figure(figsize=(10, 5))
#         # plt.plot(data, label='Original Signal')
#         # for i, coeff in enumerate(coeffs):
#         #     plt.plot(coeff, label=f'Coeff {i}')
#         # plt.title('Wavelet Decomposition Example')
#         # plt.legend()
#         num_coeffs = len(coeffs)  # Number of coefficients (including approximation)
#         fig, axs = plt.subplots(num_coeffs + 1, 1, figsize=(10, 5 * (num_coeffs + 1)))

#         # Plot original signal
#         axs[0].plot(data, label='Original Signal')
#         axs[0].set_title('Original Signal')
#         axs[0].legend()

#         # Plot each coefficient
#         for i in range(num_coeffs):
#             axs[i + 1].plot(coeffs[i], label=f'Coefficient {i}')
#             axs[i + 1].set_title(f'Wavelet Coefficient {i}')
#             axs[i + 1].legend()

#     # elif method == 'wavelet_packet_decomposition':
#     #     import pywt  # Importing wavelet library
#     #     wp = pywt.WaveletPacket(data, wavelet_type, mode='symmetric', maxlevel=level)
#     #     theoretical_result = f"Wavelet Packet Decomposition Result with {wavelet_type} at level {level}."

#     #     # Plot the original signal
#     #     plt.figure(figsize=(10, 5))
#     #     plt.plot(data, label='Original Signal')
#     #     for i, node in enumerate(wp.get_level(level, 'natural')):
#     #             # Create a custom label if node.name is empty
#     #         node_label = node.name if node.name else f'Node {i}'
#     #     plt.plot(node.data, label=node_label)
#     #     plt.title('Wavelet Packet Decomposition Example')
#     #     plt.legend()
        
#     elif method == 'wavelet_packet_decomposition':
#             import pywt  # Importing wavelet library
#             wp = pywt.WaveletPacket(data, wavelet_type, mode='symmetric', maxlevel=level)
#             theoretical_result = f"Wavelet Packet Decomposition Result with {wavelet_type} at level {level}."

#             # Plot the original signal
#             plt.figure(figsize=(10, 5))
#             plt.plot(data, label='Original Signal')

#             # Loop through the nodes and plot their data
#             for i, node in enumerate(wp.get_level(level, 'natural')):
#                 # Create a custom label if node.name is empty
#                 node_label = f'Node {i} (Path: {node.path})'
#                 plt.plot(node.data, label=node_label)  # Make sure this is inside the loop

#             plt.title('Wavelet Packet Decomposition Example')
#             plt.legend()
#             plt.show()  # Don't forget to call plt.show() to display the plot

#     elif method == 'empirical_mode_decomposition':
#         from PyEMD import EMD  # Importing EMD library
#         emd = EMD()
#         IMFs = emd(data)
#         theoretical_result = "Empirical Mode Decomposition Result."

#         # Plot the IMFs
#         plt.figure(figsize=(10, 5))
#         plt.plot(data, label='Original Signal')
#         for i, imf in enumerate(IMFs):
#             plt.plot(imf, label=f'IMF {i + 1}')
#         plt.title('Empirical Mode Decomposition Example')
#         plt.legend()

#     elif method == 'variational_mode_decomposition':
#         import VMD  # Import the VMD library
#         theoretical_result = f"Variational Mode Decomposition Result with alpha = {alpha}."
        
#         # Perform VMD
#         u, _, _ = VMD.VMD(data, alpha=alpha)
        
#         # Plot the modes
#         plt.figure(figsize=(10, 5))
#         plt.plot(data, label='Original Signal')
#         for i in range(u.shape[0]):
#             plt.plot(u[i], label=f'Mode {i + 1}')
#         plt.title('Variational Mode Decomposition Example')
#         plt.legend()

#     # Save the plot to a BytesIO object and encode it in base64
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
#     plt.close()  # Close the plot to avoid displaying it multiple times

#     return theoretical_result, f"data:image/png;base64,{plot_url}"

def perform_decomposition(data, method, wavelet_type=None, level=None, alpha=None, tau=None, K=None, DC=None, init=None, tol=None):
    print('Data for selected column', data)
    print('Method selected', method)
    
    # Convert the data to a numpy array if it's not already
    data = np.array(data)
    
    # Ensure data is 1-dimensional
    if len(data.shape) > 1:
        data = data.flatten()

    # Handle NaN values by replacing them with the mean of the data
    if np.isnan(data).any():
        data_mean = np.nanmean(data)
        data = np.where(np.isnan(data), data_mean, data)

    # Calculate statistics for the original data
    data_mean = np.mean(data)
    data_max = np.max(data)
    data_min = np.min(data)
    data_range = data_max - data_min

    # Normalize the data (scaling between 0 and 1)
    data_normalized = (data - data_min) / data_range if data_range != 0 else data

    theoretical_result = (
        f"Original Data: Mean = {data_mean}, Max = {data_max}, Min = {data_min}. "
        f"Normalization (scaled between 0 and 1): [{data_normalized.min()}, {data_normalized.max()}].\n"
    )

    # Convert level and alpha to appropriate types
    level = int(level) if level is not None else None
    alpha = float(alpha) if alpha is not None else None

    coeffs = []
    
    # Perform the selected decomposition method
    if method == 'wavelet_decomposition':
        import pywt
        import pandas as pd
        
        coeffs = pywt.wavedec(data, wavelet_type, level=level)
        theoretical_result += f"Wavelet Decomposition Result with {wavelet_type} at level {level}.\n"
        
        # Create a summary for each set of wavelet coefficients
        wavelet_summaries = [pd.Series(coeff).describe().to_string() for coeff in coeffs]
        # Append the summaries to the theoretical result
        for i, summary in enumerate(wavelet_summaries):
            theoretical_result += f"\nLevel {i} Coefficients Summary:\n{summary}\n"

        # Plotting logic for wavelet decomposition remains the same
        num_coeffs = len(coeffs)
        fig, axs = plt.subplots(num_coeffs + 1, 1, figsize=(10, 5 * (num_coeffs + 1)))
        axs[0].plot(data, label='Original Signal')
        axs[0].set_title('Original Signal')
        axs[0].legend()

        for i in range(num_coeffs):
            axs[i + 1].plot(coeffs[i], label=f'Coefficient {i}')
            axs[i + 1].set_title(f'Wavelet Coefficient {i}')
            axs[i + 1].legend()
            axs[i + 1].set_ylim(coeffs[i].min(), coeffs[i].max())  # Set limits tightly around the coefficient data
            axs[i + 1].set_xticks([])
        
        
        plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.05)  # Adjust spacing to minimum

        # Use tight layout to remove any additional spacing
        plt.tight_layout()  # Zero padding to minimize space

        # Display the plot
        plt.show()

    elif method == 'wavelet_packet_decomposition':
        import pywt
        import pandas as pd
        
        wp = pywt.WaveletPacket(data, wavelet_type, mode='symmetric', maxlevel=level)
        theoretical_result += f"Wavelet Packet Decomposition Result with {wavelet_type} at level {level}.\n"
        
        # Prepare summaries for each level
        wavelet_summaries = []
        
        for lvl in range(1, level + 1):
            nodes = wp.get_level(lvl, 'natural')
            theoretical_result += f"\nLevel {lvl} Summary:\n"

            for i, node in enumerate(nodes):
                node_data = node.data
                node_label = f'Node {i} (Path: {node.path})'
                node_summary = pd.Series(node_data).describe().to_string()
                theoretical_result += f"{node_label}:\n{node_summary}\n"
                wavelet_summaries.append((node_label, node_data))

        num_nodes = len(wavelet_summaries)
        fig, axs = plt.subplots(num_nodes + 1, 1, figsize=(10, 5 * (num_nodes + 1)))
        axs[0].plot(data, label='Original Signal')
        axs[0].set_title('Original Signal')
        axs[0].legend()

        for i, (node_label, node_data) in enumerate(wavelet_summaries):
            axs[i + 1].plot(node_data, label=node_label)
            axs[i + 1].set_title(node_label)
            axs[i + 1].legend()
            axs[i + 1].set_ylim(node_data.min(), node_data.max())  # Set limits tightly around the node data
            axs[i + 1].set_xticks([])  # Optionally hide x-ticks
            
        plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.05)  # Adjust spacing to minimum

        # Use tight layout to remove any additional spacing
        plt.tight_layout()  # Zero padding to minimize space

        # Display the plot
        plt.show()

    elif method == 'emd':
        from PyEMD import EMD
        
        emd = EMD()
        IMFs = emd.emd(data, max_imf=5)

        if IMFs is None or len(IMFs) == 0:
            theoretical_result += "EMD failed to decompose the signal. Please check your data.\n"
        else:
            theoretical_result += "Empirical Mode Decomposition Result.\n"
            imf_mean = np.mean(IMFs[0])
            imf_max = np.max(IMFs[0])
            imf_min = np.min(IMFs[0])

            theoretical_result += (
                f"First IMF: Mean = {imf_mean}, Max = {imf_max}, Min = {imf_min}.\n"
            )

            num_imfs = len(IMFs)
            fig, axs = plt.subplots(num_imfs + 1, 1, figsize=(10, 5 * (num_imfs + 1)))
            axs[0].plot(data, label='Original Signal')
            axs[0].set_title('Original Signal')
            axs[0].legend()

            for i, imf in enumerate(IMFs):
                axs[i + 1].plot(imf, label=f'IMF {i + 1}')
                axs[i + 1].set_title(f'IMF {i + 1}')
                axs[i + 1].legend()
                axs[i + 1].set_ylim(imf.min(), imf.max())  # Set limits tightly around the data
                axs[i + 1].set_xticks([]) 
                
        plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.05)  # Adjust spacing to minimum

        # Use tight layout to remove any additional spacing
        plt.tight_layout()  # Zero padding to minimize space

        # Display the plot
        plt.show()
        
    elif method == 'variational_mode_decomposition':
        import vmdpy
        import pandas as pd
        
        theoretical_result += f"Variational Mode Decomposition Result with alpha = {alpha}.\n"

        # Ensure VMD parameters are defined
        tau = float(tau) if tau is not None else 0.0
        K = int(K) if K is not None else 3
        DC = int(DC) if DC is not None else 0
        init = int(init) if init is not None else 1
        tol = 1e-6
        # tol = float(tol) if tol is not None else 1e-6
        print(tol);
        data = data.astype(np.float32)

        batch_size = 1000
        results = []

        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            u, _, _ = vmdpy.VMD(batch_data, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
            results.append(u)

        mode_summaries = []
        theoretical_result += f"\nMode summaries for the last processed batch:\n"
        
        for i, mode in enumerate(results[-1]):
            mode_label = f'Mode {i + 1}'
            mode_summary = pd.Series(mode).describe().to_string()
            theoretical_result += f"{mode_label}:\n{mode_summary}\n"
            mode_summaries.append((mode_label, mode))

        num_modes = len(mode_summaries)
        fig, axs = plt.subplots(num_modes + 1, 1, figsize=(10, 5 * (num_modes + 1)))
        axs[0].plot(data, label='Original Signal')
        axs[0].set_title('Original Signal')
        axs[0].legend()

        for i, (mode_label, mode_data) in enumerate(mode_summaries):
            axs[i + 1].plot(mode_data, label=mode_label)
            axs[i + 1].set_title(mode_label)
            axs[i + 1].legend()

    # Save the plot to a BytesIO object and encode it in base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()  # Close the plot to avoid displaying it multiple times

    return theoretical_result, f"data:image/png;base64,{plot_url}"



def preprocess_data(df):
    # Optionally: Add more preprocessing steps like scaling, outlier removal, etc.
    df_cleaned = df.dropna()  # Simple method, you can expand this
    return df_cleaned

def perform_correlation(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    if numerical_columns.empty:
        flash("No numerical columns available for correlation analysis.")
        return pd.DataFrame()

    correlation_matrix = df[numerical_columns].corr(method='pearson')

    return correlation_matrix

def filter_columns_by_correlation(correlation_matrix, threshold=0.5):
    positive_correlation_columns = []

    for column in correlation_matrix.columns:
        max_correlation = correlation_matrix[column][(correlation_matrix[column] > 0) & (correlation_matrix[column] < 1)].max()

        if max_correlation and max_correlation > threshold:
            positive_correlation_columns.append(column)

    return positive_correlation_columns

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
