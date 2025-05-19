from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_digits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
import json
import uuid
import io
import logging
import subprocess
import sys
import traceback
import base64
import gc  # Add garbage collection


import shlex
import signal

from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base directory for datasets (inside backend folder)
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

# Directory to store trained models
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset(dataset_name):
    """Loads a dataset from predefined ones or uploaded CSV files."""
    predefined_datasets = {
        "iris": load_iris(as_frame=True).frame,
        "wine": load_wine(as_frame=True).frame,
        "digits": load_digits(as_frame=True).frame
    }
    
    if dataset_name in predefined_datasets:
        return predefined_datasets[dataset_name]
    
    dataset_path = os.path.join(DATASET_DIR, f"{dataset_name}.csv")
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    
    return None

@app.route('/')
def home():
    return jsonify({"message": "Flask API is Running!"})

@app.route('/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    filename = file.filename
    
    if not filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    file_path = os.path.join(DATASET_DIR, filename)

    if os.path.exists(file_path):
        return jsonify({"error": "Dataset already exists"}), 400

    file.save(file_path)
    
    return jsonify({"message": f"Dataset {filename} uploaded successfully", "file_path": file_path})

@app.route("/datasets", methods=["GET"])
def list_datasets():
    available_datasets = ["iris", "wine", "digits"] + [f[:-4] for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    return jsonify({"datasets": available_datasets})

@app.route("/dataset-info", methods=["GET"])
def dataset_info():
    dataset_name = request.args.get("name")  # Get dataset from query params
    
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400  # Handle missing dataset
    
    df = load_dataset(dataset_name)
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404  # Handle dataset not found
    
    info = {
        "columns": list(df.columns),
        "data_types": df.dtypes.apply(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "shape": df.shape
    }
    
    return jsonify(info)
    
@app.route('/dataset-preview', methods=['GET'])
def dataset_preview():
    try:
        dataset_name = request.args.get("name")  # Get dataset name from query parameters
        if not dataset_name:
            return jsonify({"error": "Dataset name is required"}), 400
        
        dataset = load_dataset(dataset_name)  # Load dataset using your function
        
        if dataset is None:
            return jsonify({"error": "Dataset not found"}), 404
        
        return jsonify({"dataset": dataset.to_dict(orient="records")})  # Convert DataFrame to JSON
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/execute', methods=['POST'])
def execute_code():
    data = request.json
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    predefined_datasets = {
        "iris": load_iris(as_frame=True).frame,
        "wine": load_wine(as_frame=True).frame,
        "digits": load_digits(as_frame=True).frame
    }
    
    uploaded_datasets = {}
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".csv"):
            dataset_name = filename[:-4]
            uploaded_datasets[dataset_name] = pd.read_csv(os.path.join(DATASET_DIR, filename))
    
    available_datasets = {**predefined_datasets, **uploaded_datasets}
    
    safe_globals = {
        "__builtins__": { 
            "len": len, "range": range,
            "sum": sum, "min": min, "max": max, "sorted": sorted,
            "abs": abs, "round": round, "divmod": divmod,
            "enumerate": enumerate, "map": map, "filter": filter, "zip": zip,
            "__import__": __import__ 
        },
        "pd": pd,
        "np": np,
        "plt": plt,
        "datasets": available_datasets
    }
    safe_locals = {}
    
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    def custom_print(*args):
        """Custom print function to capture output without brackets."""
        formatted_output = ""
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                formatted_output += json.dumps(arg.to_dict(orient="records"), indent=4) + "\n"
            elif isinstance(arg, pd.Series):
                formatted_output += json.dumps(arg.to_dict(), indent=4) + "\n"
            else:
                formatted_output += str(arg) + "\n"
        
        output_buffer.write(formatted_output.strip() + "\n")

    # Store the original plt.show function
    original_plt_show = plt.show
    
    # Initialize plot data in function scope
    plot_data = None

    # Define a function to capture plot
    def capture_plot():
        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            encoded_plot = base64.b64encode(buf.read()).decode('utf-8')
            return encoded_plot
        except Exception as e:
            logging.error(f"Error capturing plot: {str(e)}")
            return None
        finally:
            buf.close()
    
    # Override plt.show to capture plot
    def custom_plt_show(*args, **kwargs):
        nonlocal plot_data
        plot_data = capture_plot()
        plt.close('all')  # Properly close all figures
        
    safe_globals["plt"].show = custom_plt_show
    safe_globals["print"] = custom_print
    
    try:
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect output
        sys.stdout = output_buffer
        sys.stderr = error_buffer
        
        # Execute the code
        exec(code, safe_globals, safe_locals)
        
        # Check if there are figures but plot_data wasn't set (user didn't call plt.show)
        if plt.get_fignums() and plot_data is None:
            plot_data = capture_plot()
            
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"error": error_message.strip()}), 400
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Ensure all plots are closed to prevent memory leaks
        plt.close('all')
        
        # Force garbage collection to clean up memory
        gc.collect()
        
    output = output_buffer.getvalue().strip()
    errors = error_buffer.getvalue().strip()
    
    # Clean up buffers
    output_buffer.close()
    error_buffer.close()
    
    # Construct response
    response = {"output": output, "errors": errors}
    
    if plot_data:
        response["plot"] = plot_data
        
    return jsonify(response)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)