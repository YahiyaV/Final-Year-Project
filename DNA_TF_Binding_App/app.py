import os
import sys

# Add the project directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logomaker
from flask import Flask, request, render_template, jsonify, url_for
from tensorflow.keras.models import load_model
import time

# Initialize Flask app
app = Flask(__name__)

# Define upload and static folders
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the general TF model
MODEL_PATH = "models/NewModel(26).h5"
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("General TF model loaded successfully.")
except Exception as e:
    print(f"Error loading general TF model: {e}")
    model = None

# Load the TP53 model
TP53_MODEL_PATH = "D:/Datasets/DNA dataset/TP53/tp53_model.h5"
try:
    if not os.path.exists(TP53_MODEL_PATH):
        raise FileNotFoundError(f"TP53 model file not found at {TP53_MODEL_PATH}")
    tp53_model = load_model(TP53_MODEL_PATH)
    print("TP53 model loaded successfully.")
except Exception as e:
    print(f"Error loading TP53 model: {e}")
    tp53_model = None

# Valid DNA bases
VALID_BASES = {'A', 'T', 'C', 'G'}

def one_hot_encode(sequences, max_length):
    """One-hot encode DNA sequences for the general TF model."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded = []
    for seq in sequences:
        seq = seq[:max_length]  # Trim long sequences
        seq += "A" * (max_length - len(seq))  # Pad short sequences with 'A'
        encoded.append([mapping[nt] for nt in seq])
    return np.array(encoded, dtype=np.float32)

def one_hot_encode_tp53(seq):
    """One-hot encode a single DNA sequence for the TP53 model."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping.get(nuc, [0, 0, 0, 0]) for nuc in seq])

def generate_binding_spot_logo(sequence, binding_scores, index):
    """Generate a sequence logo highlighting TF binding spots for a single sequence."""
    try:
        seq_len = len(sequence)
        # Create a DataFrame for the sequence
        data = pd.DataFrame(0.0, index=range(seq_len), columns=['A', 'C', 'G', 'T'])
        for i, base in enumerate(sequence):
            if base in VALID_BASES:
                data.at[i, base] = binding_scores[i] if i < len(binding_scores) else 0.0

        # Dynamically adjust figure width based on sequence length
        width = max(5, seq_len * 0.4)
        plt.figure(figsize=(width, 5))
        logo = logomaker.Logo(data, color_scheme='classic')
        logo.ax.set_ylabel('Binding Score')
        logo.ax.set_xlabel('Position')
        logo.ax.set_title(f'TF Binding Spots - Sequence {index}')

        logo_filename = f"binding_spot_logo_{index}_{int(time.time())}.png"
        logo_path = os.path.join(STATIC_FOLDER, logo_filename)
        plt.savefig(logo_path, dpi=300, bbox_inches="tight")
        plt.close()

        return logo_filename
    except Exception as e:
        print(f"Error generating binding spot logo for sequence {index}: {e}")
        return None

def calculate_affinity_specificity(predictions):
    """Calculate specificity for each sequence using affinity comparison."""
    predictions = [float(pred) for pred in predictions]
    total_affinity = sum(predictions)
    if total_affinity == 0:
        return [0.0] * len(predictions)
    specificities = [round(pred / total_affinity, 3) for pred in predictions]
    return specificities

def calculate_tp53_affinities(sequences, model, max_length):
    """Calculate TP53 affinities for all sequences."""
    affinities = []
    for seq in sequences:
        try:
            # One-hot encode the sequence
            encoded_sequence = one_hot_encode_tp53(seq)

            # Pad or truncate the sequence to match the model's input shape
            if len(encoded_sequence) < max_length:
                padding_length = max_length - len(encoded_sequence)
                encoded_sequence = np.pad(encoded_sequence, ((0, padding_length), (0, 0)))
            elif len(encoded_sequence) > max_length:
                encoded_sequence = encoded_sequence[:max_length]

            # Reshape for CNN + LSTM input
            encoded_sequence = encoded_sequence.reshape(1, encoded_sequence.shape[0], 4)

            # Calculate P53 affinity (binding probability)
            p53_affinity = float(model.predict(encoded_sequence, verbose=0)[0][0])
            p53_affinity = round(p53_affinity, 4)
            affinities.append(p53_affinity)
        except Exception as e:
            print(f"Error calculating TP53 affinity for sequence {seq}: {e}")
            affinities.append(0.0)
    return affinities

def calculate_binding_scores(sequence, model, max_length):
    """Calculate per-position binding scores for a sequence using the general TF model."""
    try:
        seq_len = len(sequence)
        # One-hot encode the sequence for model input (may need padding for model)
        encoded_seq = one_hot_encode([sequence], max_length)
        # Simulate per-position contribution (mock-up for illustration)
        binding_scores = np.random.rand(seq_len) * 0.5
        return binding_scores
    except Exception as e:
        print(f"Error calculating binding scores: {e}")
        return np.zeros(len(sequence))

def assess_cancer_risk(p53_spec, tf_spec):
    """Assess cancer risk based on direct comparison of p53 and TF specificity scores."""
    # Direct comparison without thresholds
    if p53_spec > tf_spec:  # High p53 specificity + low/no TF
        return "Potential", "#FF4500"  # Green
    elif tf_spec > p53_spec:  # High TF specificity + low/no p53
        return "Neutral", "#0000FF"  # Red
    elif p53_spec == 0 and tf_spec == 0:  # Both low (assuming 0 indicates low)
        return "Low", "#32CD32"  # Red
    elif p53_spec > 0 and tf_spec > 0 and p53_spec == tf_spec:  # Both high (equal and non-zero)
        return "Depends", "#FFFF00"  # Yellow
    else:
        return "Neutral", "#0000FF"  # Default for other cases

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith('.txt'):
        return jsonify({"error": "Please upload a TXT file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    print(f"File saved: {file_path}")

    try:
        # Read the text file line by line
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Split each line into sequence and optional label
        test_sequences = []
        y_true = []
        has_labels = False
        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            test_sequences.append(parts[0])
            if len(parts) > 1:
                y_true.append(int(parts[1]))
                has_labels = True
        if not has_labels:
            y_true = None
    except Exception as e:
        print(f"Error reading file: {e}")
        return jsonify({"error": "Invalid file format. Expected a TXT file with DNA sequences (and optional labels)."}), 400

    if model is None:
        return jsonify({"error": "General TF model not loaded"}), 500
    if tp53_model is None:
        return jsonify({"error": "TP53 model not loaded"}), 500

    # General TF model predictions
    max_length = model.input_shape[1]
    clean_sequences = ["".join(base if base in VALID_BASES else 'A' for base in seq) for seq in test_sequences]
    adjusted_sequences = [seq[:max_length].ljust(max_length, 'A') for seq in clean_sequences]

    encoded_test_sequences = one_hot_encode(adjusted_sequences, max_length)
    predictions = model.predict(encoded_test_sequences).ravel()
    predicted_labels = (predictions >= 0.5).astype(int)

    # Convert predictions and predicted_labels to Python-native types
    predictions = [float(pred) for pred in predictions]
    predicted_labels = [int(label) for label in predicted_labels]

    # Calculate specificity using affinity comparison (general TF model)
    specificity_scores = calculate_affinity_specificity(predictions)

    # Calculate TP53 affinities and specificities
    tp53_max_length = tp53_model.input_shape[1]
    p53_affinities = calculate_tp53_affinities(clean_sequences, tp53_model, tp53_max_length)
    p53_specificities = calculate_affinity_specificity(p53_affinities)

    # Generate individual binding spot logos for each sequence
    logo_filenames = []
    for i, seq in enumerate(clean_sequences):
        binding_scores = calculate_binding_scores(seq, model, max_length)
        logo_filename = generate_binding_spot_logo(seq, binding_scores, i + 1)
        if logo_filename:
            logo_filenames.append(logo_filename)
        else:
            logo_filenames.append(None)

    # Assess cancer risk for each sequence
    results = []
    for i, (seq, gen_spec, p53_spec, label, logo_filename) in enumerate(zip(test_sequences, specificity_scores, p53_specificities, [y if y_true else 0 for y in y_true], logo_filenames)):
        risk_level, risk_color = assess_cancer_risk(p53_spec, gen_spec)
        results.append({
            "sequence": seq,
            "probability": gen_spec,
            "affinity": predicted_labels[i],
            "specificity": gen_spec,
            "p53_affinity": p53_affinities[i],
            "p53_specificity": p53_spec,
            "label": int(label) if has_labels else None,
            "index": i + 1,
            "logo_path": url_for('static', filename=logo_filename) if logo_filename else None,
            "risk_level": risk_level,
            "risk_color": risk_color
        })

    # Debug: Print specificity values to verify
    print("Specificity scores:", specificity_scores)
    print("P53 Specificity scores:", p53_specificities)

    return jsonify({
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "False") == "True")