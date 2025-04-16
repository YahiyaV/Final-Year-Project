# Final-Year-Project
DEEPTFBIND: A HYBRID MODEL FOR TRANSCRIPTION FACTOR BINDING SITE PREDICTION

This is a web-based tool for analyzing DNA sequences to predict transcription factor (TF) binding affinities, p53 specificity, and assess potential cancer risk. The application uses machine learning models to process uploaded sequence files, generates visual representations of binding spots, and provides a risk assessment based on the relative specificity of General TF and p53 binding.

## Features
- Upload a TXT file containing DNA sequences (optionally with labels).
- Predict General TF and p53 binding affinities and specificities.
- Generate sequence logos to visualize TF binding spots using Logomaker.
- Display specificity scores in a grouped bar chart using Plotly.
- Assess cancer risk based on the dominance of p53 or TF specificity without predefined thresholds.
- Integrate with Mol* Viewer for 3D molecular visualization.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dna-sequence-analysis
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your trained models in the specified paths:
   - General TF model: `models/NewModel(26).h5`
   - TP53 model: `D:/Datasets/DNA dataset/TP53/tp53_model.h5` (update the path as needed)
5. Ensure the `static` and `uploads` directories exist or are created automatically on first run.

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`.
3. Upload a TXT file with DNA sequences (e.g., one sequence per line, optionally followed by a space and label).
   - Example format:
     ```
     ATGCATGC 1
     CGTAGCTA 0
     ```
4. View the results, including:
   - A table with sequence details, affinities, specificities, and cancer risk levels.
   - TF binding spot logos in a grid layout.
   - A grouped bar chart comparing General TF and p53 specificity scores.
5. Click "Open Mol* Viewer" to explore 3D molecular structures (opens in a new tab).

## Cancer Risk Assessment Logic
The tool assesses cancer risk based on the direct comparison of p53 and General TF specificity scores:
- **p53 specificity > TF specificity**: "Potential" (Green) — Indicates strong p53 control loss, a potential cancer risk.
- **TF specificity > p53 specificity**: "Neutral" (Red) — Suggests the site isn’t reliant on p53, neutral risk.
- **Both specificities = 0**: "Low" (Red) — Likely irrelevant due to no significant binding.
- **Both specificities equal and > 0**: "Depends" (Yellow) — Risk depends on context, possibly redundant regulation.
- **Other cases**: "Neutral" (Red) — Default for moderate or unclear dominance.

## File Structure
- `app.py`: Main Flask application logic.
- `index.html`: Frontend template with JavaScript and CSS.
- `styles.css`: Custom styles for the web interface.
- `static/`: Directory for static files (e.g., logos).
- `uploads/`: Directory for uploaded files.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Dependencies
See `requirements.txt` for the full list. Key libraries include:
- Flask for web framework
- TensorFlow for model predictions
- Matplotlib and Logomaker for sequence logos
- Pandas and NumPy for data handling
- Plotly for bar chart visualization (via CDN)

## Contributing
Feel free to fork this repository, submit issues, or create pull requests for improvements. Suggestions for enhancing the risk assessment logic or adding new visualizations are welcome!

## License
[Add your license here, e.g., MIT License. If none, specify accordingly.]

## Acknowledgments
- Built with assistance from xAI's Grok.
- Utilizes open-source libraries and tools like Logomaker and Plotly.
