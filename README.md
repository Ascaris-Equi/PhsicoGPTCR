# PhsicoGPTCR
A GPT-based Deep Learning Model for TCR-T Design and Cross-Reactivity Prediction
🚀 Introduction

This repository provides a GPT-based deep learning model capable of predicting T-cell receptor (TCR) Complementarity Determining Region 3 (CDR3) sequences from given peptide-MHC class I complexes. The model integrates physicochemical properties, positional information, amino acid embeddings, and cross-attention mechanisms, achieving accurate and biologically relevant predictions. It is particularly useful for designing TCR-T cell therapies and assessing T-cell cross-reactivity between antigens.
🌟 Key Features

    Advanced GPT Architecture: Incorporates physicochemical properties, positional embeddings, amino acid fusion encoders, and cross-attention mechanisms.
    High Biological Relevance: Predictions validated by similarity and edit-distance metrics against known functional TCR sequences.
    Broad Immunological Applications: Suitable for TCR-T therapy design and antigen cross-reactivity prediction in cancer, viral infections, and autoimmune diseases.
    Ease of Use: Simple installation and execution pipeline suitable for researchers and developers.

🧬 Model Architecture and Data

    Architecture: GPT-based model integrating:
        Physicochemical property embeddings
        Positional embeddings
        Amino acid fusion encoder
        Cross-attention layers
    Training Data: Curated TCR-peptide-MHC binding data from VDJdb, a well-established database of antigen-specific TCR sequences.

📥 Installation Guide
1. Requirements

    Python (>=3.7 recommended)
    PyTorch
    NumPy
    Pandas
    tqdm

2. Install Dependencies
bash

pip install torch numpy pandas tqdm

3. Run the Model

Execute the prediction script as follows:
bash

python generate_candidates.py

⚙️ Usage Example

Prepare your input data as a CSV file (input.csv) following the format below:
Peptide	MHC (Class I only)
AAAKLY	HLA-A*02:01
GILGFVFTL	HLA-A*02:01
...	...

Run the script to generate TCR CDR3 sequences:
bash

python generate_candidates.py --input input.csv --output predictions.csv

The output file (predictions.csv) will present the results as follows:
Peptide	MHC	Predicted TCR CDR3 Sequence
AAAKLY	HLA-A*02:01	CASSIRSSYEQYF
GILGFVFTL	HLA-A*02:01	CASSPGQETQYF
...	...	...
📊 Model Evaluation & Performance

The model predictions have been rigorously evaluated by comparing them to known functional TCR sequences from biological data (VDJdb). Evaluation metrics include:

    Sequence similarity scores
    Edit-distance metrics
    Biological validity and relevance checks against known TCR repertoires.

The model demonstrates strong predictive performance, reliably generating biologically plausible TCR sequences.
📝 Applications & Limitations
Recommended Use Cases:

    TCR-T Cell Therapy Design: Generate candidate TCR sequences for therapeutic development.
    Cross-Reactivity Prediction: Assess potential cross-reactivity among peptide antigens in cancer, viral infections, and autoimmune diseases.

Known Limitations:

    Currently limited strictly to MHC Class I molecules (e.g., HLA-A, HLA-B, HLA-C).
    Input peptide-MHC sequences must not exceed 55 amino acids, and the predicted CDR3 length is limited to a maximum of 28 amino acids.

🔮 Future Development Plans

    Expansion of the model to include MHC Class II molecules.
    Integration into an online interactive platform or API for broader community access.
    Continuous improvement through increased training data diversity and advanced embedding techniques.

🤝 Contributing

We welcome contributions! Please feel free to open an issue or submit a pull request to help improve this project. But Authors reserve all rights.
