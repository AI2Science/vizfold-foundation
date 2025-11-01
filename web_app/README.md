# Protein Structure Prediction Web Interface

This is a Flask-based web interface for running OpenFold protein structure predictions and visualizing the results in 3D.
   ```

## Running the Application

1. Change your directory to the web_app directory:
   ```bash
   cd web_app
   ```

2. Start the Flask development server:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Enter a protein ID (e.g., 6KWC)
2. Enter a residue index to focus on (default: 18)
3. Optionally, add a description
4. Enter the protein sequence in single-letter amino acid code
5. Click "Predict Structure" to run the prediction

## Features

- 3D visualization of predicted protein structures
- Interactive controls for viewing different representations (cartoon, backbone, sidechains)
- Color schemes for better visualization (by chain, by residue)
- Responsive design that works on desktop and tablet devices

## Output

The application will create the following directories in the `web_tmp_dir` directory:

- `fasta_<protein_id>/`: Contains the input FASTA file
- `outputs/attention_files_<protein_id>_demo_tri_<residue_idx>/`: Contains attention map files
- `outputs/my_outputs_align_<protein_id>_demo_tri_<residue_idx>/`: Contains output files including the predicted PDB
- `outputs/attention_images_<protein_id>_demo_tri_<residue_idx>/`: Contains visualization images

## Notes

- The first run may take longer as it needs to download the required models
- For large proteins, the prediction may take several minutes to complete
- Ensure you have sufficient disk space for temporary files and outputs
