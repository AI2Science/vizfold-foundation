# Protein Structure Prediction Web Interface

This is a Gradio-based tool for post-prediction analysis and comparison of attention patterns. This complements the Flask prediction pipeline (PR #12) by enabling batch analysis workflows.
   ```

## Running the Application
1. Change your directory to the web_app directory:
   ```bash
   cd gradio_app
   pip install -r requirements.txt
   ```

2. Start the Gradio server:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:7860
   ```

## Usage

1. Run a prediction using the main Flask web interface
2. Locate the attention files in web_tmp_dir/outputs/attention_files_<protein_id>_demo_tri_<residue_idx>/
3. Create a ZIP file containing all *attn_layer*.txt files from that directory
4. Upload the ZIP to this tool
5. Choose 1-4 proteins from the checkbox list for comparison
6. Configure analysis of Attention Type (MSA Row or Triangle Start), Layer (0-47), Head (0-7), Top-K connections to display
7. Visualize by clicking "Compare Patterns" for side-by-side arc diagrams or "Show Statistics" for layer-wise metric trends

## Features

- Load and compare up to 4 proteins side-by-side
- Arc diagrams showing top-k strongest connections between residues
- Track attention metrics across layers (average strength, connection count, max scores, diversity)
- Support for both MSA Row (sequence) and Triangle Start (pairwise) attention
- Adjust layer, head, and top-k parameters in real-time

## Notes

- The first run may take longer as it needs to download the required models
- For large proteins, the prediction may take several minutes to complete
- Ensure you have sufficient disk space for temporary files and outputs