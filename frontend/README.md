# VizFold Interactive Dashboard

This is the frontend interface for the VizFold protein visualization pipeline. It is built with React and Vite, designed to connect to the PyTorch/ESMFold backend to visualize 3D structures and internal neural network attention mechanisms in real-time.

## Features
* **Interactive 3D Viewer:** WebGL-accelerated protein rendering using 3Dmol.js, featuring confidence-based (pLDDT) coloring and clickable residue targeting.
* **Trace Explorer:** Dynamic heatmaps using Plotly.js to visualize ESM-2 Attention layers and Trunk Evolution (s_z) recycling iterations.
* **Synchronized Targeting:** Clicking physical amino acids on the 3D model draws spatial contact crosshairs directly onto the attention heatmaps.

## How to Run and Test

### 1. Start the Backend
The frontend requires the FastAPI bridge and test data to function.

1. Navigate to the root of the repository in your terminal.
2. Activate your Python environment (e.g., `openfold_env`).
3. Generate the required test tensors and PDB files:
   ```bash
   python3 run_test.py
   ```
4. Start the FastAPI server:
   ```bash
   # Default (uses test_output/ directory)
   python3 server.py

   # Custom directory
   python3 server.py --dir your_custom_directory_path
   ```

### 2. Start the Frontend
Open a separate terminal window and navigate into the `frontend` directory.

1. Install Node dependencies:
   ```bash
   npm install
   ```
2. Configure the environment variables:
   Create a `.env` file in the `frontend` directory and add the backend API URL:
   ```env
   VITE_API_URL=http://localhost:8000
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

### 3. Verification Steps
Open your browser to the local URL provided by Vite.

1. **Structure Viewer:** Ensure the 3D nanobody model loads correctly and the default "Color: Confidence" setting displays a blue core.
2. **Timeline & Traces:** Switch the Trace Explorer to "ESM-2 Attention" and scrub the slider. Verify the heatmap matrix updates without throwing 404 errors in the console.
3. **Crosshair Targeting:** Click any amino acid on the 3D model. Verify that the residue label pops up and a targeting crosshair instantly appears on the Plotly heatmap.