import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import os
import zipfile
import tempfile
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

class AttentionExplorer:
    
    def __init__(self):
        self.proteins: Dict[str, dict] = {}
        
    def load_from_zip(self, zip_path: str, protein_name: str) -> Tuple[str, Optional[List[str]]]:
        if not zip_path or not protein_name:
            return "Please provide both file and name", None
            
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(tmpdir)
                
                # Find attention files
                attention_files = list(Path(tmpdir).rglob("*attn_layer*.txt"))
                
                if not attention_files:
                    return f"No attention files found in {protein_name}", None
                
                # Parse all attention files
                attention_data = {
                    'msa_row': {},
                    'triangle_start': {}
                }
                
                for attn_file in attention_files:
                    layer = self._extract_layer_number(attn_file.name)
                    attn_type = self._get_attention_type(attn_file.name)
                    
                    if attn_type and layer is not None:
                        parsed_data = self._parse_attention_file(attn_file)
                        if parsed_data:
                            attention_data[attn_type][layer] = parsed_data
                
                # Calculate sequence length
                seq_len = self._get_sequence_length(attention_data)
                
                # Store protein data
                self.proteins[protein_name] = {
                    'attention': attention_data,
                    'seq_len': seq_len,
                    'n_msa_layers': len(attention_data['msa_row']),
                    'n_tri_layers': len(attention_data['triangle_start'])
                }
                
                status = (f"Loaded {protein_name}\n"
                         f"   • {seq_len} residues\n"
                         f"   • {len(attention_data['msa_row'])} MSA layers\n"
                         f"   • {len(attention_data['triangle_start'])} Triangle layers")
                
                return status, list(self.proteins.keys())
                
        except Exception as e:
            return f"Error loading {protein_name}: {str(e)}", None
    
    def _extract_layer_number(self, filename: str) -> Optional[int]:
        """Extract layer number from filename like 'msa_row_attn_layer24.txt'"""
        match = re.search(r'layer(\d+)', filename)
        return int(match.group(1)) if match else None
    
    def _get_attention_type(self, filename: str) -> Optional[str]:
        """Determine attention type from filename"""
        if 'msa_row' in filename:
            return 'msa_row'
        elif 'triangle_start' in filename:
            return 'triangle_start'
        return None
    
    def _parse_attention_file(self, filepath: Path) -> List[Tuple[int, int, int, float]]:
        data = []
        current_head = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse header: "Layer X, Head Y"
                if line.lower().startswith('layer'):
                    parts = line.replace(',', '').split()
                    try:
                        current_head = int(parts[-1])
                    except (ValueError, IndexError):
                        current_head = 0
                    continue
                
                # Parse data line: "res_i res_j weight"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        res_i = int(parts[0])
                        res_j = int(parts[1])
                        score = float(parts[2])
                        data.append((res_i, res_j, current_head, score))
                    except ValueError:
                        continue
        
        return data
    
    def _get_sequence_length(self, attention_data: dict) -> int:
        max_res = 0
        for attn_type in attention_data.values():
            for layer_data in attn_type.values():
                if layer_data:
                    for res_i, res_j, _, _ in layer_data:
                        max_res = max(max_res, res_i, res_j)
        return max_res + 1
    
    def compare_proteins(self, protein_names: List[str], attn_type: str, 
                        layer: int, head: int, top_k: int) -> Optional[plt.Figure]:
        if not protein_names:
            return None
        
        # Filter to only loaded proteins
        valid_proteins = [p for p in protein_names if p in self.proteins]
        if not valid_proteins:
            return None
        
        n = len(valid_proteins)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        
        for idx, prot_name in enumerate(valid_proteins):
            ax = axes[idx]
            prot_data = self.proteins[prot_name]
            
            # Get attention data for this layer
            layer_data = prot_data['attention'][attn_type].get(layer, [])
            
            if not layer_data:
                ax.text(0.5, 0.5, f'{prot_name}\nNo data for layer {layer}',
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            # Filter by head and get top-k
            head_data = [(i, j, s) for i, j, h, s in layer_data if h == head]
            head_data.sort(key=lambda x: x[2], reverse=True)
            top_connections = head_data[:top_k]
            
            # Draw arc diagram
            self._draw_arc_diagram(ax, top_connections, prot_data['seq_len'], prot_name)
        
        title = f'{attn_type.replace("_", " ").title()} - Layer {layer}, Head {head} (Top-{top_k})'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig
    
    def _draw_arc_diagram(self, ax, connections: List[Tuple[int, int, float]], 
                         seq_len: int, title: str):
        if not connections:
            ax.text(0.5, 0.5, 'No connections', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return
        
        positions = np.arange(seq_len)
        
        # Normalize weights
        weights = [s for _, _, s in connections]
        wmin, wmax = min(weights), max(weights)
        norm = lambda w: (w - wmin) / (wmax - wmin) if wmax > wmin else 0.5
        
        # Draw arcs
        for res_i, res_j, score in connections:
            if res_i > res_j:
                res_i, res_j = res_j, res_i
            
            # Arc parameters
            height = (res_j - res_i) / 4
            theta = np.linspace(0, np.pi, 50)
            x = res_i + (res_j - res_i) * (1 - np.cos(theta)) / 2
            y = height * np.sin(theta)
            
            # Style based on weight
            nw = norm(score)
            blue_intensity = int((0.5 + 0.5 * nw) * 255)
            color = (0, 0, blue_intensity / 255)
            alpha = 0.6 + 0.4 * nw
            linewidth = 0.5 + 2.5 * nw
            
            ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)
        
        # Draw baseline
        ax.plot(positions, np.zeros_like(positions), 'k-', linewidth=2)
        ax.scatter(positions, np.zeros_like(positions), c='red', s=15, zorder=3)
        
        # Styling
        ax.set_xlim(-5, seq_len + 5)
        ax.set_ylim(-5, seq_len / 3)
        ax.set_xlabel('Residue Position', fontsize=10)
        ax.set_title(f'{title}\n({seq_len} residues)', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
    
    def compute_statistics(self, protein_names: List[str], 
                          attn_type: str) -> Optional[plt.Figure]:
        valid_proteins = [p for p in protein_names if p in self.proteins]
        if not valid_proteins:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for prot_name in valid_proteins:
            prot_data = self.proteins[prot_name]
            attn_layers = prot_data['attention'].get(attn_type, {})
            
            if not attn_layers:
                continue
            
            layers = sorted(attn_layers.keys())
            
            # Metric 1: Average strength
            avg_strengths = []
            for layer in layers:
                scores = [s for _, _, _, s in attn_layers[layer]]
                avg_strengths.append(np.mean(scores) if scores else 0)
            
            axes[0, 0].plot(layers, avg_strengths, marker='o', label=prot_name, linewidth=2)
            
            # Metric 2: Connection count
            n_connections = [len(attn_layers[l]) for l in layers]
            axes[0, 1].plot(layers, n_connections, marker='s', label=prot_name, linewidth=2)
            
            # Metric 3: Max score
            max_scores = []
            for layer in layers:
                scores = [s for _, _, _, s in attn_layers[layer]]
                max_scores.append(max(scores) if scores else 0)
            
            axes[1, 0].plot(layers, max_scores, marker='^', label=prot_name, linewidth=2)
            
            # Metric 4: Diversity (std)
            diversities = []
            for layer in layers:
                scores = [s for _, _, _, s in attn_layers[layer]]
                diversities.append(np.std(scores) if len(scores) > 1 else 0)
            
            axes[1, 1].plot(layers, diversities, marker='d', label=prot_name, linewidth=2)
        
        # Formatting
        titles = [
            'Average Attention Strength',
            'Number of Connections',
            'Maximum Attention Score',
            'Attention Diversity (Std Dev)'
        ]
        
        for ax, title in zip(axes.flat, titles):
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[0, 0].set_ylabel('Avg Score')
        axes[0, 1].set_ylabel('Count')
        axes[1, 0].set_ylabel('Max Score')
        axes[1, 1].set_ylabel('Std Dev')
        
        fig.suptitle(f'{attn_type.replace("_", " ").title()} - Layer Statistics',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def get_available_layers(self, protein_names: List[str], 
                            attn_type: str) -> List[int]:
        if not protein_names:
            return list(range(48))
        
        valid_proteins = [p for p in protein_names if p in self.proteins]
        if not valid_proteins:
            return list(range(48))
        
        # Get intersection of layers
        layer_sets = []
        for prot_name in valid_proteins:
            layers = set(self.proteins[prot_name]['attention'][attn_type].keys())
            layer_sets.append(layers)
        
        if layer_sets:
            common_layers = sorted(set.intersection(*layer_sets))
            return common_layers if common_layers else list(range(48))
        
        return list(range(48))


# Initialize explorer
explorer = AttentionExplorer()


# Gradio callback functions
def load_protein_callback(zip_file, protein_name):
    if zip_file is None:
        return "Please upload a zip file", gr.update()
    
    if not protein_name or not protein_name.strip():
        return "Please provide a protein name", gr.update()
    
    status, protein_list = explorer.load_from_zip(zip_file.name, protein_name.strip())
    
    if protein_list:
        return status, gr.update(choices=protein_list, value=protein_list)
    else:
        return status, gr.update()


def compare_callback(protein_names, attn_type, layer, head, top_k):
    if not protein_names:
        return None
    
    fig = explorer.compare_proteins(protein_names, attn_type, layer, head, top_k)
    return fig


def statistics_callback(protein_names, attn_type):
    if not protein_names:
        return None
    
    fig = explorer.compute_statistics(protein_names, attn_type)
    return fig


def update_layer_range(protein_names, attn_type):
    layers = explorer.get_available_layers(protein_names, attn_type)
    if layers:
        return gr.update(minimum=min(layers), maximum=max(layers), value=min(layers))
    return gr.update()


# Build Gradio Interface
with gr.Blocks(title="OpenFold Attention Explorer") as app:
    
    gr.Markdown("""
    # OpenFold Attention Explorer
    """)
    
    with gr.Row():
        # Left column: Controls
        with gr.Column(scale=1):
            with gr.Group():
                zip_input = gr.File(
                    label="Flask Output (zip)",
                    file_types=[".zip"],
                    type="filepath"
                )
                
                protein_name_input = gr.Textbox(
                    label="Protein Name",
                    placeholder="e.g., wild_type, mutant_A, 6KWC",
                    info="Descriptive name for this protein"
                )
                
                load_btn = gr.Button("📂 Load Protein", variant="primary", size="lg")
                
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4,
                    placeholder="Status messages will appear here..."
                )
            
            gr.Markdown("---")
            gr.Markdown("### Analysis Settings")
            
            loaded_proteins = gr.CheckboxGroup(
                choices=[],
                label="Select Proteins to Compare",
                info="Choose 1-4 proteins for comparison"
            )
            
            attn_type_select = gr.Dropdown(
                choices=["msa_row", "triangle_start"],
                value="msa_row",
                label="Attention Type",
                info="MSA Row: sequence attention | Triangle Start: pairwise attention"
            )
            
            with gr.Row():
                layer_slider = gr.Slider(
                    minimum=0,
                    maximum=47,
                    step=1,
                    value=24,
                    label="Layer",
                    info="OpenFold has 48 layers (0-47)"
                )
                
                head_slider = gr.Slider(
                    minimum=0,
                    maximum=7,
                    step=1,
                    value=0,
                    label="Head",
                    info="8 attention heads per layer"
                )
            
            top_k_slider = gr.Slider(
                minimum=10,
                maximum=100,
                step=10,
                value=50,
                label="Top-K Connections",
                info="Number of strongest connections to display"
            )
            
            with gr.Row():
                compare_btn = gr.Button("🔍 Compare Patterns", variant="primary")
                stats_btn = gr.Button("📊 Show Statistics", variant="secondary")
        
        # Right column: Visualizations
        with gr.Column(scale=2):
            gr.Markdown("### Visualizations")
            
            with gr.Tab("Side-by-Side Comparison"):
                comparison_plot = gr.Plot(label="Attention Patterns")
            
            with gr.Tab("Statistical Analysis"):
                statistics_plot = gr.Plot(label="Layer Statistics")
    
    # Event handlers
    load_btn.click(
        fn=load_protein_callback,
        inputs=[zip_input, protein_name_input],
        outputs=[load_status, loaded_proteins]
    )
    
    # Update layer range when proteins or attention type changes
    for component in [loaded_proteins, attn_type_select]:
        component.change(
            fn=update_layer_range,
            inputs=[loaded_proteins, attn_type_select],
            outputs=[layer_slider]
        )
    
    compare_btn.click(
        fn=compare_callback,
        inputs=[loaded_proteins, attn_type_select, layer_slider, head_slider, top_k_slider],
        outputs=[comparison_plot]
    )
    
    stats_btn.click(
        fn=statistics_callback,
        inputs=[loaded_proteins, attn_type_select],
        outputs=[statistics_plot]
    )

# Launch application
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )