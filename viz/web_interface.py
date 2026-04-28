import os

from flask import Flask, abort, render_template_string, request, send_file

app = Flask(__name__)

IMAGE_DIR = os.environ.get(
    "VIZFOLD_IMAGE_DIR",
    "./outputs/attention_images_6KWC_demo_tri_18",
)
PROT = os.environ.get("VIZFOLD_PROT", "6KWC")
TRI_RESIDUE_IDX = int(os.environ.get("VIZFOLD_TRI_IDX", "18"))
NUM_LAYERS = int(os.environ.get("VIZFOLD_NUM_LAYERS", "48"))

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>VizFold - Attention Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        select, button { padding: 8px 12px; margin: 8px; font-size: 14px; }
        button { background-color: #2c3e50; color: white; border: none; cursor: pointer; border-radius: 4px; }
        button:hover { background-color: #34495e; }
        img { max-width: 100%; margin-top: 20px; border: 1px solid #ddd; border-radius: 4px; }
        .controls { background: #f8f9fa; padding: 20px; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>VizFold — OpenFold Attention Visualizer ({{ prot }})</h1>
    <div class="controls">
        <form method="GET" action="/">
            <label><b>Attention Type:</b></label>
            <select name="attn_type">
                <option value="msa_row" {% if attn_type == 'msa_row' %}selected{% endif %}>MSA Row Attention</option>
                <option value="triangle_start" {% if attn_type == 'triangle_start' %}selected{% endif %}>Triangle Start Attention</option>
            </select>

            <label><b>Layer:</b></label>
            <select name="layer">
                {% for l in layers %}
                <option value="{{ l }}" {% if l == layer %}selected{% endif %}>Layer {{ l }}</option>
                {% endfor %}
            </select>

            <button type="submit">Visualize</button>
        </form>
    </div>

    {% if image_path %}
        <h3>{{ attn_type }} — Layer {{ layer }}</h3>
        <img src="/image?path={{ image_path }}" alt="Attention Heatmap">
    {% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    attn_type = request.args.get("attn_type", "msa_row")
    if attn_type not in ("msa_row", "triangle_start"):
        attn_type = "msa_row"
    try:
        layer = max(0, min(int(request.args.get("layer", 0)), NUM_LAYERS - 1))
    except (ValueError, TypeError):
        layer = 0
    layers = list(range(NUM_LAYERS))

    if attn_type == "msa_row":
        fname = f"heatmap_msa_row_layer{layer}.png"
    else:
        fname = f"heatmap_triangle_start_layer{layer}.png"

    image_path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(image_path):
        image_path = None

    return render_template_string(
        HTML,
        attn_type=attn_type,
        layer=layer,
        layers=layers,
        image_path=image_path,
        prot=PROT,
    )

@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    if not path:
        abort(400)
    # Resolve both paths to prevent directory traversal attacks.
    abs_image_dir = os.path.realpath(IMAGE_DIR)
    abs_path = os.path.realpath(path)
    if not abs_path.startswith(abs_image_dir + os.sep):
        abort(403)
    if not os.path.isfile(abs_path):
        abort(404)
    return send_file(abs_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)