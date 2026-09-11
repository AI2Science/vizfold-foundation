import tempfile
from pathlib import Path

# Sample Flask attention file content
SAMPLE_FLASK_FORMAT = """Layer 24, Head 0
10 15 0.8543
12 18 0.7234
15 20 0.6891
20 25 0.5432

Layer 24, Head 1
10 16 0.9123
12 19 0.8456
14 21 0.7890
18 22 0.6543
"""

def test_parsing():
    # Create temporary file with Flask format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_FLASK_FORMAT)
        temp_path = f.name
    
    # Parse it
    data = parse_flask_attention(temp_path)
    
    # Verify results
    print(" Testing attention file parsing...\n")
    
    print(f"Total entries parsed: {len(data)}")
    print(f"Expected: 8 (4 per head × 2 heads)")
    
    # Check head separation
    head_0 = [(i, j, s) for i, j, h, s in data if h == 0]
    head_1 = [(i, j, s) for i, j, h, s in data if h == 1]
    
    print(f"\nHead 0 entries: {len(head_0)}")
    print(f"Head 1 entries: {len(head_1)}")
    
    # Show some samples
    print("\nSample parsed data (head 0):")
    for i, j, s in head_0[:3]:
        print(f"  Residue {i} → {j}: {s:.4f}")
    
    print("\nSample parsed data (head 1):")
    for i, j, s in head_1[:3]:
        print(f"  Residue {i} → {j}: {s:.4f}")
    
    # Cleanup
    Path(temp_path).unlink()
    
    if len(data) == 8 and len(head_0) == 4 and len(head_1) == 4:
        print("\nParsing test PASSED!")
        return True
    else:
        print("\nParsing test FAILED!")
        return False


def parse_flask_attention(filepath):
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


def test_matplotlib():
    print("\n Testing matplotlib...\n")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create simple test plot
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        
        # Try to save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
            fig.savefig(tmp.name)
            print(f"Test plot saved to: {tmp.name}")
        
        plt.close(fig)
        
        print(" Matplotlib test PASSED!")
        return True
        
    except Exception as e:
        print(f"Matplotlib test FAILED: {e}")
        return False


def test_gradio_imports():
    print("\n Testing package imports...\n")
    
    packages = {
        'gradio': 'Gradio',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy'
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f" {name}: OK")
        except ImportError:
            print(f" {name}: NOT INSTALLED")
            all_good = False
    
    if all_good:
        print("\n All packages installed!")
    else:
        print("\n Some packages missing. Run: pip install -r requirements.txt")
    
    return all_good


if __name__ == "__main__":
    print("=" * 60)
    print("Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Package imports
    if test_gradio_imports():
        tests_passed += 1
    
    # Test 2: Attention parsing
    if test_parsing():
        tests_passed += 1
    
    # Test 3: Matplotlib
    if test_matplotlib():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n All tests passed! Ready to run app.py")
    else:
        print("\n  Some tests failed. Fix issues before running app.py")