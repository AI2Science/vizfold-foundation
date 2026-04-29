from vizfold.backends.esmfold.inference import ESMFoldRunner

print("Loading ESMFoldRunner...")
# Initialize the lead's class
runner = ESMFoldRunner(device="cpu") 

print("Running inference...")
# Run the forward pass on our test fasta
results = runner.run(
    fasta_path="test.fasta", 
    out_dir="test_output", 
    trace_mode="attention+activations"
)

print(f"\nSuccess! Output saved to: {results['out_dir']}")