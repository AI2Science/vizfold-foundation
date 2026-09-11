"""
Unit tests for dimensionality reduction utilities.

Tests the new t-SNE, PCA, UMAP, and autoencoder functionality
added to extend Jayanth's intermediate representation extraction.

Author: Shreyas
"""

import unittest
import numpy as np
import torch
import os
import tempfile
import shutil

# Import functions to test
from visualize_dimensionality_reduction import (
    prepare_representations_for_reduction,
    apply_pca,
    apply_tsne,
    apply_umap,
    train_autoencoder,
    RepresentationAutoencoder,
    plot_2d_embedding,
    plot_3d_embedding,
    compare_reduction_methods,
    SKLEARN_AVAILABLE,
    UMAP_AVAILABLE
)


class TestRepresentationPreparation(unittest.TestCase):
    """Test data preparation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.msa_tensor_3d = torch.randn(2, 50, 128)  # (batch, seq_len, channels)
        self.pair_tensor_4d = torch.randn(2, 50, 50, 64)  # (batch, seq1, seq2, channels)
    
    def test_prepare_msa_residue_mode(self):
        """Test MSA preparation in residue mode."""
        data = prepare_representations_for_reduction(self.msa_tensor_3d, flatten_mode='residue')
        self.assertEqual(data.shape, (100, 128))  # 2*50 residues, 128 features
        self.assertIsInstance(data, np.ndarray)
    
    def test_prepare_msa_global_mode(self):
        """Test MSA preparation in global mode."""
        data = prepare_representations_for_reduction(self.msa_tensor_3d, flatten_mode='global')
        self.assertEqual(data.shape, (2, 128))  # 2 samples, 128 features
    
    def test_prepare_pair_pairwise_mode(self):
        """Test pair preparation in pairwise mode."""
        data = prepare_representations_for_reduction(self.pair_tensor_4d, flatten_mode='pairwise')
        self.assertEqual(data.shape, (5000, 64))  # 2*50*50 pairs, 64 features
    
    def test_prepare_pair_global_mode(self):
        """Test pair preparation in global mode."""
        data = prepare_representations_for_reduction(self.pair_tensor_4d, flatten_mode='global')
        self.assertEqual(data.shape, (2, 64))  # 2 samples, 64 features
    
    def test_numpy_input(self):
        """Test that numpy arrays are handled correctly."""
        numpy_data = self.msa_tensor_3d.numpy()
        data = prepare_representations_for_reduction(numpy_data, flatten_mode='residue')
        self.assertEqual(data.shape, (100, 128))
    
    def test_invalid_flatten_mode(self):
        """Test error handling for invalid flatten mode."""
        with self.assertRaises(ValueError):
            prepare_representations_for_reduction(self.msa_tensor_3d, flatten_mode='invalid')


class TestPCA(unittest.TestCase):
    """Test PCA functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.randn(100, 50)  # 100 samples, 50 features
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_pca_2d(self):
        """Test PCA reduction to 2D."""
        reduced, pca_model = apply_pca(self.data, n_components=2)
        self.assertEqual(reduced.shape, (100, 2))
        self.assertEqual(len(pca_model.explained_variance_ratio_), 2)
        self.assertTrue(0 < pca_model.explained_variance_ratio_.sum() <= 1.0)
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_pca_3d(self):
        """Test PCA reduction to 3D."""
        reduced, pca_model = apply_pca(self.data, n_components=3)
        self.assertEqual(reduced.shape, (100, 3))
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_pca_variance_ordering(self):
        """Test that PCA variance ratios are in descending order."""
        reduced, pca_model = apply_pca(self.data, n_components=5)
        variances = pca_model.explained_variance_ratio_
        self.assertTrue(all(variances[i] >= variances[i+1] for i in range(len(variances)-1)))


class TestTSNE(unittest.TestCase):
    """Test t-SNE functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.randn(50, 30)  # Smaller dataset for t-SNE speed
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_tsne_2d(self):
        """Test t-SNE reduction to 2D."""
        reduced = apply_tsne(self.data, n_components=2, n_iter=250)  # Fewer iterations for speed
        self.assertEqual(reduced.shape, (50, 2))
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_tsne_perplexity_adjustment(self):
        """Test automatic perplexity adjustment for small datasets."""
        small_data = np.random.randn(20, 30)
        # Should automatically adjust perplexity without error
        reduced = apply_tsne(small_data, n_components=2, perplexity=30, n_iter=250)
        self.assertEqual(reduced.shape, (20, 2))


class TestUMAP(unittest.TestCase):
    """Test UMAP functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.randn(100, 40)
    
    @unittest.skipIf(not UMAP_AVAILABLE, "UMAP not available")
    def test_umap_2d(self):
        """Test UMAP reduction to 2D."""
        reduced = apply_umap(self.data, n_components=2)
        self.assertEqual(reduced.shape, (100, 2))
    
    @unittest.skipIf(not UMAP_AVAILABLE, "UMAP not available")
    def test_umap_3d(self):
        """Test UMAP reduction to 3D."""
        reduced = apply_umap(self.data, n_components=3)
        self.assertEqual(reduced.shape, (100, 3))
    
    @unittest.skipIf(not UMAP_AVAILABLE, "UMAP not available")
    def test_umap_neighbors_adjustment(self):
        """Test automatic n_neighbors adjustment for small datasets."""
        small_data = np.random.randn(10, 40)
        reduced = apply_umap(small_data, n_components=2, n_neighbors=15)
        self.assertEqual(reduced.shape, (10, 2))


class TestAutoencoder(unittest.TestCase):
    """Test autoencoder functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        self.data = np.random.randn(100, 50)
    
    def test_autoencoder_architecture(self):
        """Test autoencoder model creation."""
        model = RepresentationAutoencoder(input_dim=50, latent_dim=2, 
                                         hidden_dims=[25, 10])
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(10, 50)
        reconstructed, latent = model(x)
        self.assertEqual(reconstructed.shape, (10, 50))
        self.assertEqual(latent.shape, (10, 2))
    
    def test_autoencoder_training(self):
        """Test autoencoder training."""
        model, latent_reps = train_autoencoder(
            self.data, 
            latent_dim=2,
            n_epochs=5,  # Few epochs for speed
            batch_size=32,
            device='cpu',
            verbose=False
        )
        
        self.assertEqual(latent_reps.shape, (100, 2))
        self.assertIsInstance(model, RepresentationAutoencoder)
    
    def test_autoencoder_reconstruction(self):
        """Test that autoencoder can reconstruct input."""
        model, latent_reps = train_autoencoder(
            self.data,
            latent_dim=10,  # Higher dimension for better reconstruction
            n_epochs=20,
            batch_size=32,
            device='cpu',
            verbose=False
        )
        
        # Test reconstruction
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(self.data[:10])
            reconstructed, _ = model(x)
            reconstructed = reconstructed.numpy()
        
        # Reconstruction should be reasonably close (not perfect due to bottleneck)
        mse = np.mean((self.data[:10] - reconstructed) ** 2)
        self.assertLess(mse, 5.0)  # Arbitrary threshold


class TestVisualizationFunctions(unittest.TestCase):
    """Test visualization functions."""
    
    def setUp(self):
        """Set up test data and temp directory."""
        np.random.seed(42)
        self.data_2d = np.random.randn(50, 2)
        self.data_3d = np.random.randn(50, 3)
        self.labels = np.random.randint(0, 5, 50)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plot_2d_embedding(self):
        """Test 2D embedding plot."""
        save_path = os.path.join(self.temp_dir, 'test_2d.png')
        # Should not raise an error
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        plot_2d_embedding(self.data_2d, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_2d_with_labels(self):
        """Test 2D embedding plot with labels."""
        save_path = os.path.join(self.temp_dir, 'test_2d_labels.png')
        import matplotlib
        matplotlib.use('Agg')
        plot_2d_embedding(self.data_2d, labels=self.labels, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_3d_embedding(self):
        """Test 3D embedding plot."""
        save_path = os.path.join(self.temp_dir, 'test_3d.png')
        import matplotlib
        matplotlib.use('Agg')
        plot_3d_embedding(self.data_3d, save_path=save_path)
        self.assertTrue(os.path.exists(save_path))


class TestComparisonFunctions(unittest.TestCase):
    """Test method comparison functions."""
    
    def setUp(self):
        """Set up test data and temp directory."""
        np.random.seed(42)
        self.data = np.random.randn(50, 30)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_compare_methods_pca_only(self):
        """Test method comparison with PCA only."""
        import matplotlib
        matplotlib.use('Agg')
        
        results = compare_reduction_methods(
            self.data,
            methods=['pca'],
            n_components=2,
            save_dir=self.temp_dir
        )
        
        self.assertIn('pca', results)
        self.assertEqual(results['pca'].shape, (50, 2))
    
    @unittest.skipIf(not (SKLEARN_AVAILABLE and UMAP_AVAILABLE), 
                     "scikit-learn or UMAP not available")
    def test_compare_multiple_methods(self):
        """Test comparison of multiple methods."""
        import matplotlib
        matplotlib.use('Agg')
        
        results = compare_reduction_methods(
            self.data,
            methods=['pca', 'umap'],
            n_components=2,
            save_dir=self.temp_dir
        )
        
        self.assertIn('pca', results)
        self.assertIn('umap', results)
        self.assertEqual(len(results), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
        
        # Simulate multi-layer representations
        self.layer_reps = {
            0: torch.randn(2, 50, 128),
            23: torch.randn(2, 50, 128),
            47: torch.randn(2, 50, 128)
        }
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_layer_progression_workflow(self):
        """Test complete layer progression workflow."""
        import matplotlib
        matplotlib.use('Agg')
        
        from visualize_dimensionality_reduction import visualize_layer_progression
        
        results = visualize_layer_progression(
            self.layer_reps,
            method='pca',
            n_components=2,
            save_dir=self.temp_dir,
            rep_type='msa',
            flatten_mode='residue'
        )
        
        # Check that all layers were processed
        for layer_idx in self.layer_reps.keys():
            self.assertIn(layer_idx, results)
            self.assertEqual(results[layer_idx].shape[1], 2)  # 2 components


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("Running dimensionality reduction unit tests...")
    print("="*70)
    run_tests()


