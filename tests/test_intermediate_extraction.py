"""
Unit tests for intermediate representation extraction functions.

Run with: pytest tests/test_intermediate_extraction.py -v
"""

import pytest
import torch
import numpy as np
from visualize_intermediate_reps_utils import (
    extract_msa_representations,
    extract_pair_representations,
    aggregate_channels
)


class TestAggregation:
    """Test utility functions for channel aggregation."""
    
    def test_aggregate_channels_mean(self):
        """Test mean aggregation over channel dimension."""
        # Create dummy tensor: (10, 20, 64) representing (seq, res, channels)
        tensor = torch.randn(10, 20, 64)
        
        result = aggregate_channels(tensor, method='mean', axis=-1)
        
        assert result.shape == (10, 20), f"Expected shape (10, 20), got {result.shape}"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
    
    def test_aggregate_channels_norm(self):
        """Test L2 norm aggregation over channel dimension."""
        tensor = torch.randn(10, 20, 64)
        
        result = aggregate_channels(tensor, method='norm', axis=-1)
        
        assert result.shape == (10, 20), f"Expected shape (10, 20), got {result.shape}"
        assert np.all(result >= 0), "L2 norm should be non-negative"
    
    def test_aggregate_channels_max(self):
        """Test max aggregation over channel dimension."""
        tensor = torch.randn(10, 20, 64)
        
        result = aggregate_channels(tensor, method='max', axis=-1)
        
        assert result.shape == (10, 20), f"Expected shape (10, 20), got {result.shape}"


class TestMSAExtraction:
    """Test MSA representation extraction."""
    
    @pytest.mark.skip(reason="Not yet implemented")
    def test_extract_msa_shape(self):
        """Test that extracted MSA has correct shape."""
        # TODO: Implement this test after extraction function is complete
        pass
    
    @pytest.mark.skip(reason="Not yet implemented")
    def test_extract_msa_specific_layers(self):
        """Test extracting MSA from specific layers only."""
        # TODO: Implement this test
        pass


class TestPairExtraction:
    """Test Pair representation extraction."""
    
    @pytest.mark.skip(reason="Not yet implemented")
    def test_extract_pair_shape(self):
        """Test that extracted Pair tensor has correct shape."""
        # TODO: Implement this test after extraction function is complete
        pass
    
    @pytest.mark.skip(reason="Not yet implemented")
    def test_extract_pair_symmetry(self):
        """Test if Pair representation is symmetric (i,j) == (j,i)."""
        # TODO: Implement this test
        # Note: Pair representation might not be symmetric depending on model
        pass


class TestVisualization:
    """Test visualization functions."""
    
    @pytest.mark.skip(reason="Not yet implemented")
    def test_plot_msa_heatmap_creates_file(self):
        """Test that MSA heatmap visualization creates output file."""
        # TODO: Implement this test
        pass
    
    @pytest.mark.skip(reason="Not yet implemented")
    def test_plot_pair_heatmap_creates_file(self):
        """Test that Pair heatmap visualization creates output file."""
        # TODO: Implement this test
        pass


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

