import pytest
import ants
import syntx

def test_benchmark_data_keys():
    for key in ['r16_r64', '2d', 'c', 'ellipse', 'mbhard', '3d']:
        data = syntx.benchmark_data(key)
        assert 'fixed' in data
        assert 'moving' in data
        assert 'fixed_label' in data
        assert 'moving_label' in data
        assert isinstance(data['fixed'], ants.ANTsImage)
        assert isinstance(data['moving'], ants.ANTsImage)
        assert isinstance(data['fixed_label'], ants.ANTsImage)
        assert isinstance(data['moving_label'], ants.ANTsImage)

def test_benchmark_data_invalid_key():
    with pytest.raises(ValueError, match="Unknown benchmark dataset key"):
        syntx.benchmark_data('non_existent_key')
