# -*- coding: utf-8 -*-
"""Tests for the :mod:`aiida_pyscf.calculations.cudft` module."""
# pylint: disable=redefined-outer-name
from aiida.orm import Dict
import pytest

from aiida_pyscf.calculations.cudft import CudftCalculation


@pytest.fixture
def generate_inputs(aiida_local_code_factory, generate_structure):
    """Return a dictionary of inputs for the ``CudftCalculation`."""

    def factory(**kwargs):
        inputs = {
            'code': aiida_local_code_factory('pyscf.cudft', 'python'),
            'structure': generate_structure(),
            'metadata': {
                'options': {
                    'resources': {
                        'num_machines': 1
                    }
                }
            },
        }
        inputs.update(**kwargs)
        return inputs

    return factory


def test_default(generate_calc_job, generate_inputs, file_regression):
    """Test the plugin for default inputs."""
    inputs = generate_inputs()
    tmp_path, calc_info = generate_calc_job(CudftCalculation, inputs=inputs)

    assert sorted(calc_info.retrieve_list) == sorted([
        CudftCalculation.FILENAME_RESULTS,
        CudftCalculation.FILENAME_STDERR,
        CudftCalculation.FILENAME_STDOUT,
    ])

    content_input_file = (tmp_path / CudftCalculation.FILENAME_SCRIPT).read_text()
    file_regression.check(content_input_file, encoding='utf-8', extension='.pyr')


def test_parameters_cuda(generate_calc_job, generate_inputs, file_regression):
    """Test the plugin when specifying parameters for the ``cuda`` setup."""
    inputs = generate_inputs()
    inputs['parameters'] = Dict({'cuda': {'gpu_id_list': list(range(8))}})
    tmp_path, _ = generate_calc_job(CudftCalculation, inputs=inputs)

    content_input_file = (tmp_path / CudftCalculation.FILENAME_SCRIPT).read_text()
    file_regression.check(content_input_file, encoding='utf-8', extension='.pyr')
