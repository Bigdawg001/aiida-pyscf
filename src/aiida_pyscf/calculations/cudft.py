# -*- coding: utf-8 -*-
"""``CalcJob`` plugin for CuDFT."""
from __future__ import annotations

from jinja2 import Environment, PackageLoader

from .base import PyscfCalculation

__all__ = ('CudftCalculation',)


class CudftCalculation(PyscfCalculation):
    """``CalcJob`` plugin for CuDFT."""

    def render_script(self) -> str:
        """Return the rendered input script.

        :returns: The input script template rendered with the parameters provided by ``get_parameters``.
        """
        environment = Environment(loader=PackageLoader('aiida_pyscf.calculations.base'),)
        parameters = self.get_parameters()

        return environment.get_template('script_cudft.py.j2').render(
            structure=parameters.get('structure', {}),
            cuda=parameters.get('cuda', {}),
            mean_field=parameters.get('mean_field', {}),
            optimizer=parameters.get('optimizer', None),
            results=parameters.get('results', {}),
        )
