"""Thin lazy SAM interface; external execution is isolated."""
from .feasibility import Adapter


class SAMAdapter(Adapter):
    model = 'sam'

    def generate(self, source_image, target_age, output_path, *, source_age=None):
        return self._generate(source_image, source_age, target_age, output_path)
