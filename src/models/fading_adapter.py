"""Thin lazy FADING interface; no diffusion imports or downloads."""
from .feasibility import Adapter


class FADINGAdapter(Adapter):
    model = 'fading'

    def generate(self, source_image, source_age, target_age, output_path):
        return self._generate(source_image, source_age, target_age, output_path)
