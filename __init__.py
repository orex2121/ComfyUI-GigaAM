from .gigaam_node import NODE_CLASS_MAPPINGS as gigaam_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as gigaam_name_mappings
from .Orex_CuttingSubtitles import Orex_CuttingSubtitles
from .Orex_AudioAligner import Orex_AudioAligner

# Объединяем узлы из gigaam_node и наши новые узлы
NODE_CLASS_MAPPINGS = {
    **gigaam_class_mappings,
    "orex Cutting Subtitles": Orex_CuttingSubtitles,
    "orex Audio Aligner": Orex_AudioAligner
}

# Объединяем отображаемые имена
NODE_DISPLAY_NAME_MAPPINGS = {
    **gigaam_name_mappings,
    "orex Cutting Subtitles": "✂️ Cutting Subtitles (OreX)",
    "orex Audio Aligner": "🎛️ Audio Subtitle Aligner (OreX)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("\n\033[34m[GigaAM & OreX Nodes]\033[0m Loaded successfully!")