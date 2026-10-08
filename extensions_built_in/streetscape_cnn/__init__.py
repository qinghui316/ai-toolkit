"""Opt-in streetscape research extension; ordinary trainers remain untouched."""
from toolkit.extension import Extension


class StreetscapeCNNExtension(Extension):
    uid = "streetscape_cnn_trainer"
    name = "Krea 2 frozen streetscape CNN trainer"

    @classmethod
    def get_process(cls):
        from .trainer import StreetscapeCNNTrainer

        return StreetscapeCNNTrainer


AI_TOOLKIT_EXTENSIONS = [StreetscapeCNNExtension]
