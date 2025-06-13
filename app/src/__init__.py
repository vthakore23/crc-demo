# CRC Analysis Pipeline - src package
from .config import Config
from .preprocessing import TissueSegmenter, StainNormalizer, WSIPreprocessor
from .feature_extraction import *
from .models import *
from .training import *

__version__ = "1.0.0" 