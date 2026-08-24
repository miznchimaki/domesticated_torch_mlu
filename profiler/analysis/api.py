from typing import Dict, Optional
from .profiler_parser import ProfileData
from .data_context import DataContext


def analyze_data(profiler_data_path: str, data_context: Optional[DataContext] = None):
    profile = ProfileData(profiler_data_path, data_context)
    profile.process()
