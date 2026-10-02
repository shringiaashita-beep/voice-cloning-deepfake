"""Feature Windowing Alias Module for VoxGuard Training Infrastructure.

Exposes FeatureWindowing service for slicing variable-length feature arrays
into fixed 256-frame temporal segments.
"""

from app.training.windowing import FeatureWindowing

__all__ = ["FeatureWindowing"]
