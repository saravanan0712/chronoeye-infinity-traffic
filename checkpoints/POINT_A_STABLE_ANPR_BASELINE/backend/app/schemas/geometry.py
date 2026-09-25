"""
ChronoEye Infinity - Geometry Schemas Compatibility Module
Re-exports canonical geometry / bounding box schemas to ensure backward compatibility across all modules and tests.
"""

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.simulation import BoundingBox

__all__ = ["BoundingBoxXYXY", "BoundingBox"]
