"""Output schemas for the visual-primitives pipeline.

Pydantic models used for validation and serialisation of agent outputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EquipmentClass(str, Enum):
    """The 7-class taxonomy from the implementation plan."""

    DISTILLATION_COLUMN = "distillation_column"
    PRESSURE_VESSEL = "pressure_vessel"
    HEAT_EXCHANGER = "heat_exchanger"
    STORAGE_TANK = "storage_tank"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    REACTOR = "reactor"
    OTHER = "other"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Equipment
# ---------------------------------------------------------------------------


class EquipmentEntry(BaseModel):
    """A single piece of equipment detected by Agent 1."""

    tag: str = Field(..., description='Equipment tag number, e.g. "C-201". Use "unknown" if unreadable.')
    equipment_class: EquipmentClass = Field(..., description="Equipment type per the 7-class taxonomy.")
    global_bbox: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in normalized [0, 999] space.",
    )
    confidence: Confidence = Field(default=Confidence.MEDIUM, description="Detection confidence.")
    description: Optional[str] = Field(default=None, description="Extra notes (used when class=other).")

    @field_validator("global_bbox")
    @classmethod
    def validate_bbox(cls, v: list[int]) -> list[int]:
        if not all(0 <= c <= 999 for c in v):
            raise ValueError(f"bbox coordinates must be in [0, 999], got {v}")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError(f"bbox must have x1<x2 and y1<y2, got {v}")
        return v


class EquipmentRegistry(BaseModel):
    """Full output of Agent 1 — Global Equipment Detector."""

    equipment: list[EquipmentEntry] = Field(default_factory=list)
    drawing_notes: str = Field(default="", description="Brief observations about drawing quality or anomalies.")
    total_count: int = Field(default=0, description="Total equipment items detected.")

    def model_post_init(self, __context):
        if self.total_count == 0:
            object.__setattr__(self, "total_count", len(self.equipment))


# ---------------------------------------------------------------------------
# Pipeline tracing (Agent 2)
# ---------------------------------------------------------------------------


class TraceDirection(str, Enum):
    """Cardinal directions for step-by-step line following."""

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class TraceTokenType(str, Enum):
    """Token types in the VLM tracing protocol."""

    STEP = "step"
    HIT = "hit"
    TERM = "term"


class TraceStep(BaseModel):
    """One step in a VLM-guided pipeline trace."""

    token_type: TraceTokenType
    direction: Optional[TraceDirection] = Field(default=None, description="Direction for STEP tokens.")
    distance_px: Optional[int] = Field(default=None, ge=1, description="Pixel distance for STEP tokens.")
    symbol_class: Optional[str] = Field(default=None, description="Symbol class for HIT/TERM tokens.")
    symbol_tag: Optional[str] = Field(default=None, description="Optional tag number for HIT/TERM symbols.")
    symbol_bbox_view: Optional[list[int]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Bounding box [x1,y1,x2,y2] in crop-view coordinates.",
    )
    confidence: str = Field(default="medium", description="Detection confidence.")


class TraceSegment(BaseModel):
    """A traced pipeline segment from one anchor to a terminal."""

    anchor_id: str = Field(..., description="Source page-connection object id (e.g. obj_000118).")
    anchor_bbox_global: list[int] = Field(..., description="Anchor bbox [x1,y1,x2,y2] in normalized [0,999] space.")
    start_point_global: list[int] = Field(..., description="Port point [x,y] in normalized [0,999] space.")
    start_direction: TraceDirection = Field(..., description="Initial trace direction from the anchor.")
    steps: list[TraceStep] = Field(default_factory=list)
    terminal_class: str = Field(..., description="Terminal type — equipment class, tee, page_connection, or sheet_edge.")
    terminal_tag: Optional[str] = Field(default=None, description="Tag number of terminal equipment, if any.")
    terminal_point_global: list[int] = Field(..., description="Terminal point [x,y] in normalized [0,999] space.")
    terminal_bbox_global: Optional[list[int]] = Field(default=None, description="Terminal bbox [x1,y1,x2,y2] in [0,999].")
    total_length_px: int = Field(default=0, ge=0, description="Total traced path length in original-image pixels.")


class TraceResult(BaseModel):
    """Full tracing output for one P&amp;ID sheet."""

    source_image: str = Field(..., description="Source image path.")
    model: str = Field(..., description="VLM model used for tracing.")
    source_dimensions: list[int] = Field(..., description="Original image [width, height] in pixels.")
    segments: list[TraceSegment] = Field(default_factory=list)
    total_segments: int = Field(default=0)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0

    def model_post_init(self, __context):
        if self.total_segments == 0:
            object.__setattr__(self, "total_segments", len(self.segments))


# ---------------------------------------------------------------------------


class StageArtifactMeta(BaseModel):
    """Metadata attached to every stage artifact for traceability."""

    agent: str
    model: str
    source_image: str
    source_dimensions: tuple[int, int]  # (width, height) pixels
    global_view_dimensions: tuple[int, int]  # downsampled view size
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
