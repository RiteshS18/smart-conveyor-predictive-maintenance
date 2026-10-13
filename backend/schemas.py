"""Pydantic request models."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SensorIn(BaseModel):
    temperature: Optional[float] = None
    vibration: Optional[float] = None
    load: Optional[float] = None
    speed: Optional[float] = None
    tension: Optional[float] = None


class StreamConnect(BaseModel):
    url: str = Field(..., min_length=1)


class LabelBox(BaseModel):
    cls: int = Field(..., ge=0)
    cx: float = Field(..., ge=0, le=1)
    cy: float = Field(..., ge=0, le=1)
    w: float = Field(..., gt=0, le=1)
    h: float = Field(..., gt=0, le=1)


class LabelsIn(BaseModel):
    boxes: List[LabelBox]


class TrainParams(BaseModel):
    base_model: str = Field("yolov8n", pattern="^yolov8[ns]$")
    epochs: int = Field(30, ge=1, le=500)
    imgsz: int = Field(640, ge=160, le=1280)
    batch: int = Field(8, ge=1, le=64)
    data_source: str = "labeled"        # "labeled" (captured+labeled in the app) or an upload name


class DeployIn(BaseModel):
    version: str


class ToggleIn(BaseModel):
    enabled: Optional[bool] = None
