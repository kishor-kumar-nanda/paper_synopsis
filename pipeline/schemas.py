from typing import List, Literal
from pydantic import BaseModel, Field

class TextUnderstanding(BaseModel):
    main_idea: str
    equations_explained: List[str]
    key_claims: List[str]

class VisionResponse(BaseModel):
    description: str
    key_elements: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

class ReflectorResponse(BaseModel):
    is_accurate: bool
    score: int = Field(..., ge=1, le=10)
    critique: str
    missing_info: List[str]
    hallucinated_claims: List[str]


class SynthesizedOutput(BaseModel):
    final_explanation: str
    diagram_role: str
    confidence: float

ExitReason = Literal[
    "quality_met",
    "max_retries",
    "no_improvement",
    "low_confidence"
]