"""
Input/Output schemas and validation for workflow
Configuration and workflow input/output definitions
"""


from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

"""
Schema definitions and validation for HLD generation
"""

# Re-export main models for convenience
from .models import (
    HLDState,
    ProcessingStatus,
    ExtractedContent,
    AuthenticationData,
    IntegrationData,
    EntityData,
    APIData,
    DomainData,
    RiskData,
    BehaviorData,
    DiagramData,
    OutputData
)

class ConfigSchema(BaseModel):
    """Configuration schema for HLD generation"""
    render_images: bool = True
    image_format: str = Field(default="png", pattern="^(svg|png)$")
    renderer: str = Field(default="kroki", pattern="^(kroki|mmdc)$")
    save_sources: bool = True
    theme: str = Field(default="default", pattern="^(default|neutral|dark)$")
    
    @field_validator('image_format')
    @classmethod
    def validate_image_format(cls, v):
        if v not in ['svg', 'png']:
            raise ValueError('image_format must be svg or png')
        return v
    
    @field_validator('renderer')
    @classmethod
    def validate_renderer(cls, v):
        if v not in ['kroki', 'mmdc']:
            raise ValueError('renderer must be kroki or mmdc')
        return v

class WorkflowInput(BaseModel):
    """Input schema for the HLD generation workflow"""
    pdf_path: str
    requirement_name: Optional[str] = None
    config: ConfigSchema = Field(default_factory=ConfigSchema)
    
    @field_validator('pdf_path')
    @classmethod
    def validate_pdf_path(cls, v):
        if not v.endswith('.pdf'):
            raise ValueError('pdf_path must end with .pdf')
        return v

class WorkflowOutput(BaseModel):
    """Output schema for the HLD generation workflow"""
    success: bool
    state: HLDState
    output_paths: Dict[str, Optional[str]] = Field(default_factory=dict)
    processing_time: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# Type aliases for better readability
StateDict = Dict[str, Any]
NodeResult = Dict[str, Any]
AgentResponse = Union[Dict[str, Any], str]

# Constants
WORKFLOW_STAGES = [
    "pdf_extraction",
    "auth_integrations", 
    "domain_api_design",
    "behavior_quality",
    "diagram_generation",
    "output_composition"
]

REQUIRED_GEMINI_KEYS = [
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_1", 
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_4"
]

DEFAULT_CONFIG = ConfigSchema()

def validate_state(state: Dict[str, Any]) -> HLDState:
    """Validate and convert dict to HLDState"""
    return HLDState(**state)

def create_initial_state(pdf_path: str, config: ConfigSchema) -> HLDState:
    """Create initial state for workflow"""
    from pathlib import Path
    
    requirement_name = Path(pdf_path).stem
    
    return HLDState(
        pdf_path=pdf_path,
        requirement_name=requirement_name,
        config=config.dict(),
        status={stage: ProcessingStatus(stage=stage) for stage in WORKFLOW_STAGES}
    )
