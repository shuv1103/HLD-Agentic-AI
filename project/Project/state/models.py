"""
Pydantic data models for HLD workflow state management
Type-safe state structures for all stages of HLD generation
"""

from typing import Dict, List, Optional, Any, Literal

from datetime import datetime
from pydantic import BaseModel, Field, conint, validator
from pathlib import Path


"""
LangGraph State Models for HLD Generation
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

class ProcessingStatus(BaseModel):
    """Track processing status for each stage"""
    stage: str
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

class ExtractedContent(BaseModel):
    """PDF extraction results"""
    markdown: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.1"
    generated_at: str = ""
    source: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None

class AuthenticationData(BaseModel):
    """Authentication analysis results"""
    actors: List[str] = Field(default_factory=list)
    flows: List[str] = Field(default_factory=list)
    idp_options: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)

class IntegrationData(BaseModel):
    """Integration system data"""
    system: str = ""
    purpose: str = ""
    protocol: str = ""
    auth: str = ""
    endpoints: List[str] = Field(default_factory=list)
    data_contract: Dict[str, List[str]] = Field(default_factory=dict)

class EntityData(BaseModel):
    """Domain entity data"""
    name: str = ""
    attributes: List[str] = Field(default_factory=list)

class APIData(BaseModel):
    """API specification data"""
    name: str = ""
    description: str = ""
    request: Dict[str, str] = Field(default_factory=dict)
    response: Dict[str, str] = Field(default_factory=dict)

class DomainData(BaseModel):
    """Domain and API design results"""
    entities: List[EntityData] = Field(default_factory=list)
    apis: List[APIData] = Field(default_factory=list)
    diagram_plan: Dict[str, Any] = Field(default_factory=dict)

class RiskData(BaseModel):
    """Risk assessment data"""
    id: str = ""
    desc: str = ""
    assumption: str = ""
    mitigation: str = ""
    impact: int = 3
    likelihood: int = 3

class BehaviorData(BaseModel):
    """Behavior and quality results"""
    use_cases: List[str] = Field(default_factory=list)
    nfrs: Dict[str, List[str]] = Field(default_factory=dict)
    risks: List[RiskData] = Field(default_factory=list)
    diagram_plan: Dict[str, Any] = Field(default_factory=dict)

class DiagramData(BaseModel):
    """Generated diagram information"""
    class_text: str = ""
    sequence_texts: List[str] = Field(default_factory=list)
    class_img_path: Optional[str] = None
    seq_img_paths: List[Optional[str]] = Field(default_factory=list)
    mermaid_map: Dict[str, str] = Field(default_factory=dict)

class OutputData(BaseModel):
    """Final output paths and metadata"""
    requirement_name: str = ""
    output_dir: str = ""
    hld_md_path: Optional[str] = None
    hld_html_path: Optional[str] = None
    diagrams_html_path: Optional[str] = None
    risk_heatmap_path: Optional[str] = None

class HLDState(BaseModel):
    """Main LangGraph state for HLD generation workflow"""
    
    # Input
    pdf_path: str = ""
    requirement_name: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing status
    status: Dict[str, ProcessingStatus] = Field(default_factory=dict)
    
    # Stage results
    extracted: Optional[ExtractedContent] = None
    authentication: Optional[AuthenticationData] = None
    integrations: List[IntegrationData] = Field(default_factory=list)
    domain: Optional[DomainData] = None
    behavior: Optional[BehaviorData] = None
    diagrams: Optional[DiagramData] = None
    output: Optional[OutputData] = None
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def update_status(self, stage: str, status: str, message: str = None, error: str = None):
        """Update processing status for a stage"""
        self.status[stage] = ProcessingStatus(
            stage=stage,
            status=status,
            message=message,
            error=error
        )
        self.updated_at = datetime.now()
    
    def update_state_status(self, stage:str,status:str,message:str=None,error:str=None):
        return self.update_status(stage,status,message,error)
    
    def add_error(self, error: str):
        """Add error to the state"""
        self.errors.append(error)
        self.updated_at = datetime.now()
    
    def add_warning(self, warning: str):
        """Add warning to the state"""
        self.warnings.append(warning)
        self.updated_at = datetime.now()
    
    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed successfully"""
        return (stage in self.status and 
                self.status[stage].status == "completed")
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0 or any(
            status.status == "failed" for status in self.status.values()
        )

