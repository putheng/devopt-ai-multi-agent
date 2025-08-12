from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class AgentType(str, Enum):
    """Available agent types for routing user requests"""
    DEVELOPER_ASSISTANT = "developer_assistant"
    CUSTOMER_ASSISTANT = "customer_assistant"
    GENERAL_ASSISTANT = "general_assistant"

class ExtractedParams(BaseModel):
    """Structured parameters extracted from user request"""
    directory: Optional[str] = Field(default=None, description="Directory path")
    file_path: Optional[str] = Field(default=None, description="File path")
    order_id: Optional[str] = Field(default=None, description="Order ID for customer support")

class RouteClassification(BaseModel):
    """Classification result for routing user queries to appropriate agents"""
    agent_type: AgentType = Field(..., 
        description="The type of agent that should handle this request"
    )
    confidence: float = Field(..., ge=0.0, le=1.0,
        description="Confidence score for the classification (0.0 to 1.0)"
    )
    reasoning: str = Field(..., 
        description="Brief explanation of why this agent was selected"
    )
    extracted_params: Optional[ExtractedParams] = Field(
        default=None,
        description="Key parameters extracted from the user request"
    )
    
    class Config:
        extra = "forbid"  # This ensures additionalProperties: false
