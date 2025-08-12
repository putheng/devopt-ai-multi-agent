from openai import OpenAI

from models import RouteClassification
from models import AgentType, ExtractedParams

from dotenv import load_dotenv
load_dotenv("../.env")

class AgentRouter:
    """Routes user queries to appropriate specialized agents using OpenAI structured outputs"""
    
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o"

    def classify_request(self, user_prompt: str) -> RouteClassification:
        system_prompt = """
You are an intelligent routing system that classifies user requests and directs them to specialized agents.

Available Agent Types:
- DEVELOPER_ASSISTANT: For analyzing logs, debugging code, proposing fixes
- CUSTOMER_ASSISTANT: For customer support, order inquiries, billing questions, refunds

Guidelines:
1. Analyze the user's intent and keywords
2. Choose the most appropriate agent type
3. Extract relevant parameters from the request into the structured format
4. Provide reasoning for your choice
5. Assign confidence score based on clarity of intent

Parameter Extraction:
- directory: Extract any directory paths mentioned (e.g., "project/logs", "project/src")
- file_path: Extract specific file paths (e.g., "index.js", "main.py")
- order_id: Extract order numbers or IDs if mentioned

Examples:
- "analyze logs in project/logs" → DEVELOPER_ASSISTANT, directory="project/logs"
- "check status of order 12345" → CUSTOMER_ASSISTANT, order_id="12345", category="orders"
- "I need a refund for my purchase" → CUSTOMER_ASSISTANT, category="refunds"
"""

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this request and extract parameters: {user_prompt}"}
                ],
                response_format=RouteClassification,
                temperature=0.1  # Lower temperature for more consistent routing
            )
            
            return response.choices[0].message.parsed
            
        except Exception as e:
            # Fallback to general assistant if routing fails
            return RouteClassification(
                agent_type=AgentType.GENERAL_ASSISTANT,
                confidence=0.5,
                reasoning=f"Routing failed, defaulting to general assistant: {str(e)}",
                extracted_params=ExtractedParams(category="fallback")
            )
    
    def extract_specific_params(self, classification: RouteClassification, user_prompt: str) -> dict:
        if classification.extracted_params:
            # Convert Pydantic model to dict, excluding None values
            extracted_params = classification.extracted_params.model_dump(exclude_none=True)
            # Add the original prompt if no params were extracted
            if not extracted_params:
                extracted_params["prompt"] = user_prompt

            return extracted_params
        else:
            return {"prompt": user_prompt}

# Global router instance
router = AgentRouter()