from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI
from models import AgentType
from tools import DirectoryToolInput, FileToolInput, CollectOrderIdInput
import tools
import json
from datetime import datetime
import streamlit as st

load_dotenv("../.env")

# Base class for all specialized agents
class BaseAgent(ABC):
    """Base class for all specialized agents"""

    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o"

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt specific to this agent"""
        pass

    @abstractmethod
    def get_available_tools(self) -> list:
        """Get the tools available to this agent"""
        pass

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool call - can be overridden by specific agents"""

        if tool_name == "directory_tool":
            return tools.directory_tool(**args)
        elif tool_name == "file_tool":
            return tools.file_tool(**args)
        elif tool_name == "get_order_detail":
            return tools.get_order_detail(**args)
        else:
            return "Unknown tool"

    def process_request(self, user_prompt: str, extracted_params: dict = None) -> str:
        """Process the user request with this agent"""

        system_prompt = self.get_system_prompt()

        # Retrieve user memories
        relevant_memories = extracted_params.get("memory_context", "")
        if relevant_memories:
            system_prompt += f"\n\n**User Memories:**\n{relevant_memories}"

        # Perform RAG search to get relevant documents
        rag_context = extracted_params.get("rag_context", "")
        if rag_context:
            system_prompt += f"\n\n**Company Documents:**\n{rag_context or '_No relevant documents found_'}"

        # Get today's date context
        today = datetime.now()
        date_context = f"Today is {today.strftime('%A, %B %d, %Y at %I:%M:%S %p')}."
        system_prompt += f"\n\n**Date Context:**\n{date_context}\n\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        full_response = ""
        tool_calls = []
        used_tools = []  # Track all tools used during the conversation
        
        # Create status placeholders
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        # Always use fallback for now to ensure it works
        status_placeholder.info(f"🤖 {self.__class__.__name__} is processing your request...")

        while True:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.get_available_tools(),
                stream=True
            )
            
            for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                if delta and delta.content:
                    full_response += delta.content
                    response_placeholder.markdown(full_response)
                
                elif delta and delta.tool_calls:
                    for tool_chunk in delta.tool_calls:
                        if len(tool_calls) <= tool_chunk.index:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        tc = tool_calls[tool_chunk.index]
                        
                        if tool_chunk.id:
                            tc["id"] += tool_chunk.id
                        
                        if tool_chunk.function.name:
                            tc["function"]["name"] += tool_chunk.function.name
                        
                        if tool_chunk.function.arguments:
                            tc["function"]["arguments"] += tool_chunk.function.arguments
                
                # Handle tool calls if needed
                if finish_reason == "tool_calls" and tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": call["function"]["arguments"]
                                }
                            } for call in tool_calls
                        ]
                    })

                    for call in tool_calls:
                        call_id = call["id"]
                        tool_name = call["function"]["name"]
                        
                        # Track the tool usage
                        used_tools.append(tool_name)

                        try:
                            args = json.loads(call["function"]["arguments"])
                        except json.JSONDecodeError as e:
                            continue
                        
                        try:
                            result = self.execute_tool(tool_name, args)
                        except Exception as e:
                            result = f"Error executing tool {tool_name}: {str(e)}"

                        messages.append({
                            "role": "tool",
                            "tool_name": tool_name,
                            "tool_call_id": call_id,
                            "content": result
                        })

                    tool_calls = []

            # Break if conversation is complete
            if finish_reason == "stop":
                status_placeholder.success("✅ Analysis complete!")
                break
        
        # Show tool usage if any tools were used
        if used_tools:
            with st.expander("🤖 Tool Usage", expanded=False):
                for tool_name in used_tools:
                    st.write(f"**{tool_name}**")

        return full_response

# Customer Support Agent
class CustomerAssistantAgent(BaseAgent):
    """Specialized agent for customer support and order inquiries"""

    def get_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant. Answer the user's question based on the provided Company Documents and User Memories. "
            "If the information is not available in either source, clearly state that you don't have that information.\n\n"
            "Instructions:\n"
            "- Use information from both User Memories and Company Documents\n"
            "- User Memories contain personal context and conversation history\n"
            "- Company Documents contain official business information\n"
            "- If neither source contains the answer, say so clearly\n"
            "- You can still use available tools if they help answer the user's question"
        )
    
    def get_available_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": tools.tool_schema(CollectOrderIdInput, "get_order_detail", "Read specific order ID for customer support")
            }
        ]

# Developer Assistant Agent
class DeveloperAssistantAgent(BaseAgent):
    """Specialized agent for developer-related inquiries"""

    def get_system_prompt(self) -> str:
        return """You are an autonomous Log Analysis, Code Analysis and Fix Proposal Agent. Your expertise is in:

1. **Log File Analysis**: Reading and interpreting various log formats
2. **Error Pattern Recognition**: Identifying common error patterns and anomalies
3. **Timeline Analysis**: Understanding sequence of events in logs
4. **Performance Issue Detection**: Spotting performance bottlenecks in logs
5. **Code Analysis**: Examining source code for bugs, inefficiencies, and potential issues
6. **Root Cause Analysis**: Correlating log errors with source code to identify root causes
7. **Fix Proposals**: Suggesting concrete code fixes and improvements based on analysis

**Available Tools:**
- `directory_tool`: Use this to list all files and subdirectories in a given directory
- `file_tool`: Use this to read content from a specific file, with optional line range

**Autonomous Decision Making:**
- If user asks to analyze a directory: First use `directory_tool` to see what files are available, then use `file_tool` to examine specific files
- If user asks to analyze specific files: Use `file_tool` directly to get file content
- If user asks for log analysis: Use `directory_tool` to find log files, then `file_tool` to analyze them
- If user asks for code analysis and fixes: Use `directory_tool` to explore the structure, then `file_tool` to examine code files and propose fixes

**Your Workflow:**
1. **Understand the request** - determine if you need to explore directories or read specific files
2. **Explore strategically** - use `directory_tool` to understand project structure when needed
3. **Read targeted files** - use `file_tool` to get specific file content for analysis
4. **Analyze thoroughly** - identify patterns, errors, and root causes
5. **Provide actionable solutions** - concrete fixes with code examples when requested

**Output Format:**
Always format your responses using proper markdown:
- Use headers (# ## ###) to structure your analysis
- Use code blocks (```language) for code examples and log snippets
- Use bullet points and numbered lists for findings
- Use **bold** for important findings and *italic* for emphasis
- Use tables when comparing multiple items
- Use blockquotes (>) for important warnings or notes

**Priority Order:**
- Critical errors and security issues
- Performance bottlenecks  
- Code quality improvements
- Warnings and potential issues


**Fix Proposal Format:**
## Issue Analysis
**Problem:** [clear description of the issue]
**Root Cause:** [underlying cause analysis]
**Impact:** [current impact on system/users]

## Proposed Solution
**Approach:** [high-level solution strategy]
**Components Affected:** [what will be changed]

## Implementation Steps
1. [Step 1 with specific actions]
2. [Step 2 with specific actions]
3. [Continue...]

## Code Changes
### Current Code
```[language]
[problematic code]
```

### Proposed Fix
```diff
@@ -[start_line],[num_lines] +[start_line],[num_lines] @@
- [line_num]: [lines to be removed]
+ [line_num]: [lines to be added]
  [line_num]: [unchanged context lines]
```

### Complete Fixed Code
```[language]
[corrected code with explanations]
```

Always be autonomous in your tool usage - decide which tools to use based on the user's request and your analysis findings."""
    
    def get_available_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": tools.tool_schema(DirectoryToolInput, "directory_tool", "List all files and subdirectories in a given directory")
            },
            {
                "type": "function",
                "function": tools.tool_schema(FileToolInput, "file_tool", "Read content from a specific file, optionally with line range")
            }
        ]

# General Assistant Agent
class GeneralAssistantAgent(BaseAgent):
    """General purpose assistant for unclear or general requests"""
    
    def get_system_prompt(self) -> str:
        return """You are a General Assistant Agent that helps users with various development and debugging tasks.

Your role is to:
1. **Clarify Requirements**: Help users articulate their specific needs
2. **Provide Guidance**: Offer general development advice and best practices
3. **Route to Specialists**: Suggest when a specialized agent would be more helpful
4. **Educational Support**: Explain concepts and provide learning resources

When you are not sure what the user needs, ask clarifying questions to better understand their requirements."""

    def get_available_tools(self) -> list:
        return []  # General assistant uses no tools initially

# Agent factory
class AgentFactory:
    """Factory for creating appropriate agent instances"""
    
    @staticmethod
    def create_agent(agent_type: str) -> BaseAgent:
        if agent_type == AgentType.DEVELOPER_ASSISTANT.value:
            return DeveloperAssistantAgent()
        elif agent_type == AgentType.CUSTOMER_ASSISTANT.value:
            return CustomerAssistantAgent()
        else:
            return GeneralAssistantAgent()