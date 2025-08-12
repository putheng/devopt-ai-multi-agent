from pathlib import Path
from pydantic import BaseModel, Field
import json
import os
import requests

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

# Enhanced tool models
class DirectoryToolInput(BaseModel):
    directory: str = Field(..., description="Path to directory to list files and subdirectories")

class FileToolInput(BaseModel):
    filepath: str = Field(..., description="Full path to the file to read")
    start_line: int = Field(1, description="Start line number (optional, defaults to 1)")
    end_line: int = Field(-1, description="End line number (optional, -1 means entire file)")

class CollectOrderIdInput(BaseModel):
    order_id: str = Field(..., description="Order ID for customer support")

def tool_schema(model: BaseModel, name: str, description: str):
    return {
        "name": name,
        "description": description,
        "parameters": model.model_json_schema()
    }

def directory_tool(directory: str) -> str:
    """List all files and subdirectories in the given directory"""
    dir_path = Path(directory)
    if not dir_path.exists():
        return f"[ERROR] Directory {directory} not found"
    
    if not dir_path.is_dir():
        return f"[ERROR] {directory} is not a directory"
    
    try:
        items = []
        # List directories first
        for item in sorted(dir_path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                # Show file with size
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                items.append(f"📄 {item.name} ({size_str})")
        
        if not items:
            return f"Directory {directory} is empty"
        
        return f"Contents of {directory}:\n" + "\n".join(items)
    
    except Exception as e:
        return f"[ERROR] Could not read directory {directory}: {e}"

def file_tool(filepath: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read content from a file, optionally with line range"""
    file_path = Path(filepath)
    if not file_path.exists():
        return f"[ERROR] File {filepath} not found"
    
    if file_path.is_dir():
        return f"[ERROR] {filepath} is a directory, not a file. Use directory_tool instead."
    
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # Handle line range
        if end_line == -1:
            end_line = len(lines)
        
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)
        
        if start_line > len(lines):
            return f"[ERROR] Start line {start_line} is beyond file length ({len(lines)} lines)"
        
        selected_lines = lines[start_line - 1:end_line]
        
        # Add line numbers for easier reference
        numbered_lines = []
        for i, line in enumerate(selected_lines, start=start_line):
            numbered_lines.append(f"{i:4d}: {line}")
        
        result = f"File: {filepath}\n"
        if start_line == 1 and end_line == len(lines):
            result += f"Full content ({len(lines)} lines):\n"
        else:
            result += f"Lines {start_line}-{end_line} of {len(lines)} total lines:\n"
        
        result += "\n".join(numbered_lines)
        return result
    
    except UnicodeDecodeError:
        return f"[ERROR] Could not decode file {filepath} - it may be a binary file"
    except Exception as e:
        return f"[ERROR] Could not read file {filepath}: {e}"

def get_order_detail(order_id: int):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"{API_URL}/orders?order_id=eq.{order_id}",
        headers={
            "apikey": API_KEY,
            "Content-Type": "application/json"
        }
    )

    return json.dumps(response.json())

def call_function(name, args):
    if name == "directory_tool":
        return directory_tool(**args)
    elif name == "file_tool":
        return file_tool(**args)
    elif name == "get_order_detail":
        return get_order_detail(**args)