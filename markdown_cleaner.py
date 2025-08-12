import re
from typing import List

class MarkdownCleaner:
    def __init__(self):
        self.cleaning_rules = [
            (r'<!-- image -->', ''),  # Remove image placeholders
            (r'<!--.*?-->', ''),      # Remove all HTML comments
            (r'^###\s*$', ''),        # Remove empty h3 headers
            (r'^##\s*$', ''),         # Remove empty h2 headers
            (r'&amp;', '&'),          # Fix HTML entities
            (r'\n{3,}', '\n\n'),      # Remove excessive newlines
            (r'^\s*-\s*$', ''),       # Remove empty list items
            (r'[ \t]+$', ''),         # Remove trailing whitespace
        ]
    
    def clean_file(self, input_path: str, output_path: str = None) -> str:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = self.clean_content(content)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
        
        return cleaned
    
    def clean_content(self, content: str) -> str:
        # Apply all cleaning rules
        for pattern, replacement in self.cleaning_rules:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # Extract meaningful sections
        sections = self._extract_meaningful_sections(content)
        
        return '\n\n'.join(sections)
    
    def _extract_meaningful_sections(self, content: str) -> List[str]:
        """Extract sections that contain substantial content"""
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            if line.startswith('#') and current_section:
                if self._has_content(current_section):
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add final section
        if current_section and self._has_content(current_section):
            sections.append(current_section.strip())
        
        return sections
    
    def _has_content(self, section: str) -> bool:
        """Check if section has meaningful content"""
        # Remove headers and get content
        content = re.sub(r'^#{1,6}.*$', '', section, flags=re.MULTILINE)
        
        # Count meaningful characters (letters, numbers)
        meaningful = re.findall(r'\w', content)
        
        return len(meaningful) > 15  # Threshold for meaningful content