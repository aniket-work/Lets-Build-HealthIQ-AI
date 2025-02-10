import yaml
import json, os
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_json_config(path: Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(path, 'r') as f:
        return json.load(f)

def setup_environment(hf_api_key: Optional[str] = None) -> None:
    """Setup environment variables."""
    if hf_api_key:
        os.environ["HF_API_KEY"] = hf_api_key

def format_markdown(text: str) -> str:
    """Format text as markdown with proper indentation."""
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent