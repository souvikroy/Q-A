"""File handling utilities for document analysis."""
import os
from pathlib import Path
from typing import List, Dict, Union, BinaryIO
import json
import tempfile

def safe_file_path(file_path: Union[str, Path]) -> Path:
    """Convert file path to Path object and ensure it exists."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

def get_file_metadata(file_path: Union[str, Path]) -> Dict:
    """Get metadata about a file."""
    path = safe_file_path(file_path)
    return {
        'name': path.name,
        'extension': path.suffix,
        'size': path.stat().st_size,
        'modified': path.stat().st_mtime,
        'path': str(path.absolute())
    }

def create_temp_file(content: Union[str, bytes], suffix: str = None) -> str:
    """Create a temporary file with the given content."""
    mode = 'wb' if isinstance(content, bytes) else 'w'
    encoding = None if isinstance(content, bytes) else 'utf-8'
    
    with tempfile.NamedTemporaryFile(
        mode=mode, 
        suffix=suffix,
        encoding=encoding,
        delete=False
    ) as tmp_file:
        tmp_file.write(content)
        return tmp_file.name

def save_json(data: Dict, file_path: Union[str, Path]) -> None:
    """Save data as JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path: Union[str, Path]) -> Dict:
    """Load data from JSON file."""
    path = safe_file_path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def cleanup_temp_files(temp_files: List[str]) -> None:
    """Clean up temporary files."""
    for file_path in temp_files:
        try:
            os.unlink(file_path)
        except (OSError, FileNotFoundError):
            pass  # Ignore errors during cleanup
