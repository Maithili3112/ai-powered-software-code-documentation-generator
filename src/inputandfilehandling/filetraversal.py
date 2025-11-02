"""
Simple, dependency-light file system traversal utilities used by the chunker.
Provides callbacks for folders and files with basic ignore pattern support.
Detects text files via extension set and lightweight content sniffing.
"""
from pathlib import Path
import fnmatch
from typing import Callable, List, Optional

# Simple text file detection without magic
def is_text_file(file_path: str) -> bool:
    """Simple text file detection based on extension and content sampling."""
    text_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp', 
                      '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sh', 
                      '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.sql', '.html', '.htm', 
                      '.css', '.scss', '.sass', '.less', '.xml', '.json', '.yaml', '.yml', 
                      '.toml', '.ini', '.cfg', '.conf', '.txt', '.md', '.rst', '.tex', '.r', 
                      '.m', '.pl', '.pm', '.tcl', '.lua', '.dart', '.elm', '.hs', '.ml', '.fs', 
                      '.vb', '.pas', '.ada', '.asm', '.s', '.f', '.f90', '.f95', '.f03', '.f08'}
    
    file_path_obj = Path(file_path)
    if file_path_obj.suffix.lower() in text_extensions:
        return True
    
    # Try to read first 1024 bytes to check if it's text
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # Check if it contains null bytes (binary indicator)
            if b'\x00' in chunk:
                return False
            # Try to decode as UTF-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
    except Exception:
        return False

class ProcessFileParams:
    """Data passed to file processing callback"""
    def __init__(self, file_name: str, file_path: str):
        self.file_name = file_name
        self.file_path = file_path

class ProcessFolderParams:
    """Data passed to folder processing callback"""
    def __init__(self, folder_name: str, folder_path: str, should_ignore: Callable[[str], bool]):
        self.folder_name = folder_name
        self.folder_path = folder_path
        self.should_ignore = should_ignore

class TraverseFileSystemParams:
    """Configuration for traversal"""
    def __init__(
        self,
        input_path: str,
        process_file: Optional[Callable[[ProcessFileParams], None]] = None,
        process_folder: Optional[Callable[[ProcessFolderParams], None]] = None,
        ignore: Optional[List[str]] = None,
        chunk_size: int = 500
    ):
        self.input_path = input_path
        self.process_file = process_file
        self.process_folder = process_folder
        self.ignore = ignore or []
        self.chunk_size = chunk_size

def traverse_file_system(params: TraverseFileSystemParams):
    """Traverse File System"""
    try:
        input_path = Path(params.input_path)
        if not input_path.exists():
            print("The provided folder path does not exist.")
            return

        def should_ignore(file_name: str):
            return any(fnmatch.fnmatch(file_name, pattern) for pattern in params.ignore)

        def dfs(current_path: Path):
          print(f"Entering folder: {current_path}") 
          contents = list(current_path.iterdir())
      
          for entry in contents:
              if entry.is_dir():
                  if should_ignore(entry.name):
                      print(f"Skipping folder: {entry.name}")
                      continue
                  print(f"Found folder: {entry.name}")
                  if params.process_folder:
                      params.process_folder(
                          ProcessFolderParams(
                              entry.name,
                              str(entry),
                              should_ignore,
                          )
                      )
                  dfs(entry)
      
          for entry in contents:
              if entry.is_file():
                  if should_ignore(entry.name):
                      print(f"Skipping file: {entry.name}")
                      continue
                  print(f"Found file: {entry.name}")
                  file_path = str(entry)
                  try:
                      if is_text_file(file_path):
                          if params.process_file:
                              params.process_file(
                                  ProcessFileParams(
                                      entry.name,
                                      file_path,
                                  )
                              )
                  except Exception as e:
                      print(f"Could not process file {file_path}: {e}")


        dfs(input_path)

    except RuntimeError as e:
        print(f"Error during traversal: {e}")

def main():
    # Minimal interactive example
    folder = input("Enter a folder to traverse: ").strip()
    if not folder:
        print("No folder provided.")
        return

    params = TraverseFileSystemParams(
        input_path=folder,
        process_file=lambda fp: print(f"Processing file: {fp.file_path}"),
        process_folder=lambda fr: print(f"Processing folder: {fr.folder_path}"),
        ignore=[
            "__pycache__", "*.pyc", ".venv", "env", ".env",
            ".git", ".gitignore", ".gitattributes",
            "node_modules", "package-lock.json", "yarn.lock",
            ".idea", ".vscode", "*.sublime-*",
            ".DS_Store", "Thumbs.db",
            "*.log", "*.tmp", "*.swp",
            "Dockerfile", "*.dockerfile", ".dockerignore",
            "*.env", ".env.example", "venv", "*.egg-info"
        ]
    )
    traverse_file_system(params)

if __name__ == "__main__":
    main()
