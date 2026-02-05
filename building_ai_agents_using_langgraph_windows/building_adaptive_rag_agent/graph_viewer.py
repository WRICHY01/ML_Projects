from pathlib import Path
from typing import Optional

from .config import config_header

def view_mermaid_graph(mermaid_text: str, file_path: Optional[str] = None) -> None:
    """
    Generate and save a styled Mermaid diagram to a markdown file.

    Args:
        mermaid_text: Raw mermaid diagram text from langgraph
        file_path: Optional custom file path (defaults to `graph.md` in script directory)
    """

    if file_path is None:
        script_dir = Path(__file__).parent
        file_path = script_dir/"graph.md"
        print(f"file_path look like this {script_dir} when None cond. is satisfied")
        print(f"file path look like this {file_path} when None cond. is satisfied")
        # print(f"No file provided. using default: {file_path}")
        print(f"No file provided. Using default: {file_path}")

    else:
        file_path = Path(file_path)

        if not file_path.parent.exists():
            print(f"Directory does not exist: {file_path.parent}")
            print(f"Using default file path instead...")
        elif not file_path.suffix == ".md":
            print("invalid file extension. Must be an '.md' file.")
            print("Using default path instead...")
            # file_path = Path(file_path)
            file_path = Path(__file__).parent/"graph.md"

    # Split Langgraph's config header front matter
    split_parts = mermaid_text.split("---", 2)
    no_of_split_parts = len(split_parts)
    if no_of_split_parts < 3:
        # No frontmatter found,  use original text
        graph_content = mermaid_text
    else:
        # Extract just the graph (skip  empty string and config)
        graph_content = config_header + split_parts[2].strip()

    # Wrap in markdown code fence
    full_content = f"```mermaid\n{graph_content}\n```\n"
    
    # Write to file
    try:
        file_path.write_text(full_content, encoding="utf-8")
        print(f"Mermaid diagram saved: {file_path}")
    except Exception as e:
        print(f"Failed to save diagram: {e}")
        raise e