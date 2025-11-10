#!/usr/bin/env python3
"""Remove icon files (small images) from downloaded image directories."""
from pathlib import Path
from PIL import Image

def remove_icons(directory: Path, min_size: int = 50):
    """Remove icon files (images smaller than min_size x min_size)."""
    removed = 0
    errors = 0
    
    for img_path in directory.rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                file_size = img_path.stat().st_size
                
                # 50x50 이하 또는 64x64 이하이면서 파일 크기가 2000바이트 미만인 경우 제거
                if width <= min_size or height <= min_size:
                    print(f"Removing icon: {img_path.name} ({width}x{height}, {file_size} bytes)")
                    img_path.unlink()
                    removed += 1
                elif width <= 64 and height <= 64 and file_size < 2000:
                    print(f"Removing small icon: {img_path.name} ({width}x{height}, {file_size} bytes)")
                    img_path.unlink()
                    removed += 1
        except Exception as e:
            print(f"Error checking {img_path.name}: {e}")
            errors += 1
    
    print(f"\nRemoved {removed} icon files")
    if errors > 0:
        print(f"Encountered {errors} errors")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/remove_icons.py <directory>")
        sys.exit(1)
    
    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)
    
    remove_icons(target_dir)

