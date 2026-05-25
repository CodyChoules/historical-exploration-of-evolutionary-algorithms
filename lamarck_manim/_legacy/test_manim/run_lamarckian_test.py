#!/usr/bin/env python3
"""
Script to run the Lamarckian Functions test scene.

This script handles:
- Version number incrementing
- Running the Manim scene
- Opening the output file automatically

Usage:
    python test_manim/run_lamarckian_test.py
"""

import os
import subprocess
import sys
import time
import re
from pathlib import Path


def detect_color_scheme(scene_file_path):
    """
    Detect color scheme preset from scene file by searching for color_scheme patterns.
    
    Returns:
        str or None: Detected color scheme name (or None if not found)
    """
    try:
        # Import color scheme presets and aliases
        sys.path.insert(0, str(scene_file_path.parent))
        from retro_configuration import COLOR_SCHEME_PRESETS, COLOR_SCHEME_ALIASES
        
        # Read scene file content
        with open(scene_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Search for color_scheme patterns (handles both 'color_scheme' and 'COLOR_SCHEME')
        # Look for patterns like: 'color_scheme': 'bw' or "color_scheme": "wb" etc.
        patterns = [
            r"['\"]color_scheme['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            r"['\"]COLOR_SCHEME['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                scheme = matches[0].strip()
                # Resolve alias if needed
                resolved_scheme = COLOR_SCHEME_ALIASES.get(scheme, scheme)
                if resolved_scheme in COLOR_SCHEME_PRESETS:
                    return resolved_scheme
        
        return None
    except Exception as e:
        # If detection fails, just return None (don't break the script)
        return None


def main():
    # Get the project root directory (parent of test_manim)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Version file path
    version_file = script_dir / ".version"
    
    # Read and increment version
    if version_file.exists():
        with open(version_file, 'r') as f:
            version = int(f.read().strip()) + 1
    else:
        version = 1
    
    # Write new version
    with open(version_file, 'w') as f:
        f.write(str(version))
    
    print(f"Version: v{version}")
    
    # Scene file and class name
    scene_file = "test_manim/lamarckian_functions.py"
    scene_file_path = project_root / scene_file
    scene_class = "TestPureLamarckianFunction"
    output_name = f"MyRetro3D_v{version}"
    
    # Detect and print color scheme if present
    color_scheme = detect_color_scheme(scene_file_path)
    if color_scheme:
        print(f"Color scheme: {color_scheme}")
    else:
        print("Color scheme: default (not specified)")
    
    # Run manim command
    manim_cmd = [
        "manim",
        "-s",
        "--disable_caching",
        scene_file,
        scene_class,
        "-o", output_name
    ]
    
    print(f"Running: {' '.join(manim_cmd)}")
    result = subprocess.run(manim_cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"Error: Manim command failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Check for output image and open it
    # Manim creates output in media/images/{scene_file_name}/{output_name}.png
    scene_file_name = Path(scene_file).stem  # Gets 'lamarckian_functions' without .py
    output_path = project_root / "media" / "images" / scene_file_name / f"{output_name}.png"
    
    # Wait a moment for file system to sync
    time.sleep(0.5)
    
    if output_path.exists():
        print(f"Opening: {output_path}")
        # Use xdg-open on Linux, start on Windows, open on macOS
        try:
            if sys.platform == "linux":
                subprocess.run(["xdg-open", str(output_path)], check=True)
            elif sys.platform == "win32":
                subprocess.run(["start", str(output_path)], shell=True, check=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_path)], check=True)
            else:
                print(f"Output image available at: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error opening file: {e}")
            print(f"Output saved to: {output_path}")
        except Exception as e:
            print(f"Could not open file automatically: {e}")
            print(f"Output saved to: {output_path}")
    else:
        print(f"Warning: Output image not found at {output_path}")
        # Try to find alternative paths
        alt_paths = [
            project_root / "media" / "images" / "lamarckian_functions" / f"{output_name}.png",
            project_root / "media" / "images" / scene_file_name / f"{output_name}.png"
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"Found at alternative path: {alt_path}")
                try:
                    if sys.platform == "linux":
                        subprocess.run(["xdg-open", str(alt_path)], check=True)
                    elif sys.platform == "win32":
                        subprocess.run(["start", str(alt_path)], shell=True, check=True)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", str(alt_path)], check=True)
                except Exception as e:
                    print(f"Could not open: {e}")
                break
    
    response = input("\nRun again? [y/N]: ").strip().lower()
    return response in ('y', 'yes')


if __name__ == "__main__":
    while main():
        pass
