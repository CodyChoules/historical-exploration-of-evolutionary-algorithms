#!/usr/bin/env python3
"""
Python script to run retro_tester_2.py scene with version management.

This script replicates the bash command functionality:
- Increments version number
- Runs manim with --disable_caching
- Opens the output image if it exists

Usage:
    python test_manim/run_retro_tester_2.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Track if we've already opened a viewer this session (reuse same window on "Run again")
_viewer_opened = False


def _find_venv_python(project_root: Path) -> str | None:
    """
    Find Python that can run manim. On Linux/WSL, prefer Linux-style venv (bin/python)
    to avoid path/regex issues when Windows Python sees \\wsl.localhost\\Ubuntu paths.
    """
    is_linux = sys.platform in ("linux", "darwin")
    for venv_name in ("manim.venv", "manim-linux.venv", ".venv", "venv"):
        venv = project_root / venv_name
        if (venv / "bin" / "python").exists():
            return str(venv / "bin" / "python")
        if not is_linux and (venv / "Scripts" / "python.exe").exists():
            return str(venv / "Scripts" / "python.exe")
    if is_linux:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "manim", "--version"],
                capture_output=True,
                timeout=5,
                cwd=project_root,
            )
            if r.returncode == 0:
                return sys.executable
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _path_for_manim(path: Path) -> str:
    """Normalize path to forward slashes for manim. Avoids re.error on WSL when path contains \\Ubuntu etc."""
    return str(path.resolve()).replace("\\", "/")


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
    scene_file = "test_manim/retro_tester_2.py"
    scene_file_path = project_root / scene_file
    scene_class = "SimpleRetroScene"
    output_name = f"MyRetro3D_v{version}"
    
    # Detect and print color scheme if present
    color_scheme = detect_color_scheme(scene_file_path)
    if color_scheme:
        print(f"Color scheme: {color_scheme}")
    else:
        print("Color scheme: default (not specified)")
    
    # Run manim: use venv python -m manim (avoids PATH), pass path with forward slashes (avoids re.error on WSL)
    venv_python = _find_venv_python(project_root)
    if venv_python is None and sys.platform in ("linux", "darwin"):
        print("No manim Python found. On WSL/Linux: sudo apt install python3.12-venv, then:")
        print("  python3 -m venv manim-linux.venv && manim-linux.venv/bin/pip install -r requirements.txt")
        sys.exit(1)
    scene_file_for_manim = _path_for_manim(scene_file_path)
    if venv_python:
        manim_cmd = [
            venv_python,
            "-m", "manim",
            "-s",
            "--disable_caching",
            scene_file_for_manim,
            scene_class,
            "-o", output_name,
        ]
    else:
        manim_cmd = [
            "manim",
            "-s",
            "--disable_caching",
            scene_file_for_manim,
            scene_class,
            "-o", output_name,
        ]

    print(f"Running: {' '.join(manim_cmd)}")
    result = subprocess.run(manim_cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"Error: Manim command failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Stable path for "current" image so the same viewer window can refresh instead of opening new ones
    images_dir = project_root / "media" / "images" / "retro_tester_2"
    output_path = images_dir / f"{output_name}.png"
    current_path = images_dir / "MyRetro3D_current.png"

    if output_path.exists():
        # Always update the stable "current" file so an existing viewer can refresh
        shutil.copy2(output_path, current_path)

        global _viewer_opened
        if not _viewer_opened:
            _viewer_opened = True
            print(f"Opening: {current_path}")
            # Open in background so the script doesn't block the CLI
            kwargs = {}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", str(current_path)], **kwargs)
            elif sys.platform == "win32":
                subprocess.Popen(
                    ["start", "", str(current_path)],
                    shell=True,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(current_path)], **kwargs)
            else:
                print(f"Output image available at: {current_path}")
        else:
            print(f"Updated: {current_path} (viewer should refresh)")
    else:
        print(f"Warning: Output image not found at {output_path}")
    
    response = input("\nRun again? [y/N]: ").strip().lower()
    return response in ('y', 'yes')


if __name__ == "__main__":
    while main():
        pass
