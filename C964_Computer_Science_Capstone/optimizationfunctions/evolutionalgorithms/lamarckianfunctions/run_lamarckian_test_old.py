#!/usr/bin/env python3
"""
Script to run the Lamarckian Functions test scene from the package.

Handles version incrementing, running the Manim scene, and opening the output.

Usage (from project root):
    python lamarckian_functions/run_lamarckian_test.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_current_viewer_opened = False
_previous_viewer_opened = False


def detect_color_scheme(scene_file_path, project_root):
    """Detect color scheme preset from scene file. Uses retro_manim_graph for presets."""
    try:
        sys.path.insert(0, str(project_root))
        from visualizationtool.retrograph.retro_configuration import (
            COLOR_SCHEME_PRESETS,
            COLOR_SCHEME_ALIASES,
        )
        with open(scene_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        patterns = [
            r"['\"]color_scheme['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            r"['\"]COLOR_SCHEME['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                scheme = matches[0].strip()
                resolved = COLOR_SCHEME_ALIASES.get(scheme, scheme)
                if resolved in COLOR_SCHEME_PRESETS:
                    return resolved
        return None
    except Exception:
        return None


def open_image(path):
    """Open path in default viewer (background, non-blocking)."""
    kwargs = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    if sys.platform == "linux":
        subprocess.Popen(["xdg-open", str(path)], **kwargs)
    elif sys.platform == "win32":
        subprocess.Popen(
            ["start", "", str(path)],
            shell=True,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)], **kwargs)


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    version_file = script_dir / ".version"
    if version_file.exists():
        with open(version_file, 'r') as f:
            version = int(f.read().strip()) + 1
    else:
        version = 1
    with open(version_file, 'w') as f:
        f.write(str(version))

    print(f"Version: v{version}")

    scene_file = "lamarckian_functions/core.py"
    scene_file_path = project_root / scene_file
    scene_class = "TestPureLamarckianFunction"
    output_name = f"MyRetro3D_v{version}"

    color_scheme = detect_color_scheme(scene_file_path, project_root)
    if color_scheme:
        print(f"Color scheme: {color_scheme}")
    else:
        print("Color scheme: default (not specified)")

    manim_cmd = [
        "manim",
        "-s",
        "--disable_caching",
        scene_file,
        scene_class,
        "-o", output_name,
    ]
    print(f"Running: {' '.join(manim_cmd)}")
    result = subprocess.run(manim_cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"Error: Manim command failed with exit code {result.returncode}")
        sys.exit(1)

    images_dir = project_root / "media" / "images" / scene_file_path.stem
    output_path = images_dir / f"{output_name}.png"
    current_path = images_dir / "MyRetro3D_current.png"
    previous_path = images_dir / "MyRetro3D_previous.png"

    if output_path.exists():
        if current_path.exists():
            shutil.copy2(current_path, previous_path)
        shutil.copy2(output_path, current_path)

        global _current_viewer_opened, _previous_viewer_opened
        if not _current_viewer_opened:
            _current_viewer_opened = True
            print(f"Opening current: {current_path}")
            open_image(current_path)
        else:
            print(f"Updated current: {current_path} (viewer should refresh)")

        if previous_path.exists():
            if not _previous_viewer_opened:
                _previous_viewer_opened = True
                print(f"Opening previous: {previous_path}")
                open_image(previous_path)
            else:
                print(f"Updated previous: {previous_path} (viewer should refresh)")
    else:
        print(f"Warning: Output image not found at {output_path}")

    response = input("\nRun again? [y/N]: ").strip().lower()
    return response in ('y', 'yes')


if __name__ == "__main__":
    while main():
        pass
