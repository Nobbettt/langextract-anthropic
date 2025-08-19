#!/usr/bin/env python3
"""Version bumping utility for langextract-anthropic package.

Usage:
    python scripts/bump_version.py patch   # 0.1.0 -> 0.1.1
    python scripts/bump_version.py minor   # 0.1.0 -> 0.2.0
    python scripts/bump_version.py major   # 0.1.0 -> 1.0.0
"""

import re
import sys
from pathlib import Path


def get_current_version(init_file: Path) -> str:
    """Extract current version from __init__.py file."""
    content = init_file.read_text()
    version_match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not version_match:
        raise ValueError("Could not find __version__ in __init__.py")
    return version_match.group(1)


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on semantic versioning."""
    parts = current.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current}")

    major, minor, patch = map(int, parts)

    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Invalid bump type: {bump_type}. Use: patch, minor, major")

    return f"{major}.{minor}.{patch}"


def update_file_version(file_path: Path, old_version: str, new_version: str):
    """Update version in a file."""
    content = file_path.read_text()

    if file_path.name == "__init__.py":
        # Update __version__ = "x.y.z"
        content = re.sub(
            r'__version__ = ["\'][^"\']+["\']',
            f'__version__ = "{new_version}"',
            content,
        )
    elif file_path.name == "pyproject.toml":
        # Update version = "x.y.z" in [project] section only
        content = re.sub(
            r'(\[project\].*?\n)version = ["\'][^"\']+["\']',
            rf'\1version = "{new_version}"',
            content,
            flags=re.DOTALL,
        )

    file_path.write_text(content)
    print(f"✅ Updated {file_path.name}: {old_version} -> {new_version}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["patch", "minor", "major"]:
        print(__doc__)
        sys.exit(1)

    bump_type = sys.argv[1]

    # File paths
    root_dir = Path(__file__).parent.parent
    init_file = root_dir / "langextract_anthropic" / "__init__.py"
    pyproject_file = root_dir / "pyproject.toml"

    # Get current version
    current_version = get_current_version(init_file)
    new_version = bump_version(current_version, bump_type)

    print(f"🔄 Bumping version: {current_version} -> {new_version}")

    # Update files
    update_file_version(init_file, current_version, new_version)
    update_file_version(pyproject_file, current_version, new_version)

    print("\n✅ Version bump complete!")
    print("\nNext steps:")
    print(f"1. Update CHANGELOG.md with changes for v{new_version}")
    print(
        f"2. Commit changes: git add . && git commit -m 'bump: version {current_version} -> {new_version}'"
    )
    print(f"3. Create tag: git tag v{new_version}")
    print("4. Build package: uv build")
    print("5. Publish: uv publish")


if __name__ == "__main__":
    main()
