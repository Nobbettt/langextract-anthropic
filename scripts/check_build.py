#!/usr/bin/env python3
"""Build validation script for langextract-anthropic package."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle output."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        if result.stdout.strip():
            print(f"   ✓ {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e.stderr.strip()}")
        return False


def main():
    """Run build validation checks."""
    print("🚀 LangExtract Anthropic - Build Validation")
    print("=" * 50)

    # Check UV is available
    if not run_command("uv --version", "Checking UV installation"):
        print("❌ UV is not installed. Please install it first.")
        sys.exit(1)

    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    for path in ["dist/", "build/", "*.egg-info/"]:
        if Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)

    # Build package
    print("\n📦 Building package...")
    if not run_command("uv build", "Building distribution packages"):
        print("❌ Build failed!")
        sys.exit(1)

    # Check build artifacts
    dist_path = Path("dist")
    if not dist_path.exists():
        print("❌ No dist/ directory created!")
        sys.exit(1)

    artifacts = list(dist_path.glob("*"))
    if not artifacts:
        print("❌ No build artifacts found!")
        sys.exit(1)

    print(f"   ✓ Found {len(artifacts)} build artifacts:")
    for artifact in artifacts:
        print(f"     - {artifact.name}")

    # Validate wheel content
    wheel_files = list(dist_path.glob("*.whl"))
    if wheel_files:
        print("\n🔍 Validating wheel content...")
        wheel_file = wheel_files[0]
        py = sys.executable or "python"
        if run_command(
            f"{py} -m zipfile -l {wheel_file}", "Listing wheel contents", check=False
        ):
            print("   ✓ Wheel file is valid")

    # Test installation in temporary environment
    print("\n🧪 Testing installation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_env"

        # Create test environment
        if not run_command(f"uv venv {venv_path}", "Creating test environment"):
            print("❌ Failed to create test environment!")
            sys.exit(1)

        # Install package
        wheel_file = wheel_files[0] if wheel_files else artifacts[0]
        install_cmd = f"uv pip install --python {venv_path}/bin/python {wheel_file}"
        if not run_command(install_cmd, "Installing package in test environment"):
            print("❌ Failed to install package!")
            sys.exit(1)

        # Test import
        test_import_cmd = f"{venv_path}/bin/python -c 'import langextract_anthropic; print(f\"Version: {{langextract_anthropic.__version__}}\")'"
        if not run_command(test_import_cmd, "Testing package import"):
            print("❌ Failed to import package!")
            sys.exit(1)

    print("\n✅ All build validation checks passed!")
    print("📋 Summary:")
    print(f"   - Build artifacts: {len(artifacts)}")
    print(f"   - Wheel files: {len(wheel_files)}")
    print("   - Installation test: ✓")
    print("   - Import test: ✓")


if __name__ == "__main__":
    main()
