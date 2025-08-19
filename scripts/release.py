#!/usr/bin/env python3
"""Release automation script for langextract-azureopenai package."""

import subprocess
import sys
from datetime import datetime
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
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e.stderr.strip()}")
        if check:
            sys.exit(1)
        return e


def get_current_version():
    """Get current version from pyproject.toml."""
    try:
        result = subprocess.run(
            [
                "python",
                "-c",
                "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        # Fallback to reading manually
        with open("pyproject.toml") as f:
            for line in f:
                if line.strip().startswith("version ="):
                    return line.split("=")[1].strip().strip('"')
    return "unknown"


def confirm_action(message):
    """Ask for user confirmation."""
    response = input(f"{message} (y/N): ").strip().lower()
    return response in ['y', 'yes']


def main():
    """Run release process."""
    print("🚀 LangExtract Azure OpenAI - Release Process")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Get current version
    current_version = get_current_version()
    print(f"📋 Current version: {current_version}")

    # Check git status
    print("\n🔍 Checking git status...")
    git_status = run_command(
        "git status --porcelain", "Getting git status", check=False
    )
    if git_status.stdout.strip():
        print("⚠️  Warning: You have uncommitted changes:")
        print(git_status.stdout)
        if not confirm_action("Continue with uncommitted changes?"):
            print("❌ Aborting release.")
            sys.exit(1)

    # Run test suite
    print("\n🧪 Running test suite...")
    test_result = run_command(
        "python scripts/run_tests.py", "Running comprehensive tests", check=False
    )
    if test_result.returncode != 0:
        print("❌ Tests failed! Please fix issues before release.")
        if not confirm_action("Continue despite test failures?"):
            sys.exit(1)

    # Ask for version bump
    print(f"\n📈 Current version: {current_version}")
    print("Version bump options:")
    print("  1. patch (e.g., 1.0.0 -> 1.0.1)")
    print("  2. minor (e.g., 1.0.0 -> 1.1.0)")
    print("  3. major (e.g., 1.0.0 -> 2.0.0)")
    print("  4. skip version bump")

    choice = input("Select version bump (1-4): ").strip()

    if choice in ['1', '2', '3']:
        bump_type = ['patch', 'minor', 'major'][int(choice) - 1]
        print(f"\n📝 Bumping version ({bump_type})...")
        run_command(
            f"python scripts/bump_version.py {bump_type}",
            f"Bumping {bump_type} version",
        )
        new_version = get_current_version()
        print(f"   ✓ New version: {new_version}")
    else:
        new_version = current_version
        print("   ⏭️  Skipping version bump")

    # Build package
    print("\n📦 Building package...")
    run_command("rm -rf dist/ build/ *.egg-info/", "Cleaning previous builds")
    run_command("uv build", "Building distribution packages")

    # Validate build
    print("\n🔍 Validating build...")
    run_command("python scripts/check_build.py", "Validating package build")

    # Show what will be released
    print("\n📋 Release Summary:")
    print(f"   Version: {new_version}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # List build artifacts
    dist_files = list(Path("dist").glob("*"))
    print(f"   Artifacts: {len(dist_files)} files")
    for file in dist_files:
        print(f"     - {file.name}")

    # Confirm release
    if not confirm_action(f"\n🚀 Ready to release version {new_version}?"):
        print("❌ Release cancelled.")
        sys.exit(1)

    # Commit version changes (if any)
    if choice in ['1', '2', '3']:
        print("\n📝 Committing version changes...")
        run_command(
            "git add pyproject.toml langextract_anthropic/__init__.py",
            "Staging version files",
        )
        run_command(
            f'git commit -m "Bump version to {new_version}"', "Committing version bump"
        )

    # Create git tag
    print(f"\n🏷️  Creating git tag v{new_version}...")
    run_command(
        f'git tag -a v{new_version} -m "Release version {new_version}"',
        "Creating git tag",
    )

    # Push to repository
    if confirm_action("Push changes and tags to repository?"):
        print("\n📤 Pushing to repository...")
        run_command("git push", "Pushing commits")
        run_command("git push --tags", "Pushing tags")

    # Publish to PyPI
    if confirm_action("Publish to PyPI?"):
        print("\n🌍 Publishing to PyPI...")
        token = input("Enter PyPI API token (or press Enter to skip): ").strip()
        if token:
            run_command(f"uv publish --token {token}", "Publishing to PyPI")
        else:
            print("   ⏭️  Skipping PyPI publication")
            print("   💡 You can publish later with: uv publish --token YOUR_TOKEN")

    print(f"\n🎉 Release {new_version} completed successfully!")
    print("\n📋 Next steps:")
    print("   - Update GitHub release notes")
    print("   - Announce release in relevant channels")
    print("   - Update documentation if needed")


if __name__ == "__main__":
    main()
