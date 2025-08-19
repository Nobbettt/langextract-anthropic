#!/usr/bin/env python3
"""Comprehensive test runner for langextract-anthropic package."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n🔧 {description}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, shell=True, check=False)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("🧪 LangExtract Anthropic - Test Suite")
    print("=" * 60)
    results = []

    # Detect uv availability
    has_uv = shutil.which("uv") is not None
    py = sys.executable or "python3"

    # 0. Ensure dev dependencies are installed (best-effort)
    if has_uv:
        results.append(run_command("uv sync --extra dev", "Install dev dependencies"))
    else:
        print("\n⚠️  'uv' not found. Installing dev extras with pip...")
        results.append(
            run_command(
                f"{py} -m pip install -e .[dev]", "Install dev dependencies via pip"
            )
        )

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Track test results (continuing after dependency install)

    # 1. Code formatting checks
    fmt_black = "uv run black --check ." if has_uv else "black --check ."
    fmt_isort = "uv run isort --check-only ." if has_uv else "isort --check-only ."
    results.append(run_command(fmt_black, "Code formatting (black)"))
    results.append(run_command(fmt_isort, "Import sorting (isort)"))

    # 2. Linting
    lint_cmd = "uv run ruff check ." if has_uv else "ruff check ."
    results.append(run_command(lint_cmd, "Code linting (ruff)"))

    # 3. Type checking
    mypy_cmd = (
        "uv run mypy langextract_anthropic"
        if has_uv
        else f"{py} -m mypy langextract_anthropic"
    )
    results.append(run_command(mypy_cmd, "Type checking (mypy)"))

    # 4. Unit tests
    unit_cmd = (
        "uv run pytest -m unit tests/ -v"
        if has_uv
        else f"{py} -m pytest -m unit tests/ -v"
    )
    results.append(run_command(unit_cmd, "Unit tests"))

    # 5. Parameter filtering tests are included in unit tests via markers; no separate run

    # 6. Integration tests (only if credentials are available)
    has_credentials = bool(os.getenv("ANTHROPIC_API_KEY"))

    if has_credentials:
        print("\n🔐 Anthropic API key found - running integration tests")
        integ_cmd = (
            "uv run python tests/test_anthropic_parameters.py"
            if has_uv
            else f"{py} tests/test_anthropic_parameters.py"
        )
        results.append(run_command(integ_cmd, "Anthropic API parameter tests"))
    else:
        print("\n⚠️  No Anthropic API key found - skipping integration tests")
        print("   Set ANTHROPIC_API_KEY environment variable to run them")

    # 7. Coverage report
    cov_cmd = (
        "uv run pytest tests/ --cov=langextract_anthropic --cov-report=term-missing --cov-report=html"
        if has_uv
        else f"{py} -m pytest tests/ --cov=langextract_anthropic --cov-report=term-missing --cov-report=html"
    )
    results.append(run_command(cov_cmd, "Coverage analysis"))

    # 8. Package build test
    build_cmd = (
        "uv run python scripts/check_build.py"
        if has_uv
        else f"{py} scripts/check_build.py"
    )
    results.append(run_command(build_cmd, "Package build validation"))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for _i, (description, passed_test) in enumerate(
        [
            ("Code formatting (black)", results[0]),
            ("Import sorting (isort)", results[1]),
            ("Code linting (ruff)", results[2]),
            ("Type checking (mypy)", results[3]),
            ("Unit tests", results[4]),
            ("Parameter filtering tests", results[5]),
            ("Integration tests", results[6] if has_credentials else None),
            ("Coverage analysis", results[6 if has_credentials else 5]),
            ("Package build validation", results[7 if has_credentials else 6]),
        ],
        1,
    ):
        if passed_test is None:
            print(f"⏭️  {description} - SKIPPED")
        elif passed_test:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Package is ready for release.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please fix issues before release.")
        sys.exit(1)


if __name__ == "__main__":
    main()
