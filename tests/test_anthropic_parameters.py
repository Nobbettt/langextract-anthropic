#!/usr/bin/env python3
"""Comprehensive parameter testing for Anthropic provider.

This script tests each parameter in _ANTHROPIC_CONFIG_KEYS to ensure
they work correctly with actual Anthropic API calls.
"""

import json
import os
import sys
from typing import Any

import langextract as lx

from langextract_anthropic import AnthropicLanguageModel

# Test parameters with safe values
TEST_PARAMETERS = {
    'temperature': [0.0, 0.3, 0.7, 1.0],
    'max_tokens': [10, 50, 100, 1000],
    'top_p': [0.1, 0.5, 0.9, 1.0],
    'top_k': [1, 10, 50, 200],
    'stop_sequences': [["END"], ["\n", "."], ["STOP", "DONE"]],
    'metadata': [
        {"user": "test-user"},
        {"request_id": "test-123", "session": "demo"},
        {}  # Empty metadata
    ],
    'system': [
        "You are a helpful assistant.",
        "Answer briefly and concisely.",
        "You are a test assistant for parameter validation."
    ],
    'service_tier': [
        "auto",
        "standard_only"
    ],
}

# Parameters to skip (unsupported)
SKIP_PARAMETERS = {'stream', 'tools', 'tool_choice'}


def check_environment() -> str:
    """Check if required environment variables are set."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Missing required environment variable:")
        print("  ANTHROPIC_API_KEY")
        print("\nSet your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)
    return api_key


def test_parameter(param_name: str, param_value: Any, api_key: str) -> dict[str, Any]:
    """Test a single parameter with the Anthropic API."""
    result = {
        'parameter': param_name,
        'value': param_value,
        'success': False,
        'error': None,
        'response_length': 0,
        'api_called': False
    }

    try:
        # Create provider with the parameter
        kwargs = {param_name: param_value}
        provider = AnthropicLanguageModel(
            model_id="anthropic-claude-3-5-sonnet-latest",
            api_key=api_key,
            **kwargs
        )

        # Test with a simple prompt
        prompts = [f"Say 'Parameter {param_name} test successful' and nothing else."]

        # Make API call
        responses = list(provider.infer(prompts))
        result['api_called'] = True

        if responses and responses[0] and responses[0][0]:
            output = responses[0][0].output
            result['response_length'] = len(output) if output else 0
            result['success'] = True
            print(f"   ✓ {param_name}={param_value} -> Response: {output[:100]}...")
        else:
            result['error'] = "No response received"
            print(f"   ✗ {param_name}={param_value} -> No response")

    except Exception as e:
        result['error'] = str(e)
        print(f"   ✗ {param_name}={param_value} -> Error: {e}")

    return result


def run_parameter_tests(api_key: str) -> dict[str, list[dict[str, Any]]]:
    """Run all parameter tests."""
    all_results = {}

    print("Testing Anthropic Provider Parameters")
    print("=" * 50)

    for param_name, test_values in TEST_PARAMETERS.items():
        if param_name in SKIP_PARAMETERS:
            print(f"\nSkipping {param_name} (unsupported)")
            continue

        print(f"\nTesting {param_name}:")
        param_results = []

        for value in test_values:
            result = test_parameter(param_name, value, api_key)
            param_results.append(result)

        all_results[param_name] = param_results

    return all_results


def generate_report(results: dict[str, list[dict[str, Any]]]) -> None:
    """Generate a detailed report of test results."""

    # Console summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    total_tests = sum(len(param_results) for param_results in results.values())
    successful_tests = sum(
        sum(1 for result in param_results if result['success'])
        for param_results in results.values()
    )

    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")

    print("\nParameter-wise breakdown:")
    for param_name, param_results in results.items():
        successes = sum(1 for r in param_results if r['success'])
        total = len(param_results)
        print(f"  {param_name}: {successes}/{total} ({successes/total*100:.0f}%)")

    # Save detailed JSON report
    report_file = "anthropic_results.jsonl"
    with open(report_file, 'w') as f:
        for _param_name, param_results in results.items():
            for result in param_results:
                f.write(json.dumps(result) + '\n')

    print(f"\nDetailed results saved to: {report_file}")

    # Generate HTML visualization
    html_content = generate_html_report(results)
    html_file = "anthropic_visualization.html"
    with open(html_file, 'w') as f:
        f.write(html_content)

    print(f"HTML visualization saved to: {html_file}")


def generate_html_report(results: dict[str, list[dict[str, Any]]]) -> str:
    """Generate an HTML visualization of test results."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Anthropic Provider Parameter Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .success { color: green; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .parameter-section { margin: 30px 0; }
        .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Anthropic Provider Parameter Test Results</h1>

    <div class="summary">
        <h2>Summary</h2>
"""

    total_tests = sum(len(param_results) for param_results in results.values())
    successful_tests = sum(
        sum(1 for result in param_results if result['success'])
        for param_results in results.values()
    )

    html += f"""
        <p><strong>Total tests:</strong> {total_tests}</p>
        <p><strong>Successful:</strong> {successful_tests}</p>
        <p><strong>Success rate:</strong> {successful_tests/total_tests*100:.1f}%</p>
    </div>
    """

    for param_name, param_results in results.items():
        html += f"""
    <div class="parameter-section">
        <h2>Parameter: {param_name}</h2>
        <table>
            <tr>
                <th>Value</th>
                <th>Status</th>
                <th>Response Length</th>
                <th>Error</th>
            </tr>
"""

        for result in param_results:
            status_class = "success" if result['success'] else "error"
            status_text = "✓ Success" if result['success'] else "✗ Failed"
            error_text = result['error'] if result['error'] else ""

            html += f"""
            <tr>
                <td><code>{result['value']}</code></td>
                <td class="{status_class}">{status_text}</td>
                <td>{result['response_length']}</td>
                <td>{error_text}</td>
            </tr>
"""

        html += """        </table>
    </div>
"""

    html += """
</body>
</html>"""

    return html


def main():
    """Main test execution."""
    print("Anthropic Provider Parameter Testing")
    print("This script tests all supported parameters with real API calls.")
    print("Note: This will consume API credits!")

    # Check environment
    api_key = check_environment()

    # Run tests
    results = run_parameter_tests(api_key)

    # Generate report
    generate_report(results)


if __name__ == '__main__':
    main()
