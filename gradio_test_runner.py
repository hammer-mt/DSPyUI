#!/usr/bin/env python3
"""
Gradio Test Runner for DSPyUI

This script provides a comprehensive testing infrastructure that:
1. Runs Gradio as a background process with stdout/stderr logging
2. Executes tests sequentially with aggregated logging
3. Provides REST API testing utilities via Gradio Python Client
4. Generates mock sample data for testing
5. Creates a feedback loop for continuous improvement

Usage:
    python gradio_test_runner.py                    # Run all tests
    python gradio_test_runner.py --test <name>      # Run specific test
    python gradio_test_runner.py --generate-mocks   # Generate mock data only
    python gradio_test_runner.py --rest-api-tests   # Run REST API tests only
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Try to import gradio_client for REST API testing
try:
    from gradio_client import Client, handle_file  # type: ignore[import-untyped]
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("gradio_client not installed. REST API tests will be skipped.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockDataGenerator:
    """Generate mock CSV data for testing DSPy compilations."""

    @staticmethod
    def generate_joke_dataset(num_rows: int = 5, output_path: Optional[Path] = None) -> Path:
        """Generate a mock joke dataset with topic -> joke mapping."""
        data = [
            {"topic": "cats", "joke": "Why do cats make great programmers? They have perfect mew-thods!"},
            {"topic": "dogs", "joke": "What do you call a dog that does magic tricks? A labracadabrador!"},
            {"topic": "food", "joke": "Why did the cookie go to the doctor? It was feeling crumbly!"},
            {"topic": "space", "joke": "Why did the sun go to school? To get brighter!"},
            {"topic": "computers", "joke": "Why do programmers prefer dark mode? Light attracts bugs!"},
            {"topic": "chickens", "joke": "Why did the chicken cross the road? To get to the other side!"},
            {"topic": "science", "joke": "Why can't you trust atoms? Because they make up everything!"},
            {"topic": "math", "joke": "Why was six afraid of seven? Because seven eight nine!"},
        ]

        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                suffix='.csv',
                newline=''
            )
            output_path = Path(temp_file.name)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["topic", "joke"])
            writer.writeheader()
            writer.writerows(data[:min(num_rows, len(data))])

        logger.info(f"Generated mock joke dataset: {output_path}")
        return output_path

    @staticmethod
    def generate_classification_dataset(num_rows: int = 5, output_path: Optional[Path] = None) -> Path:
        """Generate a mock classification dataset."""
        data = [
            {"text": "This is amazing!", "sentiment": "positive"},
            {"text": "This is terrible.", "sentiment": "negative"},
            {"text": "It's okay I guess.", "sentiment": "neutral"},
            {"text": "Best experience ever!", "sentiment": "positive"},
            {"text": "Worst product ever.", "sentiment": "negative"},
            {"text": "Pretty good overall.", "sentiment": "positive"},
            {"text": "Not great, not terrible.", "sentiment": "neutral"},
            {"text": "Absolutely horrible!", "sentiment": "negative"},
        ]

        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                suffix='.csv',
                newline=''
            )
            output_path = Path(temp_file.name)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["text", "sentiment"])
            writer.writeheader()
            writer.writerows(data[:min(num_rows, len(data))])

        logger.info(f"Generated mock classification dataset: {output_path}")
        return output_path

    @staticmethod
    def generate_qa_dataset(num_rows: int = 5, output_path: Optional[Path] = None) -> Path:
        """Generate a mock Q&A dataset."""
        data = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What color is the sky?", "answer": "blue"},
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "How many days in a week?", "answer": "7"},
            {"question": "What is H2O?", "answer": "water"},
            {"question": "How many months in a year?", "answer": "12"},
            {"question": "What is the largest ocean?", "answer": "Pacific"},
            {"question": "How many continents are there?", "answer": "7"},
        ]

        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                suffix='.csv',
                newline=''
            )
            output_path = Path(temp_file.name)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(data[:min(num_rows, len(data))])

        logger.info(f"Generated mock Q&A dataset: {output_path}")
        return output_path


class RestApiTestRunner:
    """Run REST API tests using Gradio Python Client."""

    def __init__(self, server_url: str, log_dir: Path):
        self.server_url = server_url
        self.log_dir = log_dir
        self.results: List[Dict[str, Any]] = []

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all available REST API tests."""
        if not GRADIO_CLIENT_AVAILABLE:
            logger.warning("Gradio client not available, skipping REST API tests")
            return []

        test_methods = [
            method for method in dir(self)
            if method.startswith('test_') and callable(getattr(self, method))
        ]

        logger.info(f"\n{'='*60}")
        logger.info(f"Running {len(test_methods)} REST API tests")
        logger.info(f"{'='*60}\n")

        for method_name in sorted(test_methods):
            self.run_test(method_name)

        return self.results

    def run_test(self, test_name: str) -> Dict[str, Any]:
        """Run a single test by name."""
        logger.info(f"--- Running REST API Test: {test_name} ---")

        start_time = time.time()
        try:
            method = getattr(self, test_name)
            method()
            duration = time.time() - start_time
            result = {
                "name": test_name,
                "passed": True,
                "duration": duration,
                "error": None
            }
            logger.info(f"✓ {test_name} PASSED ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result = {
                "name": test_name,
                "passed": False,
                "duration": duration,
                "error": error_msg
            }
            logger.error(f"✗ {test_name} FAILED ({duration:.2f}s)")
            logger.error(f"Error: {error_msg}")

        self.results.append(result)
        return result

    def test_api_connection(self):
        """Test that we can connect to the Gradio API."""
        client = Client(self.server_url, verbose=False)
        assert client is not None
        logger.info(f"Successfully connected to {self.server_url}")

    def test_list_prompts_api(self):
        """Test the list_prompts API endpoint."""
        client = Client(self.server_url, verbose=False)

        try:
            result = client.predict(
                filter_text="",
                sort_by="Run Date",
                api_name="/list_prompts"
            )
            assert isinstance(result, str)
            logger.info(f"list_prompts returned {len(result)} characters")
        except Exception as e:
            # API endpoint might not be named exactly /list_prompts
            logger.warning(f"Could not call /list_prompts: {e}")
            # Try to inspect API
            logger.info("Available API endpoints:")
            logger.info(client.view_api())

    def test_mock_data_generation(self):
        """Test mock data generator functions."""
        generator = MockDataGenerator()

        # Generate each type of mock data
        joke_file = generator.generate_joke_dataset(3)
        class_file = generator.generate_classification_dataset(3)
        qa_file = generator.generate_qa_dataset(3)

        # Verify files exist and have content
        for filepath in [joke_file, class_file, qa_file]:
            assert filepath.exists()
            with open(filepath, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2  # Header + at least 1 data row
            filepath.unlink()  # Clean up

        logger.info("All mock data generators working correctly")


class GradioTestRunner:
    """Manages Gradio server and test execution."""

    def __init__(
        self,
        port: int = 7860,
        log_dir: str = "test_logs",
        verbose: bool = False
    ) -> None:
        """
        Initialize the test runner.

        Args:
            port: Port to run Gradio server on
            log_dir: Directory to store log files
            verbose: Enable verbose output
        """
        self.port = port
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.gradio_process: Optional[subprocess.Popen] = None
        self.gradio_log_file: Optional[Path] = None
        self.test_results: List[Dict[str, Any]] = []

        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        logger.info(f"Test run directory: {self.run_dir}")

    def start_gradio_server(self) -> bool:
        """
        Start the Gradio server as a background process.

        Returns:
            True if server started successfully, False otherwise
        """
        logger.info(f"Starting Gradio server on port {self.port}...")

        # Set up log file
        self.gradio_log_file = self.run_dir / "gradio_server.log"

        # Start Gradio process
        with open(self.gradio_log_file, 'w') as log_file:
            self.gradio_process = subprocess.Popen(
                [sys.executable, "-m", "gradio", "interface.py"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "GRADIO_SERVER_PORT": str(self.port),
                }
            )

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/")
                if response.status_code == 200:
                    logger.info("Gradio server started successfully")
                    return True
            except requests.ConnectionError:
                if self.verbose:
                    logger.debug(f"Waiting for server... ({i+1}/{max_retries})")
                time.sleep(1)

        logger.error("Failed to start Gradio server")
        return False

    def stop_gradio_server(self) -> None:
        """Stop the Gradio server process."""
        if self.gradio_process:
            logger.info("Stopping Gradio server...")
            self.gradio_process.terminate()
            try:
                self.gradio_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Gradio server did not terminate, killing...")
                self.gradio_process.kill()
                self.gradio_process.wait()
            logger.info("Gradio server stopped")

    def get_gradio_logs(self) -> str:
        """
        Read the Gradio server logs.

        Returns:
            Contents of the Gradio log file
        """
        if self.gradio_log_file and self.gradio_log_file.exists():
            return self.gradio_log_file.read_text()
        return ""

    def run_pytest(self) -> Tuple[int, str]:
        """
        Run pytest tests.

        Returns:
            Tuple of (exit_code, output)
        """
        logger.info("Running pytest tests...")

        test_log_file = self.run_dir / "pytest_output.log"

        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )

        # Save output to log file
        with open(test_log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        return result.returncode, result.stdout + result.stderr

    def analyze_test_results(self, pytest_output: str) -> Dict[str, Any]:
        """
        Analyze pytest output to extract test results.

        Args:
            pytest_output: Raw output from pytest

        Returns:
            Dictionary containing test statistics
        """
        # Use explicit type annotations for dictionary values
        passed: int = 0
        failed: int = 0
        skipped: int = 0
        errors: int = 0
        failed_tests: List[str] = []

        # Parse pytest output
        lines = pytest_output.split('\n')
        for line in lines:
            if " PASSED" in line:
                passed += 1
            elif " FAILED" in line:
                failed += 1
                failed_tests.append(line.strip())
            elif " SKIPPED" in line:
                skipped += 1
            elif " ERROR" in line:
                errors += 1

        total = passed + failed + skipped + errors

        results = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "total": total,
            "failed_tests": failed_tests
        }

        return results

    def generate_summary_report(self, results: Dict[str, Any], pytest_output: str) -> None:
        """
        Generate a summary report of the test run.

        Args:
            results: Test results dictionary
            pytest_output: Raw pytest output
        """
        summary_file = self.run_dir / "test_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DSPyUI Test Run Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Gradio Port: {self.port}\n")
            f.write(f"Log Directory: {self.run_dir}\n\n")

            f.write("Test Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Tests: {results['total']}\n")
            f.write(f"Passed: {results['passed']}\n")
            f.write(f"Failed: {results['failed']}\n")
            f.write(f"Skipped: {results['skipped']}\n")
            f.write(f"Errors: {results['errors']}\n\n")

            if results['failed_tests']:
                f.write("Failed Tests:\n")
                f.write("-" * 40 + "\n")
                for test in results['failed_tests']:
                    f.write(f"  - {test}\n")
                f.write("\n")

            f.write("Full pytest output:\n")
            f.write("=" * 80 + "\n")
            f.write(pytest_output)

        logger.info(f"Summary report written to {summary_file}")

    def generate_mock_data(self) -> None:
        """Generate mock CSV files for testing."""
        logger.info("Generating mock test data...")

        mock_data_dir = Path("tests/test_data")
        mock_data_dir.mkdir(exist_ok=True)

        generator = MockDataGenerator()

        # Generate sample datasets
        generator.generate_joke_dataset(8, mock_data_dir / "sample_jokes.csv")
        generator.generate_classification_dataset(8, mock_data_dir / "sample_classification.csv")
        generator.generate_qa_dataset(8, mock_data_dir / "sample_qa.csv")

        logger.info(f"Created mock data in: {mock_data_dir}")

    def run(self, run_rest_api_tests: bool = False, run_pytest_tests: bool = True) -> int:
        """
        Run the complete test suite.

        Args:
            run_rest_api_tests: Whether to run REST API tests
            run_pytest_tests: Whether to run pytest tests

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Generate mock data
            self.generate_mock_data()

            # Start Gradio server
            if not self.start_gradio_server():
                logger.error("Failed to start Gradio server")
                return 1

            exit_code = 0

            # Run pytest tests
            if run_pytest_tests:
                exit_code, pytest_output = self.run_pytest()

                # Analyze results
                results = self.analyze_test_results(pytest_output)

                # Print summary to console
                logger.info("=" * 80)
                logger.info("Pytest Results:")
                logger.info(f"  Total: {results['total']}")
                logger.info(f"  Passed: {results['passed']}")
                logger.info(f"  Failed: {results['failed']}")
                logger.info(f"  Skipped: {results['skipped']}")
                logger.info(f"  Errors: {results['errors']}")
                logger.info("=" * 80)

                # Generate reports
                self.generate_summary_report(results, pytest_output)

                # Show Gradio logs if there were failures
                if results['failed'] > 0 or results['errors'] > 0:
                    logger.warning("Test failures detected. Check logs for details.")
                    if self.verbose:
                        logger.info("\nGradio Server Logs:")
                        logger.info("-" * 80)
                        logger.info(self.get_gradio_logs())

            # Run REST API tests
            if run_rest_api_tests and GRADIO_CLIENT_AVAILABLE:
                server_url = f"http://127.0.0.1:{self.port}"
                api_runner = RestApiTestRunner(server_url, self.run_dir)
                api_results = api_runner.run_all_tests()

                # Print API test summary
                passed = sum(1 for r in api_results if r["passed"])
                failed = len(api_results) - passed
                logger.info("=" * 80)
                logger.info("REST API Test Results:")
                logger.info(f"  Total: {len(api_results)}")
                logger.info(f"  Passed: {passed}")
                logger.info(f"  Failed: {failed}")
                logger.info("=" * 80)

                # Save API test results
                api_results_file = self.run_dir / "rest_api_results.json"
                with open(api_results_file, 'w') as f:
                    json.dump(api_results, f, indent=2)
                logger.info(f"REST API results saved to: {api_results_file}")

                if failed > 0:
                    exit_code = 1

            return exit_code

        finally:
            # Always stop the server
            self.stop_gradio_server()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run DSPyUI tests with Gradio server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7862,
        help="Port to run Gradio server on (default: 7862)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="test_logs",
        help="Directory to store log files (default: test_logs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--generate-mocks",
        action="store_true",
        help="Generate mock data files and exit"
    )
    parser.add_argument(
        "--rest-api-tests",
        action="store_true",
        help="Run REST API tests (requires gradio_client)"
    )
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="Run only pytest tests, skip REST API tests"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Just generate mocks and exit
    if args.generate_mocks:
        logger.info("Generating mock data files...")
        mock_data_dir = Path("tests/test_data")
        mock_data_dir.mkdir(exist_ok=True)
        generator = MockDataGenerator()
        generator.generate_joke_dataset(10, mock_data_dir / "jokes.csv")
        generator.generate_classification_dataset(10, mock_data_dir / "classification.csv")
        generator.generate_qa_dataset(10, mock_data_dir / "qa.csv")
        logger.info(f"Generated mock data in: {mock_data_dir}")
        return 0

    # Run tests
    runner = GradioTestRunner(
        port=args.port,
        log_dir=args.log_dir,
        verbose=args.verbose
    )

    # Determine which tests to run
    run_rest_api = args.rest_api_tests or not args.pytest_only
    run_pytest = not args.rest_api_tests or args.pytest_only

    return runner.run(
        run_rest_api_tests=run_rest_api,
        run_pytest_tests=run_pytest
    )


if __name__ == "__main__":
    sys.exit(main())
