#!/usr/bin/env python3
"""
Gradio Test Runner for DSPyUI

This script provides an automated testing infrastructure that:
1. Runs Gradio as a background process with stdout/stderr logging
2. Runs each test sequentially
3. Aggregates logs from both tests and the Gradio process
4. Uses Gradio's REST API for easier testing
5. Generates mock sample data where needed

Usage:
    python gradio_test_runner.py [--port PORT] [--log-dir DIR] [--verbose]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
                ["python", "-m", "gradio", "interface.py"],
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
        results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "total": 0,
            "failed_tests": []
        }

        # Parse pytest output
        lines = pytest_output.split('\n')
        for line in lines:
            if " PASSED" in line:
                results["passed"] += 1
            elif " FAILED" in line:
                results["failed"] += 1
                results["failed_tests"].append(line.strip())
            elif " SKIPPED" in line:
                results["skipped"] += 1
            elif " ERROR" in line:
                results["errors"] += 1

        results["total"] = results["passed"] + results["failed"] + results["skipped"] + results["errors"]

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

        # Generate sample jokes CSV if it doesn't exist
        sample_jokes = mock_data_dir / "sample_jokes.csv"
        if not sample_jokes.exists():
            with open(sample_jokes, 'w') as f:
                f.write("topic,joke\n")
                f.write("chickens,Why did the chicken cross the road? To get to the other side!\n")
                f.write("computers,Why do programmers prefer dark mode? Because light attracts bugs!\n")
                f.write("science,Why can't you trust atoms? Because they make up everything!\n")
            logger.info(f"Created mock data: {sample_jokes}")

    def run(self) -> int:
        """
        Run the complete test suite.

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

            # Run tests
            exit_code, pytest_output = self.run_pytest()

            # Analyze results
            results = self.analyze_test_results(pytest_output)

            # Print summary to console
            logger.info("=" * 80)
            logger.info("Test Results:")
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
        default=7860,
        help="Port to run Gradio server on (default: 7860)"
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

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run tests
    runner = GradioTestRunner(
        port=args.port,
        log_dir=args.log_dir,
        verbose=args.verbose
    )

    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
