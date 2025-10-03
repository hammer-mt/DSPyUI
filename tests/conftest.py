"""
Pytest configuration for DSPyUI Playwright tests.
"""
import os
import subprocess
import time
import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def gradio_server():
    """
    Start the Gradio server for testing and tear it down after tests complete.
    """
    # Set up environment variables for testing
    env = os.environ.copy()

    # Start the Gradio server in the background
    # Use a different port for testing to avoid conflicts
    env["GRADIO_SERVER_PORT"] = "7861"

    process = subprocess.Popen(
        ["python", "interface.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    max_retries = 30
    for i in range(max_retries):
        try:
            import requests
            response = requests.get("http://localhost:7861")
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        # Kill process if server didn't start
        process.kill()
        raise RuntimeError("Gradio server failed to start")

    yield "http://localhost:7861"

    # Tear down: kill the server
    process.terminate()
    process.wait(timeout=5)


@pytest.fixture(scope="function")
def page(gradio_server):
    """
    Create a new browser page for each test.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(gradio_server)

        # Wait for Gradio to fully load
        page.wait_for_selector("gradio-app", timeout=10000)

        yield page

        context.close()
        browser.close()


@pytest.fixture
def test_data_dir():
    """
    Return path to test data directory.
    """
    return os.path.join(os.path.dirname(__file__), "test_data")
