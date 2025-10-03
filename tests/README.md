# DSPyUI Test Suite

Playwright-based end-to-end tests for the DSPyUI Gradio interface.

## Setup

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

3. Ensure you have a `.env` file with API keys (tests will use conservative parameters)

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_basic_ui.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run in headed mode (visible browser):
```bash
pytest tests/ --headed
```

### Run specific test:
```bash
pytest tests/test_basic_ui.py::TestBasicUI::test_app_loads
```

## Test Structure

- **conftest.py**: Pytest configuration and fixtures
  - `gradio_server`: Starts Gradio server for testing
  - `page`: Provides a fresh browser page for each test
  - `test_data_dir`: Path to test data files

- **test_basic_ui.py**: Basic UI component tests
  - Verifies the app loads correctly
  - Checks that main tabs and components exist
  - Tests navigation between tabs

- **test_interactions.py**: User interaction tests
  - Adding/removing input/output fields
  - Configuring LLM settings
  - Loading demo examples
  - Data upload workflows

- **test_data/**: Sample data files for testing
  - `sample_jokes.csv`: Example CSV for upload testing

## Test Coverage

Current test coverage focuses on:
- ✅ UI component presence
- ✅ Basic interactions (adding fields)
- ✅ Tab navigation
- ⚠️  File uploads (requires special Playwright handling)
- ⚠️  Full compile workflow (requires API calls, uses conservative settings)
- ⚠️  Demo examples (selectors need to be identified in Gradio 5)

## Notes

- Tests run with `headless=True` by default for CI/CD compatibility
- The Gradio server starts on port 7861 for testing to avoid conflicts
- Some tests are marked as `@pytest.mark.skip` pending Gradio 5 selector verification
- API calls in tests use minimal datasets to conserve costs

## Troubleshooting

### Server doesn't start
- Check that port 7861 is available
- Ensure all dependencies are installed
- Verify `.env` file exists (even if empty)

### Tests fail on element selectors
- Gradio 5 may have different element structures
- Use `--headed` mode to inspect the page
- Update selectors based on actual rendered HTML

### Playwright browsers not found
- Run `playwright install chromium`
- Check Playwright version compatibility

## Future Improvements

- [ ] Add integration tests for complete compile workflows
- [ ] Test all three demo examples thoroughly
- [ ] Add tests for View Prompts tab functionality
- [ ] Test program generation/inference
- [ ] Add visual regression tests for UI consistency
- [ ] CI/CD integration with GitHub Actions
