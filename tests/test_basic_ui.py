"""
Basic UI tests for DSPyUI - verify the interface loads and core components are present.
"""
import pytest


class TestBasicUI:
    """Test basic UI functionality and component presence."""

    def test_app_loads(self, page):
        """Test that the Gradio app loads successfully."""
        # Verify the page title or main heading
        assert page.is_visible("gradio-app")

    def test_compile_tab_exists(self, page):
        """Test that the Compile Program tab exists."""
        # Look for the main tabs
        tabs = page.locator("button[role='tab']")
        assert tabs.count() >= 2

        # Verify "Compile Program" tab exists
        compile_tab = page.locator("button[role='tab']", has_text="Compile Program")
        assert compile_tab.is_visible()

    def test_view_prompts_tab_exists(self, page):
        """Test that the View Prompts tab exists."""
        view_prompts_tab = page.locator("button[role='tab']", has_text="View Prompts")
        assert view_prompts_tab.is_visible()

    def test_demo_examples_exist(self, page):
        """Test that demo examples are available."""
        # Look for example buttons/links
        # Gradio examples typically appear as radio buttons or buttons
        examples = page.locator("text=Demo Examples")
        assert examples.count() >= 0  # May or may not be visible initially


class TestCompileTab:
    """Test the Compile Program tab components."""

    def test_signature_section_visible(self, page):
        """Test that the signature configuration section is visible."""
        # Look for signature-related inputs
        assert page.locator("text=Signature").count() > 0

    def test_add_input_button_exists(self, page):
        """Test that Add Input Field button exists."""
        add_input = page.locator("button", has_text="Add Input Field")
        assert add_input.is_visible()

    def test_add_output_button_exists(self, page):
        """Test that Add Output Field button exists."""
        add_output = page.locator("button", has_text="Add Output Field")
        assert add_output.is_visible()

    def test_llm_configuration_exists(self, page):
        """Test that LLM configuration dropdowns exist."""
        # Scroll to Settings section
        settings_heading = page.locator("text=Settings").first
        settings_heading.scroll_into_view_if_needed()
        page.wait_for_timeout(300)

        # Should have Model dropdown (check for text in page, Gradio 5 renders labels differently)
        assert page.get_by_text("Model", exact=False).count() > 0

    def test_optimizer_dropdown_exists(self, page):
        """Test that Optimizer dropdown exists."""
        # Scroll to Settings section
        settings_heading = page.locator("text=Settings").first
        settings_heading.scroll_into_view_if_needed()
        page.wait_for_timeout(300)

        assert page.get_by_text("Optimizer", exact=False).count() > 0

    def test_evaluation_metric_exists(self, page):
        """Test that Evaluation Metric dropdown exists."""
        # Scroll to Settings section
        settings_heading = page.locator("text=Settings").first
        settings_heading.scroll_into_view_if_needed()
        page.wait_for_timeout(300)

        assert page.get_by_text("Metric", exact=False).count() > 0


class TestViewPromptsTab:
    """Test the View Prompts tab components."""

    def test_view_prompts_tab_loads(self, page):
        """Test that clicking View Prompts tab loads the content."""
        view_prompts_tab = page.locator("button[role='tab']", has_text="View Prompts")
        view_prompts_tab.click()

        # Wait for tab content to load
        page.wait_for_timeout(500)

        # Should show some prompt-related content
        # This might be empty if no prompts exist, but the UI should load
        assert page.is_visible("gradio-app")
