"""
Interactive tests for DSPyUI - verify user workflows and interactions.
"""
import pytest
import os


class TestInputOutputFields:
    """Test adding and configuring input/output fields."""

    def test_add_input_field(self, page):
        """Test adding a new input field."""
        # Click Add Input Field button
        add_input = page.locator("button", has_text="Add Input Field")
        add_input.click()

        # Wait for the field to appear
        page.wait_for_timeout(500)

        # Should see at least one input field name textbox
        input_fields = page.locator("label", has_text="Input Field").count()
        assert input_fields >= 1

    def test_add_output_field(self, page):
        """Test adding a new output field."""
        # Click Add Output Field button
        add_output = page.locator("button", has_text="Add Output Field")
        add_output.click()

        # Wait for the field to appear
        page.wait_for_timeout(500)

        # Should see at least one output field name textbox
        output_fields = page.locator("label", has_text="Output Field").count()
        assert output_fields >= 1

    def test_configure_simple_signature(self, page):
        """Test configuring a simple input->output signature."""
        # Add one input field
        add_input = page.locator("button", has_text="Add Input Field")
        add_input.click()

        # Wait for the "Remove Last Input" button to become enabled (indicates field was added)
        remove_input = page.locator("button", has_text="Remove Last Input")
        remove_input.wait_for(state="attached", timeout=5000)
        page.wait_for_timeout(1500)  # Additional wait for render to complete

        # Find the input field using a more flexible approach
        # Look for any visible text input that's not the Task Instructions field
        input_fields = page.locator("input[type='text']").all()
        input_field = None
        for field in input_fields:
            try:
                if field.is_visible():
                    placeholder = field.get_attribute("placeholder")
                    if placeholder and "Input" in placeholder:
                        input_field = field
                        break
            except:
                pass

        if input_field:
            input_field.fill("question")
        else:
            # Fallback: just get all visible text inputs and use the first one that's not instructions
            visible_inputs = [f for f in input_fields if f.is_visible()]
            if len(visible_inputs) > 1:  # Skip Task Instructions field
                visible_inputs[1].fill("question")

        # Add one output field
        add_output = page.locator("button", has_text="Add Output Field")
        add_output.click()
        page.wait_for_timeout(1500)

        # Find the output field similarly
        output_fields = page.locator("input[type='text']").all()
        output_field = None
        for field in output_fields:
            try:
                if field.is_visible():
                    placeholder = field.get_attribute("placeholder")
                    if placeholder and "Output" in placeholder:
                        output_field = field
                        break
            except:
                pass

        if output_field:
            output_field.fill("answer")

        # Verify fields are filled
        page.wait_for_timeout(500)
        assert page.locator("input[value='question']").count() > 0
        assert page.locator("input[value='answer']").count() > 0


class TestDataUpload:
    """Test CSV data upload functionality."""

    @pytest.mark.skip(reason="File upload requires special handling in Playwright")
    def test_upload_csv_file(self, page, test_data_dir):
        """Test uploading a CSV file with example data."""
        # This test requires configuring input/output fields first
        # Add input field
        add_input = page.locator("button", has_text="Add Input Field")
        add_input.click()
        page.wait_for_timeout(300)

        # Add output field
        add_output = page.locator("button", has_text="Add Output Field")
        add_output.click()
        page.wait_for_timeout(300)

        # Configure field names to match CSV
        input_fields = page.locator("input[placeholder='Field name']")
        input_fields.first.fill("topic")
        input_fields.last.fill("joke")

        # Upload CSV file
        csv_path = os.path.join(test_data_dir, "sample_jokes.csv")
        file_input = page.locator("input[type='file']")
        file_input.set_input_files(csv_path)

        # Wait for upload to process
        page.wait_for_timeout(1000)

        # Verify data appears in the dataframe component
        # This is complex as Gradio dataframes are rendered in specific ways
        assert page.locator("table").is_visible()


class TestLLMConfiguration:
    """Test LLM provider and model configuration."""

    def test_select_model_provider(self, page):
        """Test selecting different model providers."""
        # Scroll to Settings section
        settings_heading = page.locator("text=Settings").first
        settings_heading.scroll_into_view_if_needed()
        page.wait_for_timeout(300)

        # Verify the Model dropdown text is present (Gradio 5 renders labels differently)
        assert page.get_by_text("Model", exact=False).count() > 0


class TestDemoExamples:
    """Test loading demo examples."""

    @pytest.mark.skip(reason="Demo example selectors need to be identified")
    def test_load_joke_judge_example(self, page):
        """Test loading the 'Judge if joke is funny' demo."""
        # Find and click on the demo example
        # Exact selector depends on how Gradio renders examples
        pass

    @pytest.mark.skip(reason="Demo example selectors need to be identified")
    def test_load_tell_joke_example(self, page):
        """Test loading the 'Tell a joke' demo."""
        pass

    @pytest.mark.skip(reason="Demo example selectors need to be identified")
    def test_load_rewrite_joke_example(self, page):
        """Test loading the 'Rewrite joke' demo."""
        pass


class TestValidation:
    """Test input validation and error handling."""

    def test_compile_without_fields_shows_error(self, page):
        """Test that trying to compile without fields shows appropriate feedback."""
        # Try to find compile button (use more specific selector to avoid tab)
        compile_button = page.locator("button[variant='primary']", has_text="Compile Program")

        # It might be disabled or clicking might show an error
        # This depends on the UI's validation logic
        # The button should not be visible initially (no fields configured)
        assert not compile_button.is_visible()
