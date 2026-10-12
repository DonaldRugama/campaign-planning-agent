"""Optional integration checks; install requirements.txt to enable them."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('gradio'), 'Install requirements.txt for browser integration checks')
class BrowserTests(unittest.TestCase):
    def test_callback_exports(self):
        import app
        original = app.run_campaign
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(app, 'run_campaign', side_effect=lambda *a, **kw: original(*a, output_root=folder, **kw)):
                report, trace, files = app.submit('TrailSip reusable water bottle', 'College students', 'Increase visits', 'Instagram', 200, 14, 'practice')
            self.assertEqual(len(files), 3)
            self.assertTrue(all(Path(f).exists() for f in files))
            self.assertIn('Executor completed', trace)

    def test_fractional_days_rejected(self):
        import app
        import gradio as gr
        with self.assertRaises(gr.Error):
            app.submit('Bottle', 'Students', 'Visits', 'Instagram', 200, 1.5, 'practice')

    def test_initialization_failure_does_not_poison_cache(self):
        import app
        import gradio as gr
        previous = dict(app.resources)
        app.resources.clear()
        try:
            with patch.object(app, 'LocalModel', return_value=object()), patch.object(app, 'Retriever', side_effect=OSError('Simulated download failure')):
                with self.assertRaises(gr.Error):
                    app.submit('Bottle', 'Students', 'Visits', 'Instagram', 200, 14, 'ai')
            self.assertEqual(app.resources, {})
        finally:
            app.resources.clear()
            app.resources.update(previous)

    def test_uploaded_brief_and_documents_callback(self):
        import app
        original = app.run_campaign
        with tempfile.TemporaryDirectory() as folder:
            brief = app.load_request(str(Path('templates/campaign_brief.json').resolve()))
            docs = [str(p.resolve()) for p in Path('templates').glob('*.txt')]
            with patch.object(app, 'run_campaign', side_effect=lambda *a, **kw: original(*a, output_root=Path(folder)/'out', **kw)):
                report, trace, files = app.submit(*brief, 'practice', docs, 'It costs $30 and comes in white and black.', True, progress=lambda *a, **kw: None)
            self.assertIn('DeskGlow',report)
            self.assertNotIn('750 ml',report)
            self.assertEqual(len(files),3)

    def test_changed_product_without_sources_rejected(self):
        import app
        import gradio as gr
        with self.assertRaises(gr.Error):
            app.submit('Another product','Students','Visits','Instagram',40,7,'practice',progress=lambda *a, **kw: None)
