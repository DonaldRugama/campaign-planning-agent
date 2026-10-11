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
