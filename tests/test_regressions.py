import json
import tempfile
import unittest
from workflow import Brief, PracticeModel, Retriever, run_campaign

class RegressionTests(unittest.TestCase):
    def run_case(self, model, brief=None):
        with tempfile.TemporaryDirectory() as folder:
            return run_campaign(brief=brief, mode='ai', model=model, retriever=Retriever(), output_root=folder)[0]

    def test_duplicate_ai_choices_are_labeled_and_distinct(self):
        class Repeat(PracticeModel):
            def generate(self, role, payload):
                return '{"hook_id":0,"fact_id":0,"cta_id":0}' if role == 'executor' else super().generate(role, payload)
        state = self.run_case(Repeat())
        self.assertEqual(len({p['caption'] for p in state.posts}), 3)
        self.assertEqual([p['method'] for p in state.posts], ['accepted','fallback','fallback'])

    def test_nonsequential_choices_leave_valid_fallback_options(self):
        class Repeat(PracticeModel):
            def generate(self, role, payload):
                return '{"hook_id":2,"fact_id":0,"cta_id":1}' if role == 'executor' else super().generate(role, payload)
        self.assertEqual(len({p['caption'] for p in self.run_case(Repeat()).posts}), 3)

    def test_month_and_competitor_questions_cannot_enter_plan(self):
        class OffTopic(PracticeModel):
            def generate(self, role, payload):
                return json.dumps({'queries':['Compare competitors next month?']*3}) if role == 'planner' else super().generate(role,payload)
        state=self.run_case(OffTopic(), Brief(days=9))
        self.assertIn('9-day',state.plan)
        self.assertNotIn('next month',state.plan)
        self.assertEqual(state.generation_events[0]['status'],'fallback')

    def test_source_index_selection_can_succeed(self):
        state=self.run_case(PracticeModel())
        self.assertEqual(state.generation_events[1]['status'],'accepted')
        for post in state.posts:
            self.assertIn(post['source_id'],{e['id'] for e in state.selected_evidence})

    def test_missing_product_evidence_selection_falls_back(self):
        class Omit(PracticeModel):
            def generate(self, role, payload):
                if role=='researcher':
                    other=next(i for i in range(len(payload['evidence'])) if i not in payload['required_indices'])
                    return json.dumps({'selected_indices':[other]})
                return super().generate(role,payload)
        state=self.run_case(Omit())
        self.assertEqual(state.generation_events[1]['status'],'fallback')
        self.assertIn('approved product source',state.generation_events[1]['reason'])

from workflow import validate_question_ids


class QuestionIdValidationTests(unittest.TestCase):
    def test_rejects_empty_unicode_boolean_and_out_of_range_indices(self):
        for question_ids in ([], [0, 1, '２'], [0, 1, True], [-1, 1, 2], [0, 1, 3]):
            with self.subTest(question_ids=question_ids):
                with self.assertRaises(ValueError):
                    validate_question_ids({'question_ids': question_ids})

