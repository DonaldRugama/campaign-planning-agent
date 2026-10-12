import json
import tempfile
import unittest
from pathlib import Path
from request_inputs import parse_brief, prepare_sources
from workflow import Brief, PracticeModel, run_campaign

class UploadTests(unittest.TestCase):
    def sources(self, base, product='DeskLamp'):
        doc=Path(base)/'source.txt'
        doc.write_text(f'{product} costs $30. Brand tone is practical. Track product-page visits.',encoding='utf-8')
        return [str(doc)], f'{product} costs $30.'

    def test_custom_product_uses_uploaded_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            docs,quote=self.sources(d)
            retriever,facts=prepare_sources(docs,quote,True,Path(d)/'private')
            state,files=run_campaign(Brief(product='DeskLamp'),retriever=retriever,model=PracticeModel(),facts_path=facts,output_root=Path(d)/'out')
            self.assertIn('DeskLamp costs $30.',state.campaign)
            self.assertNotIn('750 ml',state.campaign)
            self.assertEqual(len({p['caption'] for p in state.posts}),3)
            self.assertTrue(all(p['source_id'].startswith('uploaded_') for p in state.posts))
            self.assertEqual(len(files),3)

    def test_isolated_runs_do_not_share_documents(self):
        with tempfile.TemporaryDirectory() as d:
            docs,q=self.sources(d)
            r1,_=prepare_sources(docs,q,True,Path(d)/'one')
            docs,q=self.sources(d,'OtherLamp')
            r2,_=prepare_sources(docs,q,True,Path(d)/'two')
            self.assertNotIn('OtherLamp',r1.chunks[0]['text'])
            self.assertIn('OtherLamp',r2.chunks[0]['text'])

    def test_quote_and_approval_required(self):
        with tempfile.TemporaryDirectory() as d:
            docs,q=self.sources(d)
            for quote,approval in [('Invented cooling benefit.',True),(q,False)]:
                with self.assertRaises(ValueError):prepare_sources(docs,quote,approval,Path(d)/str(approval))

    def test_brief_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'brief.json';obj={'product':'DeskLamp','audience':'Students','goal':'Increase visits','channel':'Instagram','budget':40,'days':7}
            p.write_text(json.dumps(obj));self.assertEqual(parse_brief(p)['days'],7)
            obj['days']=7.5;p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):parse_brief(p)
            p.write_text('{')
            with self.assertRaises(ValueError):parse_brief(p)

    def test_unsupported_and_large_documents_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.pdf';p.write_text('text')
            with self.assertRaises(ValueError):prepare_sources([p],'text',True,Path(d)/'one')
            p=Path(d)/'big.txt';p.write_text('x'*200001)
            with self.assertRaises(ValueError):prepare_sources([p],'text',True,Path(d)/'two')

    def test_duplicate_filenames_preserve_both_sources(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i in range(2):
                folder=Path(d)/str(i);folder.mkdir();p=folder/'notes.txt';p.write_text(f'Fact {i}.');paths.append(p)
            r,_=prepare_sources(paths,'Fact 1.',True,Path(d)/'private')
            self.assertEqual(len(r.chunks),2)
            self.assertEqual(len({c['id'] for c in r.chunks}),2)
