import unittest,tempfile,json,csv
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from workflow import Brief,PracticeModel,Retriever,run_campaign,LocalModel,GenerationIncomplete

class WorkflowTests(unittest.TestCase):
 def run_temp(self,**kw):
  with tempfile.TemporaryDirectory() as d:
   s,f=run_campaign(output_root=d,**kw)
   self.assertEqual(len(s.posts),3)
   self.assertEqual(len(s.trace),4)
   self.assertEqual(json.loads(Path(f[2]).read_text())['version'],'1.5')
   with open(f[1],newline='') as c:self.assertEqual(len(list(csv.reader(c))),s.brief['days']+1)
   self.assertTrue(s.campaign.endswith('Review all draft posts before publishing.'))
   return s
 def test_complete_export(self):
  s=self.run_temp();self.assertIn('Creative test allocation: $50.00',s.campaign);self.assertIn('Reserve pending review: $150.00',s.campaign)
 def test_bad_output_fallback(self):
  class Bad:
   def generate(self,*a):return 'Cooling effects! $200 per day'
  s=self.run_temp(mode='ai',model=Bad(),retriever=Retriever())
  self.assertTrue(all(e['status']=='fallback' for e in s.generation_events));self.assertNotIn('Cooling',s.campaign)
 def test_truncated_output(self):
  class Bad:
   def generate(self,*a):raise GenerationIncomplete('limit reached')
  s=self.run_temp(model=Bad());self.assertTrue(any('limit reached' in e['reason'] for e in s.generation_events))
 def test_unknown_citation(self):
  class Bad(PracticeModel):
   def generate(self,r,p):return '{"selected_ids":["invented.md:1"]}' if r=='researcher' else super().generate(r,p)
  s=self.run_temp(model=Bad());self.assertNotIn('invented.md',s.research)
 def test_handoffs(self):
  calls=[]
  class Capture(PracticeModel):
   def generate(self,r,p):calls.append((r,p));return super().generate(r,p)
  s=self.run_temp(model=Capture());self.assertEqual([x[0] for x in calls],['planner','researcher','executor','executor','executor']);self.assertEqual(calls[2][1]['research'],s.research)
 def test_bad_component(self):
  class Bad(PracticeModel):
   def generate(self,r,p):return '{"hook_id":99,"fact_id":0,"cta_id":0}' if r=='executor' else super().generate(r,p)
  s=self.run_temp(model=Bad());self.assertTrue(all(p['method']=='fallback' for p in s.posts))
 def test_unapproved_fact(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'facts.json';f.write_text('[{"source_id":"product_catalog.md:1","quote":"Keeps water cold"}]')
   with self.assertRaises(ValueError):run_campaign(output_root=Path(d)/'out',facts_path=f)
   self.assertFalse((Path(d)/'out').exists())
 def test_small_budget(self):self.assertIn('Allocated total: $0.03',self.run_temp(brief=Brief(budget=.03,days=1)).campaign)
 def test_inputs(self):
  for b in [Brief(days=1.2),Brief(days=True),Brief(budget=float('nan')),Brief(product=None)]:
   with self.assertRaises(ValueError):b.validate()
 def test_empty_retrieval(self):
  class Empty:
   def retrieve(self,*a):return []
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):run_campaign(retriever=Empty(),output_root=d)
   self.assertEqual(list(Path(d).iterdir()),[])

class Tokens:shape=(1,3)
class NewTokens:
 def __init__(self,ids):self.ids=ids
 def tolist(self):return self.ids
class Output:
 def __init__(self,ids):self.ids=ids
 def __getitem__(self,k):
  assert k==(0,slice(3,None));return NewTokens(self.ids)
class Batch(dict):
 def to(self,device):return self
class ModelContractTests(unittest.TestCase):
 def make_model(self,ids):
  m=LocalModel.__new__(LocalModel);m.torch=SimpleNamespace(inference_mode=nullcontext)
  def template(messages,**kw):
   self.assertTrue(kw['return_dict']);return Batch(input_ids=Tokens(),attention_mask=Tokens())
  def generate(**kw):
   self.assertIn('attention_mask',kw);return Output(ids)
  m.model=SimpleNamespace(device='cpu',generate=generate,generation_config=SimpleNamespace(eos_token_id=0))
  m.tokenizer=SimpleNamespace(apply_chat_template=template,eos_token_id=0,decode=lambda *a,**k:'{"queries":[]}')
  return m
 def test_attention_mask(self):
  m=self.make_model([1,0]);self.assertEqual(m.generate('planner',{}),'{"queries":[]}');self.assertFalse(m.last_generation['hit_limit'])
 def test_token_limit(self):
  m=self.make_model([1]*160)
  with self.assertRaises(GenerationIncomplete):m.generate('planner',{})
  self.assertTrue(m.last_generation['hit_limit'])
