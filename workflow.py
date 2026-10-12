"""Three role-based agents, document retrieval, and controlled file automation."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import json
import math
import re
import uuid

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
EMBEDDING_ID = 'sentence-transformers/all-MiniLM-L6-v2'
PROMPTS = {
    'planner': 'Return ONLY JSON: {"question_ids":[0,1,2]}. Order the three supplied document questions by importance. Use each zero-based integer exactly once. Do not write new questions.',
    'researcher': 'Return ONLY JSON: {"selected_indices":[0,1,2]}. Select one to three zero-based evidence indices. Include every required_index. Use only supplied evidence. Do not write source names or facts.',
    'executor': 'Return ONLY JSON: {"hook_id":0,"fact_id":0,"cta_id":0}. Choose one supplied hook, approved fact and call to action for this single post. IDs are zero-based integers. Use only available_hook_ids and available_cta_ids so posts differ. Never write additional claims or additional posts.'
}
TOKEN_LIMITS = {'planner': 160, 'researcher': 100, 'executor': 60}
HOOKS = ['Meet {product}.', 'Make room for {product}.', 'Take a closer look at {product}.']
CTAS = ['Explore the product details.', 'Which color would you choose?', 'Learn more on the product page.']

@dataclass
class Brief:
    product: str = 'TrailSip reusable water bottle'
    audience: str = 'College students'
    goal: str = 'Increase product-page visits and purchases'
    channel: str = 'Instagram'
    budget: float = 200
    days: int = 14

    def validate(self):
        for name in ('product', 'audience', 'goal', 'channel'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Please supply {name}.')
        if isinstance(self.days, bool) or not isinstance(self.days, int) or not 1 <= self.days <= 90:
            raise ValueError('Duration must be 1 to 90 days.')
        if isinstance(self.budget, bool) or not isinstance(self.budget, (int, float)) or not math.isfinite(self.budget) or not 0 <= self.budget <= 100000:
            raise ValueError('Budget must be between 0 and 100000.')

@dataclass
class State:
    brief: dict
    mode: str
    plan: str = ''
    evidence: list = field(default_factory=list)
    research: str = ''
    campaign: str = ''
    trace: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    version: str = '1.5'
    posts: list = field(default_factory=list)
    generation_events: list = field(default_factory=list)
    selected_evidence: list = field(default_factory=list)

class PracticeModel:
    """Deterministic selections; no LLM or semantic retrieval."""
    def generate(self, role, payload):
        if role == 'planner':
            return json.dumps({'question_ids': [0, 1, 2]})
        if role == 'researcher':
            return json.dumps({'selected_indices': list(dict.fromkeys(payload['required_indices'] + list(range(min(3, len(payload['evidence']))))))[:3]})
        i = payload['post_number'] - 1
        return json.dumps({'hook_id': i, 'fact_id': 0, 'cta_id': i})

class GenerationIncomplete(ValueError):
    pass

class LocalModel:
    """Downloads an open-weight model; inference runs on your runtime."""
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.last_generation = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    def generate(self, role, payload):
        messages = [{'role': 'system', 'content': PROMPTS[role]},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True,
                     add_generation_prompt=True, return_tensors='pt', return_dict=True).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=TOKEN_LIMITS[role], do_sample=False,
                       pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = output[0, inputs['input_ids'].shape[1]:]
        token_ids = new_tokens.tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        stopped = bool(token_ids) and token_ids[-1] in eos_ids
        capped = len(token_ids) >= TOKEN_LIMITS[role] and not stopped
        self.last_generation = {'role': role, 'tokens': len(token_ids), 'hit_limit': capped}
        if capped:
            raise GenerationIncomplete(f'{role} reached the generation limit; partial text was discarded.')
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise ValueError(f'The {role} model returned no text.')
        return text


class Retriever:
    def __init__(self, data_dir=ROOT / 'data', semantic=False):
        self.chunks = []
        for path in sorted(Path(data_dir).glob('*.md')):
            words = path.read_text(encoding='utf-8').split()
            for start in range(0, len(words), 90):
                self.chunks.append({'id': f'{path.name}:{start // 90 + 1}',
                                    'text': ' '.join(words[start:start + 120])})
        if not self.chunks:
            raise ValueError('Add at least one non-empty .md document to data/.')
        self.encoder = None
        if semantic:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(EMBEDDING_ID)
            self.vectors = self.encoder.encode([x['text'] for x in self.chunks], normalize_embeddings=True)

    def retrieve(self, query, k=5):
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer.')
        if self.encoder is not None:
            vector = self.encoder.encode(query, normalize_embeddings=True)
            scores = self.vectors @ vector
        else:
            terms = set(re.findall(r'\w+', query.lower()))
            scores = [len(terms & set(re.findall(r'\w+', c['text'].lower()))) for c in self.chunks]
        indices = sorted(range(len(self.chunks)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in indices if scores[i] > 0]

def structured_call(state, model, role, payload, validator, fallback):
    """One short model call; malformed or truncated output is never rendered."""
    try:
        raw = model.generate(role, payload)
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned)
        obj = json.loads(cleaned)
        value = validator(obj)
        state.generation_events.append({'role': role, 'status': 'accepted', 'raw': raw,
                                        'generation': getattr(model, 'last_generation', {})})
        return value
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        state.warnings.append(f'{role}: invalid or incomplete model selection; used a labeled Python fallback.')
        state.generation_events.append({'role': role, 'status': 'fallback', 'reason': str(exc),
                                        'generation': getattr(model, 'last_generation', {})})
        return fallback


def validate_question_ids(obj):
    ids = obj['question_ids']
    if not isinstance(ids, list) or len(ids) != 3 or any(type(i) is not int for i in ids) or sorted(ids) != [0, 1, 2]:
        raise ValueError('Select each supplied question index exactly once.')
    return ids

class PlannerAgent:
    def run(self, state, model):
        brief = state.brief
        questions = [
            f"What facts about {brief['product']} are documented in the supplied catalog?",
            f"What documented brand rules apply to the proposed {brief['channel']} campaign for {brief['audience']}?",
            f"Which documented metrics can evaluate the goal '{brief['goal']}' during this {brief['days']}-day campaign?"
        ]
        ids = structured_call(state, model, 'planner', {'brief': brief, 'questions': questions}, validate_question_ids, [0, 1, 2])
        state.plan = 'Proposed research questions (not findings):\n' + '\n'.join(f'- {questions[i]}' for i in ids)
        state.trace.append('Planner completed: questions passed to Researcher')

class ResearchAgent:
    def run(self, state, model, retriever, facts_path=None):
        state.evidence = retriever.retrieve(json.dumps(state.brief) + ' ' + state.plan)
        if not state.evidence:
            raise ValueError('No matching evidence. Add relevant documents before continuing.')
        # At least one approved product source must reach the Executor.
        facts = load_approved_facts(state.evidence, facts_path or ROOT / 'data' / 'approved_facts.json')
        required = [next(i for i, e in enumerate(state.evidence) if e['id'] == facts[0]['source_id'])]
        def validate(obj):
            indices = obj['selected_indices']
            if not isinstance(indices, list) or not 1 <= len(indices) <= 3 or any(type(i) is not int or not 0 <= i < len(state.evidence) for i in indices):
                raise ValueError('Research must select known evidence indices.')
            if not set(required).issubset(indices):
                raise ValueError('Research must include an approved product source.')
            return list(dict.fromkeys(indices))
        fallback = list(dict.fromkeys(required + list(range(len(state.evidence)))))[:3]
        indices = structured_call(state, model, 'researcher', {'brief': state.brief, 'plan': state.plan,
            'evidence': [dict(e, index=i) for i, e in enumerate(state.evidence)], 'required_indices': required}, validate, fallback)
        selected = [state.evidence[i] for i in indices]
        state.selected_evidence = selected
        state.research = 'Selected document excerpts (verbatim; no live web research):\n\n' + '\n\n'.join(f"[{e['id']}] {e['text']}" for e in selected)
        state.trace.append('Researcher completed: cited document excerpts passed to Executor')


def load_approved_facts(evidence, facts_path=ROOT / 'data' / 'approved_facts.json'):
    """Human-approved quotes must also exist in retrieved evidence."""
    facts = json.loads(Path(facts_path).read_text(encoding='utf-8'))
    known = {x['id']: ' '.join(x['text'].split()) for x in evidence}
    valid = []
    for fact in facts:
        source, quote = fact['source_id'], fact['quote']
        if isinstance(quote, str) and quote.strip() and ' '.join(quote.split()) in known.get(source, ''):
            valid.append({'source_id': source, 'quote': quote})
    if not valid:
        raise ValueError('No approved product facts match retrieved evidence. Update data/approved_facts.json for your documents.')
    return valid

class ExecutorAgent:
    def run(self, state, model, approved_facts=None):
        facts = approved_facts if approved_facts is not None else load_approved_facts(state.evidence)
        if not facts:
            raise ValueError('At least one verified approved fact is required.')
        hooks = [x.format(product=state.brief['product']) for x in HOOKS]
        used_hooks, used_ctas = set(), set()
        for i in range(3):
            def validate(obj):
                for key, choices in [('hook_id', hooks), ('fact_id', facts), ('cta_id', CTAS)]:
                    n = obj[key]
                    if type(n) is not int or not 0 <= n < len(choices):
                        raise ValueError(f'Invalid {key}')
                if obj['hook_id'] in used_hooks or obj['cta_id'] in used_ctas:
                    raise ValueError('Repeated hook or call to action; distinct posts required.')
                return {k: obj[k] for k in ('hook_id', 'fact_id', 'cta_id')}
            pick = structured_call(state, model, 'executor', {'brief': state.brief, 'plan': state.plan,
                 'research': state.research, 'post_number': i + 1, 'hooks': hooks, 'approved_facts': facts, 'ctas': CTAS, 'available_hook_ids': sorted(set(range(len(hooks))) - used_hooks),
                 'available_cta_ids': sorted(set(range(len(CTAS))) - used_ctas)},
                 validate, {'hook_id': min(set(range(len(hooks))) - used_hooks), 'fact_id': 0,
                            'cta_id': min(set(range(len(CTAS))) - used_ctas)})
            used_hooks.add(pick['hook_id'])
            used_ctas.add(pick['cta_id'])
            fact = facts[pick['fact_id']]
            state.posts.append({'number': i + 1, 'caption': f"{hooks[pick['hook_id']]} {fact['quote']} {CTAS[pick['cta_id']]}",
                                'source_id': fact['source_id'], 'method': state.generation_events[-1]['status']})
        if len(state.posts) != 3:
            raise ValueError('Expected exactly three posts.')
        budget = state.brief['budget']
        # Integer cents ensure exact arithmetic, including small budgets.
        cents = round(budget * 100)
        test = cents // 4
        reserve = cents - test
        state.campaign = f"## Proposed strategy\nTest product-fact creative on {state.brief['channel']} for {state.brief['audience']}. This audience choice is a hypothesis. Goal: {state.brief['goal']}. No numerical forecast is supplied.\n\n## Three draft posts\n" + '\n\n'.join(f"{p['number']}. {p['caption']} [{p['source_id']}] (selection: {p['method']})" for p in state.posts)
        state.campaign += f"\n\n## Budget proposal - calculated by Python\nTotal cap: ${cents / 100:.2f} across {state.brief['days']} days.\nCreative test allocation: ${test / 100:.2f}.\nReserve pending review: ${reserve / 100:.2f}.\nAllocated total: ${(test + reserve) / 100:.2f}.\nThis 25% test allocation is a template suggestion, not research. No spending was triggered."
        state.campaign += "\n\n## Measurement - fixed definitions\nCTR = link clicks / impressions.\nPurchase conversion rate = purchases / product-page visits.\nROAS = attributed revenue / ad spend.\nA rate is undefined when its denominator is zero. No customer lifetime value is calculated.\n\nReview all draft posts before publishing."
        state.trace.append('Executor completed: three source-backed template posts and fixed budget/metrics ready')


def review_draft(state):
    if state.mode == 'ai':
        state.warnings.append('AI selects questions, evidence and template components; Python renders approved quotes. This is constrained generation, not unrestricted copywriting.')
    if len(state.posts) != 3 or len({p['caption'] for p in state.posts}) != 3:
        raise ValueError('Draft requires exactly three distinct posts.')
    state.warnings.append('Source matching verifies quotation provenance, not whether the source itself is true. Human review is required.')
    return state.warnings


def export_files(state, output_root):
    """A deterministic tool: model text cannot select paths or execute commands."""
    folder = Path(output_root) / uuid.uuid4().hex
    folder.mkdir(parents=True)
    report = folder / 'campaign_report.md'
    report.write_text(f"# Campaign draft - version 1.5\n\nMode: {state.mode}\n\nReview claims, citations and budget before use.\n\n## Requested settings - authoritative\nBudget is ${state.brief['budget']:.2f} TOTAL across {state.brief['days']} days.\n{json.dumps(state.brief, indent=2)}\n\n## Draft review warnings\n{chr(10).join(state.warnings) or "No automated flags. Manual review is still required."}\n\n## Planner\n{state.plan}\n\n## Researcher\n{state.research}\n\n## Executor\n{state.campaign}\n\n## Retrieved evidence\n" + '\n\n'.join(f"[{e['id']}] {e['text']}" for e in state.evidence), encoding='utf-8')
    calendar = folder / 'content_calendar.csv'
    with calendar.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['day', 'channel', 'activity', 'status'])
        for day in range(1, state.brief['days'] + 1):
            activity = 'Review visits and purchases' if day % 7 == 0 else ['Product introduction', 'Everyday-use creative hypothesis', 'Product-page call to action'][(day - 1) % 3]
            writer.writerow([day, state.brief['channel'], activity, 'Draft - review required'])
    state.trace.append('Automation completed: report, calendar and trace exported')
    trace = folder / 'workflow_trace.json'
    trace.write_text(json.dumps(asdict(state), indent=2), encoding='utf-8')
    return [str(report), str(calendar), str(trace)]


def run_campaign(brief=None, mode='practice', output_root=ROOT / 'outputs', model=None, retriever=None, progress=None, facts_path=None):
    brief = brief or Brief()
    brief.validate()  # Missing essential fields return to the user before agents run.
    if mode not in ('practice', 'ai'):
        raise ValueError('Mode must be practice or ai.')
    progress = progress or (lambda message: None)
    progress('Preparing model and document retrieval...')
    model = model or (LocalModel() if mode == 'ai' else PracticeModel())
    retriever = retriever or Retriever(semantic=mode == 'ai')
    state = State(asdict(brief), mode)
    progress('Planner: interpreting the campaign brief...')
    PlannerAgent().run(state, model)
    progress('Researcher: retrieving and summarizing evidence...')
    ResearchAgent().run(state, model, retriever, facts_path)
    progress('Executor: drafting the campaign...')
    facts = load_approved_facts(state.selected_evidence, facts_path or ROOT / 'data' / 'approved_facts.json')
    ExecutorAgent().run(state, model, facts)
    review_draft(state)
    progress('Exporting report, calendar and workflow trace...')
    return state, export_files(state, output_root)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--product', default=Brief.product)
    parser.add_argument('--audience', default=Brief.audience)
    parser.add_argument('--goal', default=Brief.goal)
    parser.add_argument('--channel', default=Brief.channel)
    parser.add_argument('--budget', type=float, default=200)
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--mode', choices=['practice', 'ai'], default='practice')
    args = parser.parse_args()
    try:
        state, files = run_campaign(Brief(args.product, args.audience, args.goal, args.channel, args.budget, args.days), mode=args.mode, progress=lambda message: print(message, flush=True))
    except (ValueError, ImportError, OSError) as exc:
        parser.exit(1, f'Campaign could not finish: {exc}\nFor AI mode, install requirements.txt and allow model downloads.\n')
    print('\n'.join(state.trace))
    print('\n'.join(files))

