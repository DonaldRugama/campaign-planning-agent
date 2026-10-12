"""Read user briefs and prepare isolated, user-approved document sources."""
import json
from pathlib import Path
from dataclasses import asdict
from workflow import Brief, Retriever

MAX_BRIEF_BYTES = 20000
MAX_DOCUMENT_BYTES = 200000
MAX_DOCUMENTS = 10
MAX_TOTAL_BYTES = 500000


def parse_brief(path):
    path = Path(path)
    if path.suffix.lower() != '.json' or path.stat().st_size > MAX_BRIEF_BYTES:
        raise ValueError('Upload a JSON brief smaller than 20 KB.')
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError('The brief must be a JSON object.')
    names = {'product', 'audience', 'goal', 'channel', 'budget', 'days'}
    if set(obj) != names:
        raise ValueError('Brief must contain product, audience, goal, channel, budget and days only.')
    brief = Brief(**obj)
    brief.validate()
    return asdict(brief)


def prepare_sources(paths, approved_quotes, approved, folder, semantic=False):
    """Copy text to a private folder; map approved exact quotations to chunks."""
    if not paths or not 1 <= len(paths) <= MAX_DOCUMENTS:
        raise ValueError('Upload 1 to 10 product/brand documents (.md or .txt).')
    if approved is not True:
        raise ValueError('Confirm you reviewed the product facts before generating.')
    quotes = [q.strip() for q in approved_quotes.splitlines() if q.strip()]
    if not 1 <= len(quotes) <= 20 or any(len(q) > 500 for q in quotes):
        raise ValueError('Supply 1 to 20 exact product-fact quotations, one per line, up to 500 characters each.')
    folder = Path(folder)
    data = folder / 'data'
    data.mkdir(parents=True, exist_ok=False)
    total = 0
    texts = []
    for i, name in enumerate(paths):
        source = Path(name)
        size = source.stat().st_size
        total += size
        if source.suffix.lower() not in ('.md', '.txt') or size > MAX_DOCUMENT_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError('Use .md/.txt documents up to 200 KB each and 500 KB total.')
        text = source.read_text(encoding='utf-8-sig')
        if not text.strip() or '\x00' in text:
            raise ValueError('Documents must contain readable, non-empty UTF-8 text.')
        # Generated filenames avoid collisions and cannot choose a write path.
        texts.append(text)
        (data / f'uploaded_{i+1}.md').write_text(text, encoding='utf-8')
    retriever = Retriever(data_dir=data, semantic=semantic)
    facts = []
    for quote in dict.fromkeys(quotes):
        normalized = ' '.join(quote.split())
        match = next((c for c in retriever.chunks if normalized in c['text']), None)
        if match is None:
            raise ValueError('An approved quote does not match an uploaded document passage. Copy a shorter exact sentence from your product document.')
        facts.append({'source_id': match['id'], 'quote': quote})
    facts_path = folder / 'approved_facts.json'
    facts_path.write_text(json.dumps(facts), encoding='utf-8')
    return retriever, facts_path
