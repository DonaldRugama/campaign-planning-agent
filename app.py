"""Optional browser interface. Start with python app.py."""
import os
import threading
import tempfile
from pathlib import Path
import gradio as gr
from workflow import Brief, LocalModel, Retriever, run_campaign
from request_inputs import parse_brief, prepare_sources

lock = threading.Lock()
resources = {}

def submit(product, audience, goal, channel, budget, days, mode, documents=None, approved_quotes="", approved=False, progress=gr.Progress()):
    try:
        brief = Brief(product, audience, goal, channel, float(budget), days)
        if days is None or float(days) != int(days):
            raise ValueError('Days must be a whole number.')
        brief.days = int(days)
        brief.validate()
        progress(0, desc='Waiting for campaign resources...')
        with lock:
            progress(0.05, desc='Preparing model and documents...')
            if not documents and product.strip() != Brief.product:
                raise ValueError('For another product, upload its documents and approve its product facts. The sample documents describe TrailSip only.')
            stages = iter([0.15, 0.25, 0.4, 0.6, 0.9])
            def update(message):
                progress(next(stages, 0.9), desc=message)
            with tempfile.TemporaryDirectory(prefix='campaign_sources_') as source_folder:
                if documents:
                    retriever, facts_path = prepare_sources(documents, approved_quotes, approved,
                                                           source_folder, semantic=mode == 'ai')
                else:
                    facts_path = None
                    retriever = resources.get('retriever') if mode == 'ai' else None
                    if mode == 'ai' and retriever is None:
                        retriever = Retriever(semantic=True)
                model = resources.get('model') if mode == 'ai' else None
                if mode == 'ai' and model is None:
                    model = LocalModel()
                # Cache only the shared model and the original demo's retrieval.
                if mode == 'ai':
                    resources['model'] = model
                    if not documents:
                        resources['retriever'] = retriever
                state, files = run_campaign(brief, mode, model=model, retriever=retriever,
                                           facts_path=facts_path, progress=update)
        progress(1, desc='Campaign files ready')
        return Path(files[0]).read_text(encoding='utf-8'), '\n'.join(state.trace), files
    except Exception as exc:
        raise gr.Error(f'Campaign could not finish: {exc}') from exc

def load_request(path):
    try:
        if path is None:
            raise ValueError('Choose a JSON brief first.')
        obj = parse_brief(path)
        return tuple(obj[k] for k in ('product', 'audience', 'goal', 'channel', 'budget', 'days'))
    except Exception as exc:
        raise gr.Error(f'Could not load brief: {exc}') from exc

with gr.Blocks(title='Campaign Planning Agents') as demo:
    gr.Markdown('# Campaign Planning Agents\nVersion 1.5: three agents and a file-export automation. Practice mode uses scripted text and keyword retrieval. AI mode downloads models; all outputs are drafts.')
    request_file = gr.File(label='Optional: upload campaign brief (JSON)', file_types=['.json'], type='filepath')
    load_button = gr.Button('Load brief into form')
    product = gr.Textbox(value='TrailSip reusable water bottle', label='Product')
    audience = gr.Textbox(value='College students', label='Audience')
    goal = gr.Textbox(value='Increase product-page visits and purchases', label='Goal')
    channel = gr.Textbox(value='Instagram', label='Channel')
    budget = gr.Number(value=200, label='Budget (USD)', minimum=0)
    days = gr.Number(value=14, label='Days', minimum=1, maximum=90, precision=0)
    mode = gr.Radio(['practice', 'ai'], value='practice', label='Mode')
    gr.Markdown('### Your product and brand sources\nUpload UTF-8 .md or .txt documents for your product, brand rules and measurement notes. Leave uploads empty to use the fictional TrailSip demo. Product facts must be exact quotations from the uploaded documents. Uploads replace the demo sources for this run. Files and results are processed by the active app host; use information you are permitted to share.')
    documents = gr.File(label='Product and brand documents', file_types=['.md', '.txt'], file_count='multiple', type='filepath')
    approved_quotes = gr.Textbox(label='Approved product facts — one exact quotation per line', lines=4)
    approved = gr.Checkbox(label='I reviewed these facts and approve them for this product.', value=False)
    button = gr.Button('Generate campaign draft')
    report = gr.Markdown()
    trace = gr.Textbox(label='Agent handoffs')
    downloads = gr.File(label='Download outputs', file_count='multiple')
    load_button.click(load_request, [request_file], [product, audience, goal, channel, budget, days])
    button.click(submit, [product, audience, goal, channel, budget, days, mode, documents, approved_quotes, approved], [report, trace, downloads])

if __name__ == '__main__':
    demo.queue().launch(share=os.environ.get('GRADIO_SHARE') == '1')

