"""Optional browser interface. Start with python app.py."""
import os
import threading
from pathlib import Path
import gradio as gr
from workflow import Brief, LocalModel, Retriever, run_campaign

lock = threading.Lock()
resources = {}

def submit(product, audience, goal, channel, budget, days, mode, progress=gr.Progress()):
    try:
        brief = Brief(product, audience, goal, channel, float(budget), days)
        if days is None or float(days) != int(days):
            raise ValueError('Days must be a whole number.')
        brief.days = int(days)
        brief.validate()
        progress(0, desc='Waiting for campaign resources...')
        with lock:
            progress(0.05, desc='Preparing model and documents...')
            if mode == 'ai' and not {'model', 'retriever'}.issubset(resources):
                model = LocalModel()
                retriever = Retriever(semantic=True)
                resources.update(model=model, retriever=retriever)
            stages = iter([0.15, 0.25, 0.4, 0.6, 0.9])
            def update(message):
                progress(next(stages, 0.9), desc=message)
            state, files = run_campaign(brief, mode, model=resources.get('model') if mode == 'ai' else None,
                                       retriever=resources.get('retriever') if mode == 'ai' else None, progress=update)
        progress(1, desc='Campaign files ready')
        return Path(files[0]).read_text(encoding='utf-8'), '\n'.join(state.trace), files
    except Exception as exc:
        raise gr.Error(f'Campaign could not finish: {exc}') from exc

with gr.Blocks(title='Campaign Planning Agents') as demo:
    gr.Markdown('# Campaign Planning Agents\nVersion 1.4: three agents and a file-export automation. Practice mode uses scripted text and keyword retrieval. AI mode downloads models; all outputs are drafts.')
    product = gr.Textbox(value='TrailSip reusable water bottle', label='Product')
    audience = gr.Textbox(value='College students', label='Audience')
    goal = gr.Textbox(value='Increase product-page visits and purchases', label='Goal')
    channel = gr.Textbox(value='Instagram', label='Channel')
    budget = gr.Number(value=200, label='Budget (USD)', minimum=0)
    days = gr.Number(value=14, label='Days', minimum=1, maximum=90, precision=0)
    mode = gr.Radio(['practice', 'ai'], value='practice', label='Mode')
    button = gr.Button('Generate campaign draft')
    report = gr.Markdown()
    trace = gr.Textbox(label='Agent handoffs')
    downloads = gr.File(label='Download outputs', file_count='multiple')
    button.click(submit, [product, audience, goal, channel, budget, days, mode], [report, trace, downloads])

if __name__ == '__main__':
    demo.queue().launch(share=os.environ.get('GRADIO_SHARE') == '1')
