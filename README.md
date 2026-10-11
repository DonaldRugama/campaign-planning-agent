# Campaign Planning Agents - v1.4
A reusable Python workflow for a fictional e-commerce campaign: Planner -> Researcher -> Executor -> file exports.

## What changed
AI calls return short JSON selections. Python rejects malformed responses, unknown IDs and token-limit truncation. A labeled fallback completes the draft when a selection fails. Research displays verbatim source passages. Exactly three posts are assembled from authored hooks, human-approved document quotations and calls to action. Python calculates the total budget allocation and renders fixed metric formulas. This is constrained AI-assisted generation: the LLM selects components rather than freely writing the final report. A fallback is not claimed as model success.

## Google Colab
[Open the GitHub notebook in Colab](https://colab.research.google.com/github/DonaldRugama/campaign-planning-agent./blob/main/Campaign_Agents_GitHub_Colab_v1_4.ipynb). This version clones the public repository automatically; run its cells in order. The original upload-based notebook below is retained for the separately distributed project ZIP.

Unzip Campaign_Planning_Agents_v1_4_2026-10-11.zip and upload Campaign_Agents_Colab_v1_4.ipynb to a fresh Colab session. Follow its cells, uploading the matching v1.4 ZIP when asked. It verifies VERSION.txt, runs tests, then offers practice mode, AI mode, output download and optional browser sharing. The browser cell stays active to host the app. Runtime resources and public tunnels are not guaranteed. Colab-preinstalled diffusers may conflict with our Transformers setup; see the notebook note for a dedicated-session workaround.

## Terminal (Python 3.12 or newer)
Clone this repository, or download and unzip it using GitHub's Code menu, then open a terminal in the folder containing workflow.py.
```bash
python workflow.py --mode practice
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python workflow.py --mode ai
python app.py
```
Practice mode uses scripted selections and keyword retrieval. AI mode downloads Qwen2.5-0.5B-Instruct and all-MiniLM-L6-v2; inference uses your runtime and no paid API key. CPU runs may be slow. A GPU is used if available. GitHub stores the code; it does not host the Python runtime.

## Files and outputs
workflow.py is the engine; app.py is the Gradio interface; data/ contains fictional documents and approved_facts.json; STUDY_GUIDE.md explains concepts and reuse. Each run exports campaign_report.md, content_calendar.csv and workflow_trace.json in a unique outputs folder. The trace records version 1.4, each call's acceptance/fallback and selected post components. Reports are drafts; no posting, spending or live research occurs.

## Customize
Change the brief or terminal flags; replace permitted Markdown documents; update data/approved_facts.json with exact product-fact quotations and matching retrieved chunk IDs. Facts must match retrieved evidence or export stops. Edit HOOKS, CTAS, PROMPTS and TOKEN_LIMITS to change the style and AI tasks. Update tests with behavior changes. No vector database is needed for this small source collection.

## Limits and verification
Quote matching proves provenance, not truth. Approved facts are a human trust boundary. Questions are authored templates; distinct captions can still be formulaic. The calendar is generated in Python. Validation: 17 dependency-free tests pass. The user completed v1.4 AI runs in Colab and through Gradio, downloading reports, calendars and traces. Both runs produced three distinct cited captions and a correct $200 / 14-day allocation. In the browser run, Planner and Researcher selections were accepted; two Executor selections were accepted and one repeated choice used a labeled fallback. This establishes the demonstrated sample flow, not reliability for all briefs. Three optional automated Gradio integration tests were skipped in the build environment; the progress display has not been independently verified. See VALIDATION.md.

## Sources
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
https://www.gradio.app/guides/sharing-your-app
Models and packages retain their own licenses. Original code and fictional sample materials are MIT licensed.

## v1.4 behavior
The Planner orders three authored questions built from the supplied product, audience, channel, goal and duration. It does not write unrestricted questions. The Researcher selects numeric evidence indices; Python maps them to source IDs and requires an approved product source. The Executor must choose unused hook and CTA indices. Repeated selections trigger a labeled Python fallback; this provides distinct wording, not three independently researched creative strategies. Approved facts must match the selected evidence passed between agents. Gradio displays coarse stage progress, including resource preparation; percentages are stage indicators, not time estimates.

Use a fresh Colab session with the matching v1.4 ZIP and notebook. The hosting cell stays active while the demo is available. If the public page is blank, try a private browser window or a hard refresh. Downloads remain available from the notebook even if the public tunnel fails.
