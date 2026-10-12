# Beginner study guide - v1.4
Prepared for Donald Rugama.

## Concepts
An agent is a role with instructions and relevant context. Our three roles share one language model. The Planner selects questions; the Researcher retrieves evidence and selects numeric evidence indices; the Executor chooses components for exactly three posts. Orchestration is the Python code controlling their order. Shared state is their common notebook: brief, questions, evidence, research, posts, warnings and trace.

The LLM generates short JSON selections. A prompt describes the role and allowed output. Tokenization turns text into token IDs; an attention mask tells the model which input positions are meaningful. Embeddings turn passages into numerical representations of meaning; normalized vectors support cosine similarity via a dot product. RAG retrieves relevant documents and supplies them to the agents. Source IDs connect quotations to retrieved passages.

Schema validation checks the response's structure and allowed IDs. Grounding checks approved quotations exist in retrieved text. Neither proves that the source is true. A fallback is authored Python logic used when AI output is invalid or incomplete; the trace labels it honestly. Automation writes the report, CSV calendar and JSON trace, but does not post or spend.

## Code walkthrough
Brief stores campaign settings and rejects invalid inputs. State holds handoffs and generation_events. PracticeModel returns scripted JSON. LocalModel imports torch and Transformers, loads the tokenizer/model, requests a dictionary with input_ids and attention_mask, and passes both using **inputs. It generates a short bounded response, checks whether an EOS token ended it, and discards output that hits the token limit without EOS.

Retriever reads Markdown files into 120-word passages with a 90-word step. AI mode imports SentenceTransformer, embeds the passages and query, and ranks their similarity. Keyword practice retrieval is a separate method.

structured_call parses JSON and validates it with the relevant role's validator. Invalid JSON, unknown IDs, wrong types, or truncation become labeled fallback events. Download errors are not silently converted into AI success.

PlannerAgent asks the model to order three authored document questions built from the brief. ResearchAgent retrieves passages, asks the model to select existing numeric indices, then displays the selected text verbatim. load_approved_facts checks human-approved product quotes against selected evidence. ExecutorAgent makes three separate component-selection calls and builds exactly three distinct caption objects, rejecting repeated hooks or calls to action. The model cannot insert arbitrary prose into those captions. Python renders the strategy template, 25% test allocation, remaining reserve, total budget and metric formulas. Integer cents prevent floating-point budget sum errors.

run_campaign performs validation, creates or reuses models, calls all three agents, checks post count, and exports. export_files writes a complete report, a day-by-day Python calendar and the state trace. app.py collects the brief, caches model/retriever together, serializes requests with a lock and displays errors. The browser server stays running after campaign completion so its link remains accessible.

## How to repurpose
For a different store, replace Markdown documents and approved_facts.json. Each entry needs source_id (for example product_catalog.md:1) and quote (an exact sentence from that retrieved passage). Keep the quotes short and review them. The quote must be retrieved or the workflow stops.

Change product, audience, goal, channel, budget and days in Brief, the interface or terminal flags. Edit HOOKS and CTAS for tone; edit PROMPTS for selection behavior. Edit TOKEN_LIMITS cautiously: a too-small limit causes labeled fallbacks. Change export_files to produce other files.

A study assistant would need different template components and approved study facts; you would also update Brief, prompts, rendering and tests. Changing prompts alone does not change the Python-controlled report or calendar. Changing the embedding model requires rebuilding embeddings.

## Reading your result
The report says mode practice or ai and version 1.4. Three caption objects are present even if some selections fall back. In workflow_trace.json, status accepted means structurally valid JSON selections; it does not prove good marketing judgment. Status fallback means Python substituted approved components. Read warnings before sharing any campaign.

CTR = link clicks / impressions. Purchase conversion = purchases / product-page visits. ROAS = attributed revenue / ad spend. Zero denominators make rates undefined. We do not calculate customer lifetime value or predict sales growth.

## All setup and imports
The notebook installs torch, transformers, sentence-transformers, gradio and httpx[socks]. Every Python import is included in the source. Model-related imports happen inside their classes, so practice mode needs only Python's standard library. Install commands and Windows environment activation remain the same as the previous version.

## v1.5 uploads
request_inputs.py validates a JSON brief and copies uploaded text documents into a per-request temporary folder with generated filenames. It maps human-approved exact quotations to document chunks. app.py creates that request's retriever and facts file, calls the same run_campaign coordinator, then removes its temporary copies. The shared cache contains the language model and original demo retriever; custom document retrieval is never reused between users. Gradio separately manages uploads and exported downloads. To support PDF or Word later, add text extraction and test it before passing text to this pipeline.
