# v1.5 — Custom campaign uploads
Upload a structured JSON brief and product/brand documents through Gradio. Exact user-approved facts are matched to uploaded passages, then passed through the existing multi-agent workflow. Each request uses its own sources and retriever. A new sample template and GitHub-loading Colab notebook are included.

28 tests pass, including Gradio callbacks, upload validation and request isolation. AI model behavior and the Colab upload UI still require a new end-to-end user run. PDF, Word and freeform request interpretation are not included.
