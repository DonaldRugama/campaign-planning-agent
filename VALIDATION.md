# Validation — v1.4

17 dependency-free tests pass; 3 optional automated Gradio checks were skipped in the build environment. Python files compile. Regression tests cover repeated choices, duration-constrained planning, numeric source selection, missing approved product sources, source provenance, invalid inputs, budget arithmetic and truncated responses.

## User testing on October 11, 2026
The user ran the matching v1.4 Colab notebook in AI mode and downloaded all three output files. Three distinct captions cite approved product facts; $50 test allocation plus $150 reserve equals the $200 total across 14 days. Planner selection used a fallback; Researcher and two Executor selections passed; one repeated Executor selection used a fallback.

The user subsequently ran AI mode through the Gradio interface and downloaded a report and 14-day calendar, providing the workflow trace in chat. Planner and Researcher selections passed; two Executor selections passed; a repeated choice for the second post used a fallback. The downloaded report and calendar match that trace. These artifacts verify generation and export through the browser, not the visual progress indicator or permanent hosting.

## Scope and limits
The demonstrated brief uses a fictional TrailSip bottle, college students and Instagram. No paid API, live web research, posting, analytics collection or spending occurs. AI selects authored questions, evidence and template components; Python calculates budgets and renders quotes. The model can return invalid or repetitive choices, which are recorded as fallbacks. This is a portfolio workflow demo, not unrestricted AI copywriting or an autonomous campaign manager.
