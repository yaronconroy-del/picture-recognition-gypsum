# Evaluation & Docs

Role: tracking how well the model is doing and writing down decisions — the project's memory, not its code.

What's here:
- `Project Workflow.md` — the project overview: the idea, the dataset, and the intended workflow. Moved here from the project root.
- `filter-guide.html` — static copy of the "Filter State Field Guide" Claude artifact explaining the 5 dataset classes. Images are hosted assets, so open the live link (noted in the file) to actually see it rendered.
- `project-status.html` — static, self-contained copy of the "Gypsum Recognition Progress" status page (phase-by-phase done/to-do breakdown). Open directly in a browser.
- `Final Presentation.pptx` / `.pdf` — the course submission deck (stage 8), covering the guideline's required sections: repo link, problem statement, background/motivation (incl. the monetized business case), methodology, experiments/results, and conclusions/limitations/further directions. Built with `python-pptx`; the PDF is exported 1:1 from the PPTX via PowerPoint COM automation, so the two stay in sync — regenerate both together, never hand-edit just one.
- `context/` — empty for now. Drop in accuracy results, evaluation notes, or decision logs as you go.
