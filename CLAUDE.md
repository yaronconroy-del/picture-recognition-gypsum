# Picture Recognition — Gypsum Filter

Chief-of-staff folder for a university-level AI course final project: an image classifier that reads a gypsum belt filter's camera feed and tells whether the cake is valid or invalid (wet gypsum) — see `Evaluation & Docs/Project Workflow.md` §1.2 for the real yield-loss/downtime business case behind it.

## Areas

- [Model & Training](<Model & Training/CLAUDE.md>) — the labeled photo dataset, training code, and model experiments. Where "make the model better" happens.
- [Evaluation & Docs](<Evaluation & Docs/CLAUDE.md>) — accuracy tracking, write-ups, and project decisions.

## How this folder works

Each area folder has its own CLAUDE.md explaining its role, and an empty `context/` folder for notes you add yourself over time. Nothing important should sit loose at this root — new material belongs inside the area it relates to.

The one deliberate exception: `final project/` — a curated, submission-ready copy of the deck, the written report, the consolidated notebook, the demo scripts, and the EDA plots, assembled for zipping up and handing in. Not tracked in git (see `Evaluation & Docs/CLAUDE.md`); everything in it is a copy of something that lives properly inside one of the two area folders.

See [north-star.md](north-star.md) for the project in one page.
