# Project website

Live URL: https://zrchen03.github.io/SnapBench/

GitHub Pages publishes the `main` branch's `/docs` directory. The `.nojekyll` marker serves the static files directly. Existing DATA.md, EVAL.md, and MOOR.md remain in this directory.

## Update the website

Edit `index.html`, `style.css`, and `app.js`, then commit to `main`. Data is supplied in both `data.json` and `data.js` (for offline/local opening); keep these in sync. Relative asset URLs support the `/SnapBench/` project path.

Image demos use the official repository corruption generator, not browser approximations. The two released flower queries have 15 image operations × 3 severity levels each. Their text variants come directly from released metadata. Generator commit, parameters, seeds, file hashes, and environment details are recorded in `generation-manifest.json`. These newly generated examples are separate from the paper's aggregate model results.

Logos and research figure/image provenance are recorded in `asset-sources.json`.
