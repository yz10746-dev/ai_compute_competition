# AI Compute Competition by 2030

Yuehua Zhuang

Plotly Dash dashboard for Epoch AI data centers and AI models, snapshot 23 Nov 2026.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m dashboard.app
```

Open http://127.0.0.1:8060/

Python 3.11. Dependencies are listed in `requirements.txt`. The CSVs are already in `data/epoch/`.

## Views

The left card sets the company set and the year. Capacity is the top-right card. Models are the bottom-right card.

- Drag horizontally on the time series, or click a year in either table, to change the year in every view.
- Hover a company to highlight it in the other views.
- Click a company once to keep only that company. Click it again to restore all nine.
- Count as Owner or User. OpenAI, Anthropic, and xAI have no owned campuses, so Owner shows them empty.

## Data

Epoch AI, [AI data centers](https://epoch.ai/data/ai-data-centers) and [AI models](https://epoch.ai/data/ai-models). Speculative campuses are excluded. Rank is the mean of the within-tier percentiles for training compute, parameters, and training data, using only the metrics a model reports.
