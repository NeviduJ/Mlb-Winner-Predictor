---
title: Mlb Winner Predictor
emoji: 🏆
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 4.43.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Overview
This app predicts the likely winner of an MLB game mid-game using the current linescore up to a selected inning. It uses TensorFlow models specialized for innings Five through Eight and fetches live game data via `python-mlb-statsapi`.

## Repository contents
- `app.py`: Gradio UI and prediction logic
- `requirements.txt`: Python dependencies
- `Score_prediction_dataset_25th_August_TS_3seas.csv`: Reference dataset to align features/encodings
- `ANNR_ts_CLAS_inn5_exp4_model.keras`: Model for inning Five
- `ANNR_ts_CLAS_inn6_exp7_model.keras`: Model for inning Six
- `ANNR_ts_CLAS_inn7_exp11_model.keras`: Model for inning Seven
- `ANNR_ts_CLAS_inn8_exp8_model.keras`: Model for inning Eight

## Setup
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Key libraries: `gradio`, `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `python-mlb-statsapi`.

## Run locally
```bash
python app.py
```
This launches a Gradio interface in your browser.

## Using the app
- Select the inning: one of "Five", "Six", "Seven", "Eight".
- Enter `Game_ID`: MLB `gamePk` (integer). The app retrieves the linescore up to the selected inning and predicts the winner.
- Click "Predict". Output shows `Home_Team` or `Away_Team`.

### How to find a Game_ID
You can get `gamePk` via MLB Stats API or `python-mlb-statsapi`. Example:
```python
import mlbstatsapi
mlb = mlbstatsapi.Mlb()
schedule = mlb.get_schedule()
for d in schedule.dates:
    for g in d.games:
        print(g.gamePk, g.teams.home.team.name, 'vs', g.teams.away.team.name)
```

## Model selection
`app.py` loads a model based on the inning:
- Five → `ANNR_ts_CLAS_inn5_exp4_model.keras`
- Six → `ANNR_ts_CLAS_inn6_exp7_model.keras`
- Seven → `ANNR_ts_CLAS_inn7_exp11_model.keras`
- Eight → `ANNR_ts_CLAS_inn8_exp8_model.keras`

## Notes
- Internet access is required at runtime to fetch live game data.
- If inning data is not yet available for a game, the app will inform you; predictions need sufficient innings for both teams.
- The dataset is used to construct and align one-hot encoded features to match model expectations.

## Deployment (Hugging Face Spaces)
This repo contains Spaces metadata targeting `app.py` with Gradio SDK `4.43.0`. Pushing this repository to a Space will auto-run the app.
