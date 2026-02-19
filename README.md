#  Board Game Tracker

A Python app to track board game scores, analyze player performance, and recommend games using ML.

**Team:** 3 members | **Stack:** Python · MySQL · Streamlit · scikit-learn · Plotly

---

##  Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/board-game-tracker.git
cd board-game-tracker
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your database credentials
```bash
# Copy the template and fill in your own MySQL credentials
cp .env.example .env
```
Edit `.env` with your MySQL host, username, and password.

### 4. Initialize the database
```bash
python db/init_db.py
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
board-game-tracker/
├── app.py                  # Streamlit main app (UI entry point)
├── recommender.py          # ML recommendation engine (KNN + fallback)
├── requirements.txt        # Python dependencies
├── .env.example            # DB credentials template (copy to .env)
│
├── data/
│   └── game_attributes.py  # Game feature vectors for ML
│
├── db/
│   ├── schema.sql          # MySQL table definitions
│   ├── init_db.py          # Script to create tables + load sample data
│   └── crud.py             # Database read/write functions
│
└── pages/                  # Streamlit multi-page components
    ├── data_entry.py       # Add games, players, log sessions
    ├── statistics.py       # Win rates, score charts
    └── recommendations.py  # ML-powered game suggestions
```

---

##  How the Recommendation Works

1. **Game Attributes** — Each game has a feature vector (8 dimensions):
   `strategy`, `luck`, `negotiation`, `deduction`, `deck_building`, `cooperation`, `complexity`, `duration`

2. **Player Profile** — Built from the player's history:
   weighted average of game vectors, weighted by the player's relative performance (win rate / score vs. others)

3. **ML Matching** — KNN with cosine similarity finds unplayed games closest to the player's profile

4. **Fallback** — If not enough data, uses statistical analysis:
   find games the player excels at → identify their category → recommend similar unplayed games

---

##  Team Roles

| Member | Area |
|--------|------|
| CY | ML Recommendation Engine (`recommender.py`, `data/`) |
| R | Database + Backend (`db/`, MySQL schema, CRUD) |
| yza| Frontend UI (`app.py`, `pages/`, Plotly charts) |

---

##  Git Workflow

```bash
# Before starting work
git pull origin main

# Create your own branch
git checkout -b feature/your-feature-name

# After finishing
git add .
git commit -m "describe what you did"
git push origin feature/your-feature-name

# Then open a Pull Request on GitHub to merge into main
```

---

##  Key Libraries

| Library | Purpose |
|---------|---------|
| `streamlit` | Web dashboard UI |
| `pandas` | Data analysis |
| `scikit-learn` | KNN recommendation model |
| `plotly` | Interactive charts |
| `mysql-connector-python` | MySQL database connection |
| `python-dotenv` | Load credentials from `.env` |

