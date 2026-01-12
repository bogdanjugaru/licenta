# Licenta - Analiza si Predictia Accidentarilor Jucatorilor de Fotbal

Acest proiect ofera o aplicatie completa pentru analiza si predictia riscului de accidentare la fotbalul european. Include:

- Pipeline ML cu feature engineering si model Gradient Boosting.
- Componenta de analiza medicala/biomecanica (fatigue score, acute/chronic ratio).
- Dashboard interactiv Streamlit pentru explorare, antrenare si predictii.

## Structura proiectului

```
app/            # Aplicatia Streamlit
configs/        # Configurari YAML
data/raw/       # Date locale de exemplu
src/            # Codul sursa
```

## Instalare

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rulare aplicatie

```bash
streamlit run app/main.py
```

## Note despre date

`data/raw/sample_players.csv` include nume reale de jucatori, insa valorile statistice sunt sintetice si au rol demonstrativ.
Conectorii pentru Transfermarkt/UEFA/club sunt stubs. Inlocuieste cu exporturi licentiate sau dataset-uri aprobate.
Poti incepe cu `data/raw/sample_players.csv` sau genera date sintetice in aplicatie.
