# ContextShield

Streamlit prototype for claim verification with retrieved evidence.

## Run locally

Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.

The current prototype uses word TF-IDF retrieval and a saved evidence verifier. Its class scores are not calibrated probabilities, and outputs should be treated as provisional rather than definitive fact-checks.
