# Module 1 — Data Pipeline

## Run

From the repository root:

```bash
python data_pipeline/pipeline.py
```

The script:
1. Scrapes the first five paginated pages of `books.toscrape.com`.
2. Captures title, price GBP, star rating text, availability and category.
3. Cleans price, rating and in-stock fields.
4. Converts GBP to INR using the fixed rate `1 GBP = 105.50 INR`.
5. Creates a normalized SQLite database with `categories` and `books`.
6. Executes at least five SQL queries covering the required SQL operations.
7. Reads two SQL results using `pd.read_sql`.
8. Reproduces a JOIN using `pd.merge`.

Outputs are written into this directory:

- `zepto_books.db`
- `query_outputs/`
- `books_clean.csv`

## Notes

The scraper uses the public training site specified in the brief. Parse failures are recorded and skipped rather than silently creating malformed rows.
