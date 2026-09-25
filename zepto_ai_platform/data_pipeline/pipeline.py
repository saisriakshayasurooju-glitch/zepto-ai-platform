from pathlib import Path
import sqlite3
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/catalogue/"
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "zepto_books.db"
CSV_PATH = ROOT / "books_clean.csv"
OUT_DIR = ROOT / "query_outputs"
RATE_GBP_INR = 105.50

HEADERS = {"User-Agent": "Mozilla/5.0 (training scraper; capstone project)"}


def parse_rating(text):
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    if not text:
        return None
    return mapping.get(text.strip())


def parse_price(text):
    if not text:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None


def parse_stock(text):
    return bool(text and "in stock" in text.lower())


def scrape():
    rows = []
    failures = []

    for page in range(1, 6):
        url = f"{BASE}page-{page}.html"
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:
            failures.append({"page": page, "error": str(exc)})
            continue

        for article in soup.select("article.product_pod"):
            try:
                title_tag = article.select_one("h3 a")
                price_tag = article.select_one(".price_color")
                rating_tag = article.select_one("p.star-rating")
                availability_tag = article.select_one(".availability")

                title = title_tag.get("title", "").strip() if title_tag else ""
                rating_text = ""
                if rating_tag:
                    classes = rating_tag.get("class", [])
                    rating_text = next((x for x in classes if x != "star-rating"), "")

                price_gbp = parse_price(price_tag.get_text(" ", strip=True) if price_tag else "")
                in_stock = parse_stock(availability_tag.get_text(" ", strip=True) if availability_tag else "")

                # Category is obtained from the product detail page because
                # the listing card itself does not expose the category.
                href = title_tag.get("href") if title_tag else None
                category = "Unknown"
                if href:
                    detail_url = BASE + href.replace("../", "")
                    try:
                        d = requests.get(detail_url, headers=HEADERS, timeout=20)
                        d.raise_for_status()
                        ds = BeautifulSoup(d.text, "html.parser")
                        breadcrumb = ds.select("ul.breadcrumb li a")
                        # Product pages contain breadcrumb links:
                        # Home > Books > Category > Product.
                        if len(breadcrumb) >= 3:
                            category = breadcrumb[-1].get_text(" ", strip=True)
                    except Exception as exc:
                        failures.append({"title": title, "error": f"category: {exc}"})

                if not title or price_gbp is None or rating_text == "":
                    failures.append({"title": title, "error": "required field parse failure"})
                    continue

                rows.append({
                    "title": title,
                    "price_gbp": price_gbp,
                    "price_inr": round(price_gbp * RATE_GBP_INR, 2),
                    "rating": parse_rating(rating_text),
                    "rating_text": rating_text,
                    "in_stock": in_stock,
                    "category": category,
                })
            except Exception as exc:
                failures.append({"title": "", "error": str(exc)})

        time.sleep(0.2)

    df = pd.DataFrame(rows).drop_duplicates(subset=["title"]).reset_index(drop=True)
    if len(df) < 60:
        raise RuntimeError(f"Only {len(df)} valid books were scraped; at least 60 are required.")

    if df["category"].nunique() < 3:
        raise RuntimeError("Fewer than 3 categories were captured.")

    return df, failures


def build_database(df):
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL UNIQUE,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER,
        rating_text TEXT,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    for category in sorted(df["category"].dropna().unique()):
        cur.execute("INSERT INTO categories(name) VALUES (?)", (category,))

    category_map = {
        name: cid for cid, name in cur.execute("SELECT category_id, name FROM categories")
    }

    records = []
    for row in df.itertuples(index=False):
        records.append((
            row.title,
            float(row.price_gbp),
            float(row.price_inr),
            int(row.rating) if pd.notna(row.rating) else None,
            row.rating_text,
            int(bool(row.in_stock)),
            category_map[row.category],
        ))

    cur.executemany("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, rating_text, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    return conn


def save_query_results(conn):
    OUT_DIR.mkdir(exist_ok=True)
    queries = {
        "q1_where": """SELECT title, price_gbp, rating FROM books WHERE price_gbp > 20 ORDER BY price_gbp DESC;""",
        "q2_order_limit": """SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;""",
        "q3_distinct": """SELECT DISTINCT rating FROM books ORDER BY rating;""",
        "q4_in": """SELECT title, price_gbp, category_id FROM books WHERE rating IN (4,5) ORDER BY price_gbp DESC;""",
        "q5_between": """SELECT title, price_gbp, rating FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp;""",
        "q6_join": """SELECT b.title, b.price_gbp, b.rating, c.name AS category
                     FROM books b JOIN categories c ON b.category_id = c.category_id
                     ORDER BY b.price_gbp DESC LIMIT 15;"""
    }

    for name, sql in queries.items():
        pd.read_sql_query(sql, conn).to_csv(OUT_DIR / f"{name}.csv", index=False)

    # Requirement: read at least two query results using pd.read_sql.
    sql_a = "SELECT * FROM books WHERE rating >= 4 ORDER BY price_gbp DESC LIMIT 10"
    sql_b = """SELECT b.title, b.price_gbp, c.name AS category
               FROM books b JOIN categories c ON b.category_id = c.category_id
               ORDER BY b.title"""
    books_a = pd.read_sql(sql_a, conn)
    books_b = pd.read_sql(sql_b, conn)

    # Reproduce the JOIN with pandas merge.
    books_raw = pd.read_sql("SELECT * FROM books", conn)
    cats_raw = pd.read_sql("SELECT * FROM categories", conn)
    merged = books_raw.merge(
        cats_raw,
        left_on="category_id",
        right_on="category_id",
        suffixes=("_book", "_category")
    )
    merged_result = merged[["title", "price_gbp", "rating", "name"]].rename(
        columns={"name": "category"}
    ).sort_values("price_gbp", ascending=False).head(15)

    books_b_sorted = books_b.sort_values("price_gbp", ascending=False).head(15).reset_index(drop=True)
    merged_result = merged_result.reset_index(drop=True)

    print("\nJOIN equivalence check:", books_b_sorted.equals(merged_result))

    return queries


def main():
    print("Scraping books.toscrape.com ...")
    df, failures = scrape()
    print(f"Valid books: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    if failures:
        print(f"Parse/request issues recorded: {len(failures)}")

    df.to_csv(CSV_PATH, index=False)
    conn = build_database(df)
    queries = save_query_results(conn)

    print(f"\nDatabase: {DB_PATH}")
    print(f"CSV: {CSV_PATH}")
    print("Query outputs:", OUT_DIR)
    print("\nSQL query names:")
    for name in queries:
        print(" -", name)

    conn.close()


if __name__ == "__main__":
    main()
