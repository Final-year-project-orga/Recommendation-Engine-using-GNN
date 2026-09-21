import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def get_product(item_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            item_id,
            name,
            category,
            brand,
            price,
            stock,
            rating
        FROM demo_products
        WHERE item_id = %s
    """, (item_id,))

    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return None

    return {
        "item_id": row[0],
        "name": row[1],
        "category": row[2],
        "brand": row[3],
        "price": float(row[4]),
        "stock": row[5],
        "rating": float(row[6])
    }


def get_available_items(item_ids):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            item_id,
            name,
            category,
            brand,
            price,
            stock,
            rating
        FROM products
        WHERE item_id = ANY(%s)
        AND stock > 0
    """, (item_ids,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    products = {}

    for row in rows:

        products[row[0]] = {
            "item_id": row[0],
            "name": row[1],
            "category": row[2],
            "brand": row[3],
            "price": float(row[4]),
            "stock": row[5],
            "rating": float(row[6])
        }

    return products
