from psycopg_pool import ConnectionPool
from flask import Flask

# Create the pool once at startup, shared across all requests
pool = ConnectionPool(
    conninfo="host=db dbname=postgres user=postgres password=postgres",
    min_size=2,  # connections kept alive in the pool
    max_size=10,  # max concurrent connections
)
# atexit.register(pool.close)

app = Flask()


@app.route("/")
def get_data():
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM apparel")
            data = cur.fetchall()
        output = []
        for item in data:
            output.append({"name": item[0]})

    return output
