import sqlite3
import random
from datetime import datetime, timedelta

DB_NAME = "sales.db"

def create_tables(cursor):
    # Products Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
    """)

    # Customers Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        country TEXT NOT NULL,
        signup_date DATE NOT NULL
    )
    """)

    # Orders Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        date DATE NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers (id)
    )
    """)

    # Order Items Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        amount REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    """)

def seed_data(cursor):
    # 1. Seed Products
    categories = {
        "Protein": ["Whey Isolate", "Casein", "Vegan Pea Protein", "Mass Gainer", "Protein Bar Box"],
        "Vitamins": ["Multivitamin Men", "Multivitamin Women", "Vitamin D3", "Omega 3 Fish Oil", "Magnesium"],
        "Pre-Workout": ["High Stim Pre", "Pump Non-Stim", "Creatine Monohydrate", "Beta Alanine"],
        "Gear": ["Shaker Bottle", "Lifting Belt", "Wrist Wraps", "Gym Bag"]
    }
    
    products = []
    for cat, items in categories.items():
        for item in items:
            price = round(random.uniform(15.0, 80.0), 2)
            if cat == "Gear" and "Bottle" in item: price = 10.0
            cursor.execute("INSERT INTO products (name, category, price) VALUES (?, ?, ?)", (item, cat, price))
            products.append((cursor.lastrowid, price)) # Store ID and Price

    print(f"Seeded {len(products)} products.")

    # 2. Seed Customers
    countries = ["USA", "Canada", "UK", "Germany", "Australia", "France"]
    first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Katie", "Michael", "Sarah", "David", "Laura"]
    last_names = ["Smith", "Doe", "Johnson", "Brown", "Williams", "Jones", "Miller", "Davis", "Garcia", "Wilson"]
    
    customer_ids = []
    for _ in range(200): # 200 Customers
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        country = random.choice(countries)
        start_date = datetime.now() - timedelta(days=730) # 2 years ago
        signup_date = start_date + timedelta(days=random.randint(0, 700))
        
        cursor.execute("INSERT INTO customers (name, country, signup_date) VALUES (?, ?, ?)", 
                       (name, country, signup_date.strftime("%Y-%m-%d")))
        customer_ids.append(cursor.lastrowid)
        
    print(f"Seeded {len(customer_ids)} customers.")

    # 3. Seed Orders and Items
    statuses = ["Completed", "Completed", "Completed", "Shipped", "Processing", "Returned"]
    
    order_count = 0
    item_count = 0
    
    start_date = datetime.now() - timedelta(days=365) # 1 year of orders
    
    for _ in range(1500): # 1500 Orders
        customer_id = random.choice(customer_ids)
        order_date = start_date + timedelta(days=random.randint(0, 365))
        status = random.choice(statuses)
        
        cursor.execute("INSERT INTO orders (customer_id, date, status) VALUES (?, ?, ?)", 
                       (customer_id, order_date.strftime("%Y-%m-%d"), status))
        order_id = cursor.lastrowid
        order_count += 1
        
        # Create 1-5 items per order
        num_items = random.randint(1, 5)
        for _ in range(num_items):
            prod_id, prod_price = random.choice(products)
            qty = random.randint(1, 3)
            amount = round(prod_price * qty, 2)
            
            cursor.execute("INSERT INTO order_items (order_id, product_id, quantity, amount) VALUES (?, ?, ?, ?)",
                           (order_id, prod_id, qty, amount))
            item_count += 1

    print(f"Seeded {order_count} orders with {item_count} line items.")

def main():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Drop tables to reset
    cursor.execute("DROP TABLE IF EXISTS order_items")
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("DROP TABLE IF EXISTS products")
    
    create_tables(cursor)
    seed_data(cursor)
    
    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} created successfully.")

if __name__ == "__main__":
    main()
