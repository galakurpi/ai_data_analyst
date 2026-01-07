"""
Rich Database Seed for AI Data Analyst
Creates realistic data with patterns for testing:
- RFM Analysis (customer segments: Champions, At-Risk, etc.)
- Trend Analysis (seasonal patterns, growth)
- Product Performance (Pareto distribution - 80/20)
- Customer Clustering (distinct behavioral groups)
- Cohort Analysis (retention patterns)
- Geographic Analysis (country variations)
"""

import sqlite3
import random
from datetime import datetime, timedelta
import math

DB_NAME = "sales.db"

# Seed for reproducibility
random.seed(42)


def create_tables(cursor):
    # Products Table (extended with cost and launch date)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        cost REAL NOT NULL,
        launch_date DATE NOT NULL
    )
    """)

    # Customers Table (extended with segment, channel, age group)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        country TEXT NOT NULL,
        city TEXT,
        age_group TEXT NOT NULL,
        acquisition_channel TEXT NOT NULL,
        signup_date DATE NOT NULL
    )
    """)

    # Orders Table (extended with discount and shipping)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        date DATE NOT NULL,
        status TEXT NOT NULL,
        discount_pct REAL DEFAULT 0,
        shipping_cost REAL DEFAULT 0,
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
        unit_price REAL NOT NULL,
        amount REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    """)


def seed_products(cursor):
    """
    Create products with Pareto distribution (20% generate 80% of sales).
    Star products get more orders later.
    """
    products_data = [
        # STAR PRODUCTS (will get ~60% of sales) - Protein category dominates
        ("Whey Protein Isolate", "Protein", 59.99, 25.00, "2022-01-01"),
        ("Whey Protein Concentrate", "Protein", 44.99, 18.00, "2022-01-01"),
        ("Creatine Monohydrate 500g", "Pre-Workout", 29.99, 8.00, "2022-01-01"),
        ("BCAA Powder", "Amino Acids", 34.99, 12.00, "2022-03-01"),
        ("Pre-Workout Extreme", "Pre-Workout", 39.99, 14.00, "2022-02-01"),
        
        # GOOD PERFORMERS (will get ~25% of sales)
        ("Casein Protein", "Protein", 54.99, 22.00, "2022-01-01"),
        ("Mass Gainer 5lb", "Protein", 49.99, 20.00, "2022-04-01"),
        ("Omega 3 Fish Oil", "Vitamins", 24.99, 8.00, "2022-01-01"),
        ("Multivitamin Men", "Vitamins", 19.99, 6.00, "2022-01-01"),
        ("Multivitamin Women", "Vitamins", 19.99, 6.00, "2022-01-01"),
        ("Vitamin D3 5000IU", "Vitamins", 14.99, 4.00, "2022-05-01"),
        ("ZMA Sleep Support", "Vitamins", 17.99, 5.00, "2022-06-01"),
        ("Glutamine Powder", "Amino Acids", 27.99, 10.00, "2022-03-01"),
        ("EAA Complex", "Amino Acids", 32.99, 11.00, "2022-07-01"),
        
        # SLOW MOVERS (will get ~15% of sales)
        ("Vegan Protein Blend", "Protein", 52.99, 24.00, "2022-08-01"),
        ("Collagen Peptides", "Protein", 39.99, 18.00, "2023-01-01"),
        ("Beta Alanine", "Pre-Workout", 22.99, 7.00, "2022-01-01"),
        ("Citrulline Malate", "Pre-Workout", 26.99, 9.00, "2022-09-01"),
        ("Magnesium Glycinate", "Vitamins", 16.99, 5.00, "2022-01-01"),
        ("Zinc Picolinate", "Vitamins", 12.99, 3.00, "2022-10-01"),
        ("Shaker Bottle Pro", "Accessories", 12.99, 4.00, "2022-01-01"),
        ("Gym Bag Deluxe", "Accessories", 34.99, 15.00, "2022-01-01"),
        ("Lifting Belt", "Accessories", 44.99, 20.00, "2022-11-01"),
        ("Wrist Wraps", "Accessories", 14.99, 5.00, "2022-01-01"),
        ("Resistance Bands Set", "Accessories", 24.99, 8.00, "2023-02-01"),
    ]
    
    product_ids = {}
    for name, category, price, cost, launch in products_data:
        cursor.execute(
            "INSERT INTO products (name, category, price, cost, launch_date) VALUES (?, ?, ?, ?, ?)",
            (name, category, price, cost, launch)
        )
        product_ids[name] = (cursor.lastrowid, price)
    
    print(f"Seeded {len(products_data)} products.")
    return product_ids


def seed_customers(cursor):
    """
    Create customers with distinct behavioral segments.
    Build in realistic patterns for clustering and RFM.
    """
    # Countries with market share weights
    countries = {
        "USA": {"weight": 0.40, "cities": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]},
        "UK": {"weight": 0.20, "cities": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"]},
        "Canada": {"weight": 0.15, "cities": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"]},
        "Germany": {"weight": 0.12, "cities": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"]},
        "Australia": {"weight": 0.08, "cities": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]},
        "France": {"weight": 0.05, "cities": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"]},
    }
    
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    age_weights = [0.20, 0.35, 0.25, 0.15, 0.05]  # Fitness products skew younger
    
    channels = ["Organic Search", "Paid Ads", "Social Media", "Referral", "Email", "Direct"]
    channel_weights = [0.25, 0.30, 0.20, 0.10, 0.10, 0.05]
    
    first_names = [
        "James", "Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Sophia", "William", "Isabella",
        "Elijah", "Mia", "Lucas", "Charlotte", "Mason", "Amelia", "Ethan", "Harper", "Alexander", "Evelyn",
        "Daniel", "Abigail", "Matthew", "Emily", "Henry", "Elizabeth", "Sebastian", "Sofia", "Jack", "Avery"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White"
    ]
    
    customer_data = []
    
    # Create 500 customers with varied signup dates over 2 years
    base_date = datetime.now() - timedelta(days=730)
    
    for i in range(500):
        # Select country based on weights
        country = random.choices(list(countries.keys()), weights=[c["weight"] for c in countries.values()])[0]
        city = random.choice(countries[country]["cities"])
        
        age_group = random.choices(age_groups, weights=age_weights)[0]
        channel = random.choices(channels, weights=channel_weights)[0]
        
        first = random.choice(first_names)
        last = random.choice(last_names)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{random.randint(1, 999)}@email.com"
        
        # Signup dates: more recent customers for cohort analysis
        # Earlier cohorts have better retention (they're still here)
        days_ago = random.randint(0, 700)
        signup_date = base_date + timedelta(days=days_ago)
        
        cursor.execute(
            """INSERT INTO customers (name, email, country, city, age_group, acquisition_channel, signup_date) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, email, country, city, age_group, channel, signup_date.strftime("%Y-%m-%d"))
        )
        
        customer_data.append({
            "id": cursor.lastrowid,
            "country": country,
            "age_group": age_group,
            "channel": channel,
            "signup_date": signup_date
        })
    
    print(f"Seeded {len(customer_data)} customers.")
    return customer_data


def seed_orders(cursor, customers, product_ids):
    """
    Create orders with realistic patterns:
    - Seasonal variation (Q4 peak, January spike, summer dip)
    - Customer segments (Champions order frequently, At-Risk stopped)
    - Product popularity (Pareto distribution)
    - Geographic patterns
    """
    
    # Star products get more weight
    star_products = ["Whey Protein Isolate", "Whey Protein Concentrate", "Creatine Monohydrate 500g", 
                     "BCAA Powder", "Pre-Workout Extreme"]
    good_products = ["Casein Protein", "Mass Gainer 5lb", "Omega 3 Fish Oil", "Multivitamin Men",
                     "Multivitamin Women", "Vitamin D3 5000IU", "ZMA Sleep Support", "Glutamine Powder", "EAA Complex"]
    
    # Build product selection weights (Pareto)
    product_weights = {}
    for name, (pid, price) in product_ids.items():
        if name in star_products:
            product_weights[name] = 5.0  # 5x more likely
        elif name in good_products:
            product_weights[name] = 2.0  # 2x more likely
        else:
            product_weights[name] = 1.0
    
    statuses = ["Completed", "Completed", "Completed", "Completed", "Shipped", "Processing", "Cancelled", "Returned"]
    
    # Seasonal multipliers by month (1 = January)
    seasonal_multipliers = {
        1: 1.4,   # New Year resolutions
        2: 1.2,
        3: 1.1,
        4: 1.0,
        5: 0.9,
        6: 0.85,  # Summer dip
        7: 0.85,
        8: 0.9,
        9: 1.0,   # Back to routine
        10: 1.1,
        11: 1.3,  # Black Friday
        12: 1.5,  # Holiday gifts + year-end
    }
    
    # Assign customer behavior types
    customer_segments = {}
    for cust in customers:
        cid = cust["id"]
        signup_days_ago = (datetime.now() - cust["signup_date"]).days
        
        # Segment assignment based on signup recency and random factor
        r = random.random()
        if signup_days_ago > 500 and r < 0.3:
            customer_segments[cid] = "champion"  # Old, still active
        elif signup_days_ago > 300 and r < 0.5:
            customer_segments[cid] = "loyal"
        elif signup_days_ago > 180 and r < 0.4:
            customer_segments[cid] = "at_risk"  # Haven't ordered recently
        elif signup_days_ago < 90:
            customer_segments[cid] = "new"
        elif r < 0.2:
            customer_segments[cid] = "lost"  # Stopped ordering
        else:
            customer_segments[cid] = "regular"
    
    order_count = 0
    item_count = 0
    
    # Generate orders
    for cust in customers:
        cid = cust["id"]
        segment = customer_segments[cid]
        signup_date = cust["signup_date"]
        country = cust["country"]
        
        # Number of orders based on segment
        if segment == "champion":
            num_orders = random.randint(15, 30)
            recency_cap = 30  # Ordered within last 30 days
        elif segment == "loyal":
            num_orders = random.randint(8, 15)
            recency_cap = 60
        elif segment == "at_risk":
            num_orders = random.randint(3, 8)
            recency_cap = 180  # Last order was 3-6 months ago
        elif segment == "new":
            num_orders = random.randint(1, 3)
            recency_cap = 30
        elif segment == "lost":
            num_orders = random.randint(1, 3)
            recency_cap = 365  # Haven't ordered in a year
        else:  # regular
            num_orders = random.randint(3, 8)
            recency_cap = 90
        
        # Generate order dates after signup
        days_since_signup = (datetime.now() - signup_date).days
        
        for _ in range(num_orders):
            # Order must be after signup but consider recency based on segment
            if segment in ["at_risk", "lost"]:
                # These customers stopped ordering - last order was a while ago
                min_days_ago = recency_cap
                max_days_ago = min(days_since_signup, 500)
                if min_days_ago >= max_days_ago:
                    continue
                days_ago = random.randint(min_days_ago, max_days_ago)
            else:
                # Active customers - orders spread out, recent ones possible
                max_days = min(days_since_signup, 365)
                days_ago = random.randint(0, max_days)
            
            order_date = datetime.now() - timedelta(days=days_ago)
            
            # Skip if before signup
            if order_date < signup_date:
                continue
            
            # Apply seasonal multiplier (affects if order happens)
            month = order_date.month
            if random.random() > seasonal_multipliers[month] * 0.7:
                continue
            
            status = random.choice(statuses)
            
            # Champions get discounts, new customers might get welcome discount
            discount = 0
            if segment == "champion" and random.random() < 0.3:
                discount = random.choice([5, 10, 15])
            elif segment == "new" and random.random() < 0.5:
                discount = 10  # Welcome discount
            
            # Shipping cost based on country
            shipping = 5.99 if country == "USA" else 9.99 if country in ["UK", "Canada"] else 12.99
            if random.random() < 0.2:
                shipping = 0  # Free shipping promo
            
            cursor.execute(
                """INSERT INTO orders (customer_id, date, status, discount_pct, shipping_cost) 
                   VALUES (?, ?, ?, ?, ?)""",
                (cid, order_date.strftime("%Y-%m-%d"), status, discount, shipping)
            )
            order_id = cursor.lastrowid
            order_count += 1
            
            # Order items (1-4 items per order)
            num_items = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.2, 0.05])[0]
            
            # Select products with weights
            product_names = list(product_weights.keys())
            weights = list(product_weights.values())
            
            selected_products = random.choices(product_names, weights=weights, k=num_items)
            
            for prod_name in selected_products:
                prod_id, unit_price = product_ids[prod_name]
                qty = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                
                # Champions might buy more quantity
                if segment == "champion" and random.random() < 0.3:
                    qty += 1
                
                amount = round(unit_price * qty * (1 - discount/100), 2)
                
                cursor.execute(
                    """INSERT INTO order_items (order_id, product_id, quantity, unit_price, amount) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (order_id, prod_id, qty, unit_price, amount)
                )
                item_count += 1
    
    print(f"Seeded {order_count} orders with {item_count} line items.")
    
    # Print segment distribution
    segment_counts = {}
    for seg in customer_segments.values():
        segment_counts[seg] = segment_counts.get(seg, 0) + 1
    print(f"Customer segments: {segment_counts}")


def main():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Drop tables to reset
    cursor.execute("DROP TABLE IF EXISTS order_items")
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("DROP TABLE IF EXISTS products")
    
    create_tables(cursor)
    product_ids = seed_products(cursor)
    customers = seed_customers(cursor)
    seed_orders(cursor, customers, product_ids)
    
    conn.commit()
    
    # Print summary stats
    cursor.execute("SELECT COUNT(*) FROM customers")
    print(f"\nTotal customers: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM orders")
    print(f"Total orders: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT SUM(amount) FROM order_items")
    print(f"Total revenue: ${cursor.fetchone()[0]:,.2f}")
    
    cursor.execute("SELECT country, COUNT(*) FROM customers GROUP BY country ORDER BY COUNT(*) DESC")
    print(f"\nCustomers by country:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    cursor.execute("""
        SELECT p.category, SUM(oi.amount) as revenue 
        FROM order_items oi 
        JOIN products p ON oi.product_id = p.id 
        GROUP BY p.category 
        ORDER BY revenue DESC
    """)
    print(f"\nRevenue by category:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: ${row[1]:,.2f}")
    
    conn.close()
    print(f"\nDatabase {DB_NAME} created successfully with rich test data!")


if __name__ == "__main__":
    main()
