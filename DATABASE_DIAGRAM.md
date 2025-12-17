# Database Structure Diagram

## Overview
The database contains two main parts:
1. **Chat Application Tables** (Django ORM managed)
2. **Sales Data Tables** (Created by seed.py)

---

## Chat Application Tables

```
┌─────────────────────────┐
│   chat_conversation     │
├─────────────────────────┤
│ PK  id (UUID)           │
│     created_at          │
│     title (nullable)    │
└─────────────────────────┘
           │
           │ 1:N
           │
           ▼
┌─────────────────────────┐
│     chat_message        │
├─────────────────────────┤
│ PK  id                  │
│ FK  conversation_id     │──┐
│     role (user/ai)      │  │
│     content             │  │
│     sql_executed        │  │
│     data_headers        │  │
│     viz_code            │  │
│     created_at          │  │
└─────────────────────────┘  │
                             │
                             │ references
                             │
                             └──┘
```

---

## Sales Data Tables

```
┌─────────────────────────┐
│      customers          │
├─────────────────────────┤
│ PK  id                  │
│     name                │
│     email               │
│     country             │
│     city                │
│     age_group           │
│     acquisition_channel │
│     signup_date         │
└─────────────────────────┘
           │
           │ 1:N
           │
           ▼
┌─────────────────────────┐
│       orders            │
├─────────────────────────┤
│ PK  id                  │
│ FK  customer_id         │──┐
│     date                │  │
│     status              │  │
│     discount_pct        │  │
│     shipping_cost       │  │
└─────────────────────────┘  │
           │                 │
           │ 1:N             │
           │                 │
           ▼                 │
┌─────────────────────────┐  │
│     order_items         │  │
├─────────────────────────┤  │
│ PK  id                  │  │
│ FK  order_id            │──┘
│ FK  product_id          │──┐
│     quantity            │  │
│     unit_price          │  │
│     amount              │  │
└─────────────────────────┘  │
                             │
                             │ references
                             │
                             ▼
┌─────────────────────────┐
│      products           │
├─────────────────────────┤
│ PK  id                  │
│     name                │
│     category            │
│     price               │
│     cost                │
│     launch_date         │
└─────────────────────────┘
```

---

## Complete ER Diagram

```
┌──────────────────────┐         ┌──────────────────────┐
│ chat_conversation    │         │      customers       │
├──────────────────────┤         ├──────────────────────┤
│ PK id (UUID)         │         │ PK id                │
│    created_at        │         │    name              │
│    title             │         │    email             │
└──────────────────────┘         │    country           │
         │                        │    city              │
         │ 1:N                    │    age_group         │
         │                        │    acquisition_channel│
         ▼                        │    signup_date       │
┌──────────────────────┐         └──────────────────────┘
│   chat_message       │                    │
├──────────────────────┤                    │ 1:N
│ PK id                │                    │
│ FK conversation_id    │                    ▼
│    role              │         ┌──────────────────────┐
│    content           │         │       orders         │
│    sql_executed      │         ├──────────────────────┤
│    data_headers      │         │ PK id                │
│    viz_code          │         │ FK customer_id       │
│    created_at        │         │    date              │
└──────────────────────┘         │    status            │
                                 │    discount_pct      │
                                 │    shipping_cost      │
                                 └──────────────────────┘
                                            │
                                            │ 1:N
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │    order_items       │
                                 ├──────────────────────┤
                                 │ PK id                │
                                 │ FK order_id          │
                                 │ FK product_id        │──┐
                                 │    quantity          │  │
                                 │    unit_price        │  │
                                 │    amount            │  │
                                 └──────────────────────┘  │
                                                          │ N:1
                                                          │
                                                          ▼
                                 ┌──────────────────────┐
                                 │      products        │
                                 ├──────────────────────┤
                                 │ PK id                │
                                 │    name              │
                                 │    category          │
                                 │    price             │
                                 │    cost              │
                                 │    launch_date       │
                                 └──────────────────────┘
```

---

## Table Details

### Chat Tables

#### `chat_conversation`
- **Purpose**: Stores chat conversation sessions
- **Primary Key**: `id` (UUID)
- **Fields**:
  - `id`: UUID (auto-generated)
  - `created_at`: DateTime (auto-set on creation)
  - `title`: CharField (max 200, nullable)

#### `chat_message`
- **Purpose**: Stores individual messages in conversations
- **Primary Key**: `id` (auto-increment)
- **Foreign Keys**:
  - `conversation_id` → `chat_conversation.id` (CASCADE delete)
- **Fields**:
  - `id`: Auto-increment integer
  - `conversation`: ForeignKey to Conversation (nullable)
  - `role`: CharField ('user' or 'ai')
  - `content`: TextField (message content)
  - `sql_executed`: TextField (SQL queries executed, nullable)
  - `data_headers`: TextField (column headers from queries, nullable)
  - `viz_code`: TextField (visualization code, nullable)
  - `created_at`: DateTime (auto-set on creation)

---

### Sales Data Tables

#### `products`
- **Purpose**: Product catalog
- **Primary Key**: `id` (auto-increment)
- **Fields**:
  - `id`: Auto-increment integer
  - `name`: Text (product name)
  - `category`: Text (e.g., "Protein", "Vitamins", "Pre-Workout")
  - `price`: Real (selling price)
  - `cost`: Real (cost price)
  - `launch_date`: Date

#### `customers`
- **Purpose**: Customer information
- **Primary Key**: `id` (auto-increment)
- **Fields**:
  - `id`: Auto-increment integer
  - `name`: Text
  - `email`: Text
  - `country`: Text
  - `city`: Text (nullable)
  - `age_group`: Text (e.g., "18-24", "25-34")
  - `acquisition_channel`: Text (e.g., "Organic Search", "Paid Ads")
  - `signup_date`: Date

#### `orders`
- **Purpose**: Order headers
- **Primary Key**: `id` (auto-increment)
- **Foreign Keys**:
  - `customer_id` → `customers.id`
- **Fields**:
  - `id`: Auto-increment integer
  - `customer_id`: Integer (FK to customers)
  - `date`: Date
  - `status`: Text (e.g., "Completed", "Shipped", "Cancelled")
  - `discount_pct`: Real (default 0)
  - `shipping_cost`: Real (default 0)

#### `order_items`
- **Purpose**: Order line items
- **Primary Key**: `id` (auto-increment)
- **Foreign Keys**:
  - `order_id` → `orders.id`
  - `product_id` → `products.id`
- **Fields**:
  - `id`: Auto-increment integer
  - `order_id`: Integer (FK to orders)
  - `product_id`: Integer (FK to products)
  - `quantity`: Integer
  - `unit_price`: Real
  - `amount`: Real (calculated: quantity × unit_price)

---

## Relationships Summary

1. **Conversation → Message**: One-to-Many (one conversation has many messages)
2. **Customer → Order**: One-to-Many (one customer has many orders)
3. **Order → OrderItem**: One-to-Many (one order has many line items)
4. **Product → OrderItem**: One-to-Many (one product appears in many order items)

---

## Notes

- The **chat tables** are managed by Django ORM and migrations
- The **sales tables** are created directly via SQL in `seed.py` (not Django ORM)
- Both sets of tables coexist in the same SQLite database (`sales.db`)
- The sales data is designed for analytics with realistic patterns (RFM analysis, seasonal trends, Pareto distribution)
