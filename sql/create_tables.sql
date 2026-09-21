DROP TABLE IF EXISTS order_item_refunds;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS website_pageviews;
DROP TABLE IF EXISTS website_sessions;
DROP TABLE IF EXISTS products;

CREATE TABLE website_sessions (
    website_session_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    user_id INTEGER,
    is_repeat_session INTEGER,
    utm_source TEXT,
    utm_campaign TEXT,
    utm_content TEXT,
    device_type TEXT,
    http_referer TEXT
);

CREATE TABLE website_pageviews (
    website_pageview_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    website_session_id INTEGER NOT NULL,
    pageview_url TEXT
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    product_name TEXT
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    website_session_id INTEGER NOT NULL,
    user_id INTEGER,
    primary_product_id INTEGER,
    items_purchased INTEGER,
    price_usd REAL,
    cogs_usd REAL
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    is_primary_item INTEGER,
    price_usd REAL,
    cogs_usd REAL
);

CREATE TABLE order_item_refunds (
    order_item_refund_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    order_item_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    refund_amount_usd REAL
);
