-- Monthly traffic and conversion
SELECT
    substr(s.created_at, 1, 7) AS month,
    COUNT(DISTINCT s.website_session_id) AS sessions,
    COUNT(DISTINCT o.order_id) AS orders,
    ROUND(100.0 * COUNT(DISTINCT o.order_id) / COUNT(DISTINCT s.website_session_id), 2) AS conversion_rate_pct
FROM website_sessions s
LEFT JOIN orders o
    ON s.website_session_id = o.website_session_id
GROUP BY substr(s.created_at, 1, 7)
ORDER BY month;

-- Monthly revenue and gross profit
SELECT
    substr(created_at, 1, 7) AS month,
    COUNT(*) AS orders,
    ROUND(SUM(price_usd), 2) AS gross_revenue,
    ROUND(SUM(price_usd - cogs_usd), 2) AS gross_profit,
    ROUND(AVG(price_usd), 2) AS average_order_value
FROM orders
GROUP BY substr(created_at, 1, 7)
ORDER BY month;

-- Marketing source performance
SELECT
    COALESCE(s.utm_source, 'direct/other') AS source,
    COUNT(DISTINCT s.website_session_id) AS sessions,
    COUNT(DISTINCT o.order_id) AS orders,
    ROUND(SUM(COALESCE(o.price_usd, 0)), 2) AS revenue,
    ROUND(100.0 * COUNT(DISTINCT o.order_id) / COUNT(DISTINCT s.website_session_id), 2) AS conversion_rate_pct
FROM website_sessions s
LEFT JOIN orders o
    ON s.website_session_id = o.website_session_id
GROUP BY COALESCE(s.utm_source, 'direct/other')
ORDER BY revenue DESC;
