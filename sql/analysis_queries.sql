-- ============================================================
-- Retail Demand Intelligence
-- Business and forecast analysis queries
-- ============================================================


-- ------------------------------------------------------------
-- 1. Daily store demand
-- ------------------------------------------------------------
-- Aggregates item-level observations into total daily unit demand.
-- Useful for identifying demand trends, peaks, and unusual dates.

SELECT
    date,
    SUM(sales) AS total_sales
FROM daily_demand
GROUP BY date
ORDER BY date;


-- ------------------------------------------------------------
-- 2. Demand by product category
-- ------------------------------------------------------------
-- Shows how much each category contributes to total observed demand.

SELECT
    cat_id,
    SUM(sales) AS total_sales,
    ROUND(
        100.0 * SUM(sales)
        / SUM(SUM(sales)) OVER (),
        2
    ) AS sales_share_pct
FROM daily_demand
GROUP BY cat_id
ORDER BY total_sales DESC;


-- ------------------------------------------------------------
-- 3. Average demand by weekday
-- ------------------------------------------------------------
-- First aggregates to daily store demand so weekdays with more
-- item records do not receive disproportionate weight.

WITH daily_totals AS (
    SELECT
        date,
        weekday,
        SUM(sales) AS total_sales
    FROM daily_demand
    GROUP BY
        date,
        weekday
)

SELECT
    weekday,
    ROUND(
        AVG(total_sales),
        2
    ) AS average_daily_sales
FROM daily_totals
GROUP BY weekday
ORDER BY average_daily_sales DESC;


-- ------------------------------------------------------------
-- 4. Product availability and zero-demand behavior
-- ------------------------------------------------------------
-- Separates unavailable observations from available observations
-- with zero or positive sales. Availability and zero demand are
-- intentionally treated as different business states.

SELECT
    CASE
        WHEN is_available = 0
            THEN 'unavailable'
        WHEN sales = 0
            THEN 'available_zero_sales'
        ELSE 'available_positive_sales'
    END AS demand_state,

    COUNT(*) AS observations,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*)) OVER (),
        2
    ) AS observation_share_pct

FROM daily_demand

GROUP BY demand_state
ORDER BY observations DESC;


-- ------------------------------------------------------------
-- 5. SNAP versus non-SNAP demand
-- ------------------------------------------------------------
-- This is descriptive association only. It should not be interpreted
-- as a causal estimate of SNAP's effect on demand.

WITH daily_snap_demand AS (
    SELECT
        date,
        snap_CA,
        SUM(sales) AS total_sales
    FROM daily_demand
    GROUP BY
        date,
        snap_CA
)

SELECT
    snap_CA,
    COUNT(*) AS number_of_days,

    ROUND(
        AVG(total_sales),
        2
    ) AS average_daily_sales

FROM daily_snap_demand

GROUP BY snap_CA
ORDER BY snap_CA;


-- ------------------------------------------------------------
-- 6. Average selling price by category
-- ------------------------------------------------------------
-- NULL prices are excluded automatically by AVG(), which is desirable
-- because missing price represents unavailable/pre-release periods
-- rather than a zero-dollar selling price.

SELECT
    cat_id,

    ROUND(
        AVG(sell_price),
        2
    ) AS average_sell_price,

    ROUND(
        MIN(sell_price),
        2
    ) AS minimum_sell_price,

    ROUND(
        MAX(sell_price),
        2
    ) AS maximum_sell_price

FROM daily_demand

WHERE is_available = 1

GROUP BY cat_id
ORDER BY average_sell_price DESC;


-- ------------------------------------------------------------
-- 7. Highest-demand products
-- ------------------------------------------------------------
-- Ranks products by total observed unit sales.

SELECT
    item_id,
    dept_id,
    cat_id,
    SUM(sales) AS total_sales,

    ROUND(
        AVG(sales),
        3
    ) AS average_daily_sales

FROM daily_demand

GROUP BY
    item_id,
    dept_id,
    cat_id

ORDER BY total_sales DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 8. Forecast MAE by model
-- ------------------------------------------------------------
-- Evaluates forecasts only when actual demand has become available.

SELECT
    ROUND(
        AVG(
            ABS(
                actual_sales
                - poisson_prediction
            )
        ),
        4
    ) AS poisson_mae,

    ROUND(
        AVG(
            ABS(
                actual_sales
                - two_stage_prediction
            )
        ),
        4
    ) AS two_stage_mae

FROM demand_forecasts

WHERE actual_sales IS NOT NULL;


-- ------------------------------------------------------------
-- 9. Forecast WMAPE by model
-- ------------------------------------------------------------
-- WMAPE expresses total absolute forecast error relative to total
-- actual demand. NULLIF prevents division by zero.

SELECT
    ROUND(
        SUM(
            ABS(
                actual_sales
                - poisson_prediction
            )
        )
        / NULLIF(
            SUM(actual_sales) * 1.0,
            0
        ),
        4
    ) AS poisson_wmape,

    ROUND(
        SUM(
            ABS(
                actual_sales
                - two_stage_prediction
            )
        )
        / NULLIF(
            SUM(actual_sales) * 1.0,
            0
        ),
        4
    ) AS two_stage_wmape

FROM demand_forecasts

WHERE actual_sales IS NOT NULL;


-- ------------------------------------------------------------
-- 10. Forecast bias by model
-- ------------------------------------------------------------
-- Positive values indicate aggregate overforecasting.
-- Negative values indicate aggregate underforecasting.

SELECT
    ROUND(
        100.0
        * SUM(
            poisson_prediction
            - actual_sales
        )
        / NULLIF(
            SUM(actual_sales),
            0
        ),
        2
    ) AS poisson_bias_pct,

    ROUND(
        100.0
        * SUM(
            two_stage_prediction
            - actual_sales
        )
        / NULLIF(
            SUM(actual_sales),
            0
        ),
        2
    ) AS two_stage_bias_pct

FROM demand_forecasts

WHERE actual_sales IS NOT NULL;


-- ------------------------------------------------------------
-- 11. Forecast performance by product category
-- ------------------------------------------------------------
-- Joins forecasts back to the demand table to identify categories
-- where forecast accuracy is stronger or weaker.

SELECT
    d.cat_id,

    COUNT(*) AS observations,

    ROUND(
        AVG(
            ABS(
                f.actual_sales
                - f.poisson_prediction
            )
        ),
        4
    ) AS poisson_mae,

    ROUND(
        AVG(
            ABS(
                f.actual_sales
                - f.two_stage_prediction
            )
        ),
        4
    ) AS two_stage_mae

FROM demand_forecasts AS f

INNER JOIN daily_demand AS d
    ON f.date = d.date
    AND f.item_id = d.item_id
    AND f.store_id = d.store_id

WHERE f.actual_sales IS NOT NULL

GROUP BY d.cat_id
ORDER BY two_stage_mae;