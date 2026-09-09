-- ============================================================
-- Retail Demand Intelligence
-- Analytical database schema
-- ============================================================
--
-- Grain:
-- One row represents one item at one store on one calendar day.
--
-- This schema is intentionally compact. It captures the fields needed
-- for demand, availability, pricing, calendar, and forecast analysis
-- without reproducing every intermediate ML feature.


CREATE TABLE IF NOT EXISTS daily_demand (
    date TEXT NOT NULL,
    item_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    dept_id TEXT NOT NULL,
    cat_id TEXT NOT NULL,

    -- Observed unit demand for the item-day.
    sales INTEGER NOT NULL CHECK (sales >= 0),

    -- Price may be NULL when the product is not yet available.
    sell_price REAL,

    -- 1 means the item is available for sale; 0 means unavailable.
    is_available INTEGER NOT NULL
        CHECK (is_available IN (0, 1)),

    weekday TEXT,
    month INTEGER NOT NULL
        CHECK (month BETWEEN 1 AND 12),
    year INTEGER NOT NULL,

    -- California SNAP indicator from the M5 calendar.
    snap_CA INTEGER NOT NULL
        CHECK (snap_CA IN (0, 1)),

    event_name_1 TEXT,
    event_type_1 TEXT,

    -- Prevent duplicate observations at the analytical grain.
    PRIMARY KEY (
        date,
        item_id,
        store_id
    )
);


CREATE TABLE IF NOT EXISTS demand_forecasts (
    date TEXT NOT NULL,
    item_id TEXT NOT NULL,
    store_id TEXT NOT NULL,

    -- Actual demand is nullable so forecasts can also be stored before
    -- the target-period sales outcome becomes known.
    actual_sales INTEGER
        CHECK (
            actual_sales IS NULL
            OR actual_sales >= 0
        ),

    poisson_prediction REAL NOT NULL
        CHECK (poisson_prediction >= 0),

    positive_demand_probability REAL NOT NULL
        CHECK (
            positive_demand_probability >= 0
            AND positive_demand_probability <= 1
        ),

    two_stage_prediction REAL NOT NULL
        CHECK (two_stage_prediction >= 0),

    is_available INTEGER NOT NULL
        CHECK (is_available IN (0, 1)),

    PRIMARY KEY (
        date,
        item_id,
        store_id
    )
);


-- This index supports common time-series queries such as daily demand
-- trends and date-range filtering.
CREATE INDEX IF NOT EXISTS idx_daily_demand_date
ON daily_demand (date);


-- This index supports product-level demand and pricing analysis.
CREATE INDEX IF NOT EXISTS idx_daily_demand_item
ON daily_demand (item_id);


-- This index supports forecast evaluation over a date range.
CREATE INDEX IF NOT EXISTS idx_demand_forecasts_date
ON demand_forecasts (date);