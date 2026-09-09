from pathlib import Path
import sqlite3

import pytest


# Resolve SQL files relative to the repository root so the tests work
# regardless of the directory from which pytest is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
QUERIES_PATH = PROJECT_ROOT / "sql" / "analysis_queries.sql"


@pytest.fixture
def database():
    """
    Create a fresh in-memory SQLite database for every SQL test.

    Using an in-memory database keeps the test suite fast and prevents
    test data from creating persistent database files in the project.
    """

    connection = sqlite3.connect(
        ":memory:"
    )

    schema_sql = SCHEMA_PATH.read_text(
        encoding="utf-8"
    )

    connection.executescript(
        schema_sql
    )

    yield connection

    connection.close()


def test_schema_creates_expected_tables(database):
    """
    The SQL schema should create both analytical tables successfully.
    """

    rows = database.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
        """
    ).fetchall()

    table_names = {
        row[0]
        for row in rows
    }

    assert "daily_demand" in table_names
    assert "demand_forecasts" in table_names


def test_daily_demand_rejects_duplicate_grain(database):
    """
    The primary key should prevent duplicate item-store-date records.
    """

    row = (
        "2026-01-01",
        "ITEM_1",
        "CA_1",
        "FOODS_1",
        "FOODS",
        5,
        2.50,
        1,
        "Thursday",
        1,
        2026,
        0,
        None,
        None,
    )

    database.execute(
        """
        INSERT INTO daily_demand (
            date,
            item_id,
            store_id,
            dept_id,
            cat_id,
            sales,
            sell_price,
            is_available,
            weekday,
            month,
            year,
            snap_CA,
            event_name_1,
            event_type_1
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        row,
    )

    with pytest.raises(
        sqlite3.IntegrityError
    ):
        database.execute(
            """
            INSERT INTO daily_demand (
                date,
                item_id,
                store_id,
                dept_id,
                cat_id,
                sales,
                sell_price,
                is_available,
                weekday,
                month,
                year,
                snap_CA,
                event_name_1,
                event_type_1
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            row,
        )


def test_daily_demand_enforces_availability_constraint(
    database,
):
    """
    Availability must be encoded as either 0 or 1.
    """

    with pytest.raises(
        sqlite3.IntegrityError
    ):
        database.execute(
            """
            INSERT INTO daily_demand (
                date,
                item_id,
                store_id,
                dept_id,
                cat_id,
                sales,
                sell_price,
                is_available,
                weekday,
                month,
                year,
                snap_CA
            )
            VALUES (
                '2026-01-01',
                'ITEM_1',
                'CA_1',
                'FOODS_1',
                'FOODS',
                5,
                2.50,
                2,
                'Thursday',
                1,
                2026,
                0
            );
            """
        )


def test_forecast_table_allows_unknown_actual_sales(
    database,
):
    """
    Future forecasts should be storable before actual demand becomes
    available, so actual_sales must permit NULL.
    """

    database.execute(
        """
        INSERT INTO demand_forecasts (
            date,
            item_id,
            store_id,
            actual_sales,
            poisson_prediction,
            positive_demand_probability,
            two_stage_prediction,
            is_available
        )
        VALUES (
            '2026-01-01',
            'ITEM_1',
            'CA_1',
            NULL,
            4.5,
            0.80,
            4.5,
            1
        );
        """
    )

    actual_sales = database.execute(
        """
        SELECT actual_sales
        FROM demand_forecasts
        WHERE item_id = 'ITEM_1';
        """
    ).fetchone()[0]

    assert actual_sales is None


def test_forecast_probability_constraint(database):
    """
    Positive-demand probabilities outside [0, 1] should be rejected.
    """

    with pytest.raises(
        sqlite3.IntegrityError
    ):
        database.execute(
            """
            INSERT INTO demand_forecasts (
                date,
                item_id,
                store_id,
                actual_sales,
                poisson_prediction,
                positive_demand_probability,
                two_stage_prediction,
                is_available
            )
            VALUES (
                '2026-01-01',
                'ITEM_1',
                'CA_1',
                5,
                4.5,
                1.20,
                4.5,
                1
            );
            """
        )


def test_analysis_queries_execute_successfully(database):
    """
    Every statement in the analytical SQL file should execute against
    the declared SQLite schema without syntax or schema errors.

    Small representative datasets are inserted because several queries
    contain aggregations, joins, and forecast evaluation logic.
    """

    database.executemany(
        """
        INSERT INTO daily_demand (
            date,
            item_id,
            store_id,
            dept_id,
            cat_id,
            sales,
            sell_price,
            is_available,
            weekday,
            month,
            year,
            snap_CA,
            event_name_1,
            event_type_1
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                "2026-01-01",
                "ITEM_1",
                "CA_1",
                "FOODS_1",
                "FOODS",
                5,
                2.50,
                1,
                "Thursday",
                1,
                2026,
                0,
                None,
                None,
            ),
            (
                "2026-01-01",
                "ITEM_2",
                "CA_1",
                "HOUSEHOLD_1",
                "HOUSEHOLD",
                0,
                5.00,
                1,
                "Thursday",
                1,
                2026,
                0,
                None,
                None,
            ),
            (
                "2026-01-02",
                "ITEM_1",
                "CA_1",
                "FOODS_1",
                "FOODS",
                7,
                2.50,
                1,
                "Friday",
                1,
                2026,
                1,
                None,
                None,
            ),
        ],
    )

    database.executemany(
        """
        INSERT INTO demand_forecasts (
            date,
            item_id,
            store_id,
            actual_sales,
            poisson_prediction,
            positive_demand_probability,
            two_stage_prediction,
            is_available
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                "2026-01-01",
                "ITEM_1",
                "CA_1",
                5,
                4.0,
                0.80,
                4.0,
                1,
            ),
            (
                "2026-01-01",
                "ITEM_2",
                "CA_1",
                0,
                1.0,
                0.20,
                0.0,
                1,
            ),
            (
                "2026-01-02",
                "ITEM_1",
                "CA_1",
                7,
                6.0,
                0.90,
                6.0,
                1,
            ),
        ],
    )

    query_text = QUERIES_PATH.read_text(
        encoding="utf-8"
    )

    # The analysis file contains independent SELECT statements.
    # Splitting on semicolons is sufficient here because these queries
    # contain no semicolons inside string literals or SQL procedures.
    statements = [
        statement.strip()
        for statement in query_text.split(";")
        if statement.strip()
    ]

    assert len(statements) == 11

    for statement in statements:
        cursor = database.execute(
            statement
        )

        # Force SQLite to materialize each SELECT result so execution
        # errors cannot remain hidden behind an unconsumed cursor.
        cursor.fetchall()