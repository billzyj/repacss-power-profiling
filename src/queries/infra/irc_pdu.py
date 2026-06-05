# IRC and PDU Data Queries for INFRA Database

def get_irc_metrics_with_joins(metric_id: str, hostname: str = None, start_time: str = None, end_time: str = None, limit: int = 100):
    """
    Unified API to get IRC metrics from any IRC table with joins to get hostname and units.

    Args:
        metric_id: Metric ID from public.metrics_definition (also the IRC table name)
        hostname: Optional hostname to filter by (resolved via public.nodes). If None, returns data for all nodes
        start_time: Optional start timestamp for time range filter
        end_time: Optional end timestamp for time range filter
        limit: Number of records to return (default: 100, only applied when no time range is specified)

    Time Filtering Logic:
        - Both start_time and end_time: Use BETWEEN clause
        - Only start_time: From start_time to now (most recent data)
        - Only end_time: Raises ValueError (invalid case)
        - Neither: Returns most recent data with limit

    Returns:
        SQL query string that joins with public tables and applies filters
    """
    table_name = metric_id.lower()

    where_conditions = []

    if hostname is not None:
        where_conditions.append(f"n.hostname = '{hostname}'")

    if start_time is not None and end_time is not None:
        where_conditions.append(f"m.timestamp BETWEEN '{start_time}' AND '{end_time}'")
    elif start_time is not None and end_time is None:
        where_conditions.append(f"m.timestamp >= '{start_time}'")
    elif start_time is None and end_time is not None:
        raise ValueError("end_time cannot be specified without start_time. Please provide both start_time and end_time, or only start_time.")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    if start_time is None and end_time is None:
        return f"""
        WITH limited AS (
            SELECT
                m.timestamp,
                n.hostname,
                m.value,
                md.units
            FROM irc.{table_name} m
            JOIN public.nodes n ON m.nodeid = n.nodeid
            LEFT JOIN public.metrics_definition md ON LOWER(md.metric_id) = LOWER('{metric_id}')
            {where_clause}
            ORDER BY m.timestamp DESC
            LIMIT {limit}
        )
        SELECT *
        FROM limited
        ORDER BY timestamp ASC;
        """

    return f"""
    SELECT 
        m.timestamp,
        n.hostname,
        m.value,
        md.units
    FROM irc.{table_name} m
    JOIN public.nodes n ON m.nodeid = n.nodeid
    LEFT JOIN public.metrics_definition md ON LOWER(md.metric_id) = LOWER('{metric_id}')
    {where_clause}
    ORDER BY m.timestamp ASC;
    """


def get_pdu_metrics_with_joins(hostname: str = None, start_time: str = None, end_time: str = None, limit: int = 100):
    """
    Unified API to get PDU metrics from the single PDU table with joins to get hostname and units.

    Args:
        hostname: Optional hostname to filter by (resolved via public.nodes). If None, returns data for all nodes
        start_time: Optional start timestamp for time range filter
        end_time: Optional end timestamp for time range filter
        limit: Number of records to return (default: 100, only applied when no time range is specified)

    Time Filtering Logic:
        - Both start_time and end_time: Use BETWEEN clause
        - Only start_time: From start_time to now (most recent data)
        - Only end_time: Raises ValueError (invalid case)
        - Neither: Returns most recent data with limit

    Returns:
        SQL query string that joins with public tables and applies filters
    """
    where_conditions = []

    if hostname is not None:
        where_conditions.append(f"n.hostname = '{hostname}'")

    if start_time is not None and end_time is not None:
        where_conditions.append(f"m.timestamp BETWEEN '{start_time}' AND '{end_time}'")
    elif start_time is not None and end_time is None:
        where_conditions.append(f"m.timestamp >= '{start_time}'")
    elif start_time is None and end_time is not None:
        raise ValueError("end_time cannot be specified without start_time. Please provide both start_time and end_time, or only start_time.")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    if start_time is None and end_time is None:
        return f"""
        WITH limited AS (
            SELECT
                m.timestamp,
                n.hostname,
                m.value,
                'W' as units
            FROM pdu.pdu m
            JOIN public.nodes n ON m.nodeid = n.nodeid
            {where_clause}
            ORDER BY m.timestamp DESC
            LIMIT {limit}
        )
        SELECT *
        FROM limited
        ORDER BY timestamp ASC;
        """

    return f"""
    SELECT 
        m.timestamp,
        n.hostname,
        m.value,
        'W' as units
    FROM pdu.pdu m
    JOIN public.nodes n ON m.nodeid = n.nodeid
    {where_clause}
    ORDER BY m.timestamp ASC;
    """
