# iDRAC Power and Temperature Queries for H100 and ZEN4 Databases

# Unified API for getting power metrics with joins
def get_compute_metrics_with_joins(metric_id: str, hostname: str = None, start_time: str = None, end_time: str = None, limit: int = 100):
    """
    Unified API to get power metrics from any iDRAC table with cross joins to get hostname, source name, and fqdd name.
    
    Args:
        metric_id: Metric ID from public.metrics_definition (will be converted to lowercase for idrac table name)
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
    # Convert metric_id to lowercase for idrac table name
    table_name = metric_id.lower()
    
    # Build WHERE clause based on provided parameters
    where_conditions = []
    
    if hostname is not None:
        # Filter by hostname via join to public.nodes (n)
        where_conditions.append(f"n.hostname = '{hostname}'")
    
    # Handle time filtering with proper corner cases
    if start_time is not None and end_time is not None:
        # Both start_time and end_time provided - use BETWEEN
        where_conditions.append(f"p.timestamp BETWEEN '{start_time}' AND '{end_time}'")
    elif start_time is not None and end_time is None:
        # Only start_time provided - from start_time to now (most recent)
        where_conditions.append(f"p.timestamp >= '{start_time}'")
    elif start_time is None and end_time is not None:
        # Only end_time provided - this is an error case
        raise ValueError("end_time cannot be specified without start_time. Please provide both start_time and end_time, or only start_time.")
    
    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)
    
    if start_time is None and end_time is None:
        return f"""
        WITH limited AS (
            SELECT
                p.timestamp,
                n.hostname,
                s.source,
                f.fqdd,
                p.value,
                m.units
            FROM idrac.{table_name} p
            LEFT JOIN public.nodes n ON p.nodeid = n.nodeid
            LEFT JOIN public.source s ON p.source = s.id
            LEFT JOIN public.fqdd f ON p.fqdd = f.id
            LEFT JOIN public.metrics_definition m ON LOWER(m.metric_id) = LOWER('{metric_id}')
            {where_clause}
            ORDER BY p.timestamp DESC
            LIMIT {limit}
        )
        SELECT *
        FROM limited
        ORDER BY timestamp ASC;
        """

    return f"""
    SELECT 
        p.timestamp,
        n.hostname,
        s.source,
        f.fqdd,
        p.value,
        m.units
    FROM idrac.{table_name} p
    LEFT JOIN public.nodes n ON p.nodeid = n.nodeid
    LEFT JOIN public.source s ON p.source = s.id
    LEFT JOIN public.fqdd f ON p.fqdd = f.id
    LEFT JOIN public.metrics_definition m ON LOWER(m.metric_id) = LOWER('{metric_id}')
    {where_clause}
    ORDER BY p.timestamp ASC;
    """
