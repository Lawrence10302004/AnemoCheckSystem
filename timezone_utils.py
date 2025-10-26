"""
Timezone utilities for Philippines time handling
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Philippines timezone
PH_TZ = ZoneInfo('Asia/Manila')

def get_philippines_time():
    """Get current time in Philippines timezone."""
    return datetime.now(PH_TZ)

def format_philippines_time(dt=None):
    """Format datetime in Philippines timezone."""
    if dt is None:
        dt = get_philippines_time()
    elif dt.tzinfo is None:
        # Assume UTC if no timezone info
        dt = dt.replace(tzinfo=ZoneInfo('UTC')).astimezone(PH_TZ)
    else:
        dt = dt.astimezone(PH_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def get_philippines_time_for_db():
    """Get Philippines time formatted for database storage."""
    return get_philippines_time().strftime('%Y-%m-%d %H:%M:%S')

def parse_philippines_time(timestamp_str):
    """Parse timestamp string and convert to Philippines time."""
    try:
        if isinstance(timestamp_str, str):
            if 'T' in timestamp_str:
                # ISO format with T
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # SQLite format - check if it has microseconds (like '2025-10-23 22:01:40.381275')
                # This indicates it came from Python datetime.now() which is UTC
                if '.' in timestamp_str:
                    # Has microseconds - this is UTC from the database
                    dt = datetime.fromisoformat(timestamp_str)
                    dt = dt.replace(tzinfo=ZoneInfo('UTC'))
                    dt = dt.astimezone(PH_TZ)
                else:
                    # No microseconds - assume it's already in Philippines time
                    dt = datetime.fromisoformat(timestamp_str)
                    dt = dt.replace(tzinfo=PH_TZ)
        else:
            # Already a datetime object
            dt = timestamp_str
        
        # If no timezone info (shouldn't happen now, but just in case)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=PH_TZ)
        
        return dt
    except Exception as e:
        print(f"Error parsing timestamp: {e}")
        return None

def format_philippines_time_display(timestamp_str):
    """Format timestamp for display in Philippines time."""
    ph_time = parse_philippines_time(timestamp_str)
    if ph_time:
        return ph_time.strftime('%Y-%m-%d %H:%M:%S %Z')
    return timestamp_str

def format_philippines_time_ampm(timestamp_str):
    """Format timestamp for display in Philippines time with AM/PM."""
    ph_time = parse_philippines_time(timestamp_str)
    if ph_time:
        return ph_time.strftime('%Y-%m-%d %I:%M:%S %p')
    return timestamp_str

def get_philippines_time_plus_minutes(minutes):
    """Get Philippines time plus specified minutes."""
    return get_philippines_time() + timedelta(minutes=minutes)
