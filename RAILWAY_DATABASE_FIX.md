# Railway Database Persistence Setup

## Problem
Railway uses ephemeral storage - the SQLite database gets reset on every deployment, losing all data.

## Solution: Add Persistent Volume

### 1. Create Railway Volume
1. Go to your Railway project dashboard
2. Click **"New"** → **"Volume"**
3. Name it: `anemo-db`
4. Mount path: `/app/data`

### 2. Update Database Path
The database should be stored in the persistent volume instead of the app directory.

### 3. Environment Variables
Add these environment variables in Railway:
- `DATABASE_PATH=/app/data/anemia_classification.db`

### 4. Update Code
Modify `database.py` to use the persistent path:

```python
import os

# Use persistent volume if available, fallback to local
DB_PATH = os.environ.get('DATABASE_PATH', 'anemia_classification.db')
```

## Alternative: PostgreSQL Database

### 1. Add PostgreSQL Service
1. In Railway dashboard, click **"New"** → **"Database"** → **"PostgreSQL"
2. Railway will provide connection details

### 2. Update Database Code
Replace SQLite with PostgreSQL using `psycopg2` or `SQLAlchemy`.

## Quick Fix for Now

If you want to keep using SQLite temporarily:

1. **Don't redeploy** unless necessary
2. **Export data** before deployments
3. **Use Railway volumes** for persistence

## Recommended Action

1. Add a Railway volume for `/app/data`
2. Update database path to use the volume
3. Redeploy once with the volume mounted

This will make your database persistent across deployments.
