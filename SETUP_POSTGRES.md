# PostgreSQL Setup for Railway

Your application now supports PostgreSQL for persistent database storage on Railway!

## What Changed

✅ **Added PostgreSQL support** - Your app automatically detects and uses PostgreSQL when available
✅ **SQLite fallback** - Still works locally with SQLite
✅ **Auto-detection** - No configuration needed, works automatically

## How to Set Up PostgreSQL on Railway

### Step 1: Add PostgreSQL Service

1. Go to your Railway project dashboard
2. Click **"New"** or **"Add Service"**
3. Select **"Database"** → **"PostgreSQL"**
4. Railway will automatically:
   - Create a PostgreSQL database
   - Set environment variables (`PGHOST`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`)
   - Your app will automatically connect to it!

### Step 2: Redeploy

Railway will automatically redeploy your app. The database will be initialized automatically on first run.

### Step 3: Verify

1. Check your app logs - you should see:
   ```
   Using PostgreSQL database (Railway)
   Database initialized successfully.
   ```

2. Create an account and test - data will now persist across deployments!

## Benefits

✅ **Persistent storage** - Data survives deployments
✅ **Production-ready** - PostgreSQL is industry standard
✅ **Automatic** - No manual configuration needed
✅ **Backwards compatible** - Still works locally with SQLite

## Environment Variables (Auto-set by Railway)

Railway automatically provides these when you add PostgreSQL:
- `PGHOST` - Database host
- `PGPORT` - Database port (usually 5432)
- `PGDATABASE` - Database name
- `PGUSER` - Database user
- `PGPASSWORD` - Database password

## Troubleshooting

**Problem:** Database connection errors
**Solution:** Make sure PostgreSQL service is running in Railway dashboard

**Problem:** Tables not created
**Solution:** Check logs for initialization errors, may need to restart deployment

**Problem:** Local development not working
**Solution:** The app automatically uses SQLite locally, no PostgreSQL needed

## Next Steps

1. Add PostgreSQL service in Railway
2. Wait for automatic redeploy
3. Test by creating an account
4. Data will now persist! 🎉

