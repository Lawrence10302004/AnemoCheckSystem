# Railway configuration for persistent database
# This file tells Railway to mount a persistent volume

# Add this to your Railway project:
# 1. Go to Railway dashboard
# 2. Click "New" → "Volume"
# 3. Name: anemo-db
# 4. Mount path: /app/data
# 5. Add environment variable: DATABASE_PATH=/app/data/anemia_classification.db

# The database will now persist across deployments
