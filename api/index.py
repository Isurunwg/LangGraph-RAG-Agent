import os
import sys

# Ensure root directory is in sys.path for Vercel serverless imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
