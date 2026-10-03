"""
Direct Vercel Serverless Function entrypoint for /api/predict.
Routes directly to the core Flask application in api/index.py.
"""

from api.index import app
