{
  "version": 2,
  "builds": [{
    "src": "api/index.py",
    "use": "@vercel/python",
    "config": {
      "maxLambdaSize": "50mb",
      "suppressLoaders": true,
      "externalNodeModules": ["numpy", "scikit-learn", "joblib"],
      "includeFiles": ["model.joblib", "le_*.npy", "meal_ideas.json"],
      "runtime": "python3.9"
    }
  }],
  "functions": {
    "api/**": {
      "excludeFiles": "{__pycache__,tests,*.log,*.tmp}/**"
    }
  },
  "env": {
    "PYTHON_ENABLE_WASM": "1",
    "PYTHON_USE_PYPI": "false"
  }
}
