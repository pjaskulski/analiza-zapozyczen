import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import numpy as np

# Patch NumPy 2.x compatibility for ChromaDB
if not hasattr(np, 'float_'):
    np.float_ = np.float64 # type: ignore
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'NaN'):
    np.NaN = np.nan # type: ignore
import logging
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
from app import app

if __name__ == '__main__':
    # Run the server in debug mode on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
