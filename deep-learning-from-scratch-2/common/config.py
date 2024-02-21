import os

# Activate CuPy with environment variable "USE_GPU"
if os.environ.get("USE_GPU") == "1":
    GPU = True
else:
    GPU = False
