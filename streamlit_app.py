# Streamlit Cloud entry point
# This file exists so Streamlit Cloud can find the app at repo root.
# It simply delegates to dashboard/app.py

import runpy, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
runpy.run_path(os.path.join(os.path.dirname(__file__), "dashboard", "app.py"),
               run_name="__main__")
