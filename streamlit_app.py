"""Streamlit Community Cloud entry point.

Cloud looks for this file at the repository root by default. Everything lives in
``cfb/dashboard/app.py``; this is only the door.

Deploying:
  1. share.streamlit.io -> Create app -> pick this repo and branch
  2. Main file path:  streamlit_app.py
  3. Advanced settings -> Secrets:  CFBD_API_KEY = "..."

Streamlit executes this file as a script with ``__name__ == "__main__"``, so the
guard below runs the app there while keeping a plain ``import streamlit_app``
side-effect free (which is what CI checks).
"""
from cfb.dashboard.app import main

if __name__ == "__main__":
    main()
