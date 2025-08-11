#!/usr/bin/env python3
"""
Streamlit startup script with proper environment handling
"""
import sys
import os
import subprocess

# Add venv to Python path
venv_path = "/Users/hc/Documents/HAM10K/skin-cancer-detection/venv/lib/python3.9/site-packages"
project_path = "/Users/hc/Documents/HAM10K/skin-cancer-detection"

if venv_path not in sys.path:
    sys.path.insert(0, venv_path)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Set environment variables
os.environ['PYTHONPATH'] = f"{venv_path}:{project_path}:{os.environ.get('PYTHONPATH', '')}"

# Import and run Streamlit
try:
    from streamlit.web import cli as stcli
    import streamlit as st
    
    print("✅ Streamlit imported successfully")
    print("🚀 Starting Streamlit frontend with monitoring dashboard...")
    
    # Run Streamlit
    sys.argv = [
        "streamlit", 
        "run", 
        "src/frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    
    stcli.main()
    
except ImportError as e:
    print(f"❌ Error importing Streamlit: {e}")
    print("Available packages:")
    import pkg_resources
    for pkg in pkg_resources.working_set:
        if 'streamlit' in pkg.project_name.lower():
            print(f"  - {pkg.project_name}: {pkg.version}")
