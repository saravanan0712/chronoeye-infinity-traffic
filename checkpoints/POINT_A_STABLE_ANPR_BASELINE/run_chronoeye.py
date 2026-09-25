"""
ChronoEye Infinity - One-Command Master Launcher
Starts the FastAPI Backend Server, WebSocket Broadcaster, and React Command Center Dashboard.
"""

import os
import sys
import time
import subprocess
import webbrowser

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Locate Virtual Environment Python
VENV_PYTHON = os.path.join(PROJECT_ROOT, "chronoeye", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

def main():
    print("=" * 70)
    print("        CHRONOEYE INFINITY - MASTER SYSTEM LAUNCHER")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python Venv : {VENV_PYTHON}")
    print("-" * 70)

    # 1. Environment Setup & PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = BACKEND_DIR

    processes = []

    try:
        # 2. Launch FastAPI Backend
        print("[1/2] Launching ChronoEye Backend API & WebSocket Server...")
        backend_cmd = [
            VENV_PYTHON,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
        ]
        p_backend = subprocess.Popen(backend_cmd, cwd=BACKEND_DIR, env=env)
        processes.append(p_backend)
        print("      -> Backend server starting at http://127.0.0.1:8000")
        print("      -> Swagger API Docs available at http://127.0.0.1:8000/docs")

        # Give backend a moment to boot
        time.sleep(2)

        # 3. Launch Frontend Vite Dev Server
        print("[2/2] Launching React Command Center Dashboard...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        frontend_cmd = [npm_cmd, "run", "dev"]
        p_frontend = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR, shell=True)
        processes.append(p_frontend)
        print("      -> Frontend Dashboard starting at http://localhost:3000")

        time.sleep(2)

        # 4. Open Browser
        print("\nOpening ChronoEye Command Center in default browser...")
        try:
            webbrowser.open("http://localhost:3000")
        except Exception:
            pass

        print("\n" + "=" * 70)
        print(" CHRONOEYE INFINITY SYSTEM IS LIVE AND RUNNING!")
        print("=" * 70)
        print(" -> Dashboard URL : http://localhost:3000")
        print(" -> Backend API   : http://127.0.0.1:8000")
        print(" -> Swagger Docs  : http://127.0.0.1:8000/docs")
        print(" -> Health Check  : http://127.0.0.1:8000/api/v1/health")
        print(" -> WebSocket     : ws://127.0.0.1:8000/ws/traffic")
        print("=" * 70)
        print("Press CTRL+C to terminate all services cleanly.\n")

        # Keep parent alive until interrupt
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down ChronoEye Infinity services...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        print("All processes terminated successfully.")

if __name__ == "__main__":
    main()
