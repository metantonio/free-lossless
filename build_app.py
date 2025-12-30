import PyInstaller.__main__
import os
import shutil

# --- Configuration ---
script_name = "main.py"
exe_name = "FreeLossless"
icon_path = None # Add icon path here if available

# Data folders to include
# Format: (Source, Destination)
datas = [
    ("models", "models"),
]

# Hidden imports that might be missed
hidden_imports = [
    "onnxruntime",
    "cv2",
    "pygame",
    "multiprocessing",
    "win32gui",
    "win32con",
    "win32api",
    "psutil",
    "requests",
]

def build():
    print(f"Building {exe_name}...")
    
    # Ensure build directories are clean
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")

    params = [
        script_name,
        "--name", exe_name,
        "--onefile",
        "--noconsole", # GUI mode
        "--clean",
    ]

    # Add datas
    for src, dst in datas:
        if os.path.exists(src):
            params.extend(["--add-data", f"{src}{os.pathsep}{dst}"])

    # Add hidden imports
    for imp in hidden_imports:
        params.extend(["--hidden-import", imp])

    # Run PyInstaller
    PyInstaller.__main__.run(params)
    
    print("\nBuild Complete! Executable is in the 'dist' folder.")

if __name__ == "__main__":
    build()
