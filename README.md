# Free Lossless: Open AI Frame Generation

Free Lossless is a high-performance, non-intrusive Frame Generation tool for Windows. It enhances game fluidity by creating synthetic frames using AI, similar to technologies like DLSS 3 or LSFG, but without requiring any modification to the game's code.

## How it works (Simple words)

Unlike most frame generation tools that need to be "inside" the game (using its engine data), Free Lossless works from the **outside**:

1.  **High-Speed Capture**: It looks at your screen and "captures" the game window at high speed (using DXCAM or BitBlt).
2.  **AI Interpolation**: It uses the **RIFE AI model** to analyze the movement between two real frames and calculates exactly how the intermediate frame would look.
3.  **Transparent Overlay**: It creates a completely transparent window (an "overlay") that sits on top of your game. It displays the original and the new AI-generated frames in perfect sequence.

**No Injection**: Since it doesn't inject files, hook into the game process, or modify the GPU pipeline, it is extremely safe and compatible with almost any application.

## Why use it?

*   **No Game Access**: Perfect for older games, emulators, or modern titles that don't support DLSS/FSR frame generation natively.
*   **Engine Independent**: It works regardless of the graphics engine (DirectX, OpenGL, Vulkan).
*   **Lossless Approach**: No need to lower your internal game resolution unless you want extra performance.

---

## Setup Guide

### 1. Create a Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate the Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
python main.py
```

## Building the Executable
To create a standalone `.exe`:
```powershell
python build_app.py
```
The executable will be generated in the `dist` folder.

---

**Hardware Support**: For the best experience with the AI engine, the app utilizes **DirectML**. This ensures high-performance acceleration on almost any modern GPU (NVIDIA, AMD, or Intel) under Windows.
