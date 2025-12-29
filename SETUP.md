# Guía de Configuración del Entorno

Sigue estos pasos para configurar un entorno virtual y ejecutar la aplicación de Frame Generation:

### 1. Crear el Entorno Virtual
Abre una terminal en la carpeta del proyecto y ejecuta:
```powershell
python -m venv venv
```

### 2. Activar el Entorno Virtual
En Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Instalar Dependencias
Una vez activado el entorno, instala las librerías necesarias:
```powershell
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación
```powershell
python main.py
```

---
**Nota sobre RIFE:** El proyecto usa por defecto un motor de Optical Flow rápido. Para usar RIFE de alta calidad, se recomienda tener instalado y configurado el soporte para Vulkan en tu sistema.
