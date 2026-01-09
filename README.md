# Major Project - 2

Small web app (Flask) for water/air image pollution classification.

Quick start (Windows, `cmd.exe`)

1. Create and activate a virtual environment (recommended):
```
cd /d "C:\Users\LENOVO\OneDrive\Desktop\rosh\Major Project - 2"
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the app:
```
python app.py
```

Notes
- The model files (`air_best_model.h5`, `water_best_model.h5`) are large and are `.gitignore`d by default. Use Git LFS if you want to track them in the repo.
- If you hit TensorFlow / NumPy DLL errors on Windows, ensure you have the Microsoft Visual C++ Redistributable x64 installed: https://aka.ms/vs/17/release/vc_redist.x64.exe

Repo & Push
- I can initialize a Git repo here and help push to GitHub. Tell me whether you want me to create the remote repository for you (I will need a GitHub personal access token or the `gh` CLI configured), or whether you'd prefer to run the final `git push` yourself.
