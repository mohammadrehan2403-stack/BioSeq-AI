# Deploy BioSeq AI Advanced on Render

This package contains a standalone Gradio application (`bioseq_ai_advanced.py`) and its dependencies (`requirements_advanced.txt`). It does **not** overwrite the existing `bioseq_ai.py` app automatically.

## GitHub files to add
Add these files to the repository root (leave existing app files unchanged):
- `bioseq_ai_advanced.py`
- `requirements_advanced.txt`

Commit the changes to the branch connected to the NEW Render service.

## Render settings
In the **new** Render web service:
- Build Command: `pip install -r requirements_advanced.txt`
- Start Command: `python bioseq_ai_advanced.py`

Keep the existing working service unchanged until the new service reports **Live** and the advanced tabs have been tested.

## Notes
- Startup tries to download the UCI promoter dataset. Other DNA tools should still start if that download fails; the promoter classifier will show as unavailable.
- NCBI BLAST requires internet access and may take several minutes. It can fail when NCBI is unavailable or rate-limiting.
- The app is educational and not for clinical decisions.
