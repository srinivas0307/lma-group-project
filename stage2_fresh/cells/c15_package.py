# ===== Package the run (download this ZIP from the Output tab) =====
zip_path = WORK_DIR / f"{RUN_NAME}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to(OUT_DIR.parent))
print("wrote", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)")
results.groupby(["split", "outcome"]).size().unstack(fill_value=0)
