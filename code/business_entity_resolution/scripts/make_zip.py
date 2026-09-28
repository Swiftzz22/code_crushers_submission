"""Build <team>_submission.zip with the exact structure required by the brief."""
import sys, zipfile
from pathlib import Path
repo, outdir, dest = Path(sys.argv[1]), Path(sys.argv[2]).expanduser(), Path(sys.argv[3])
pkg = repo / "code" / "business_entity_resolution"
entries = [(outdir / "matching_results.tsv", "output/matching_results.tsv"),
           (outdir / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
           (repo / "Documentation_template.md", "Documentation_template.md")]
for f in sorted(pkg.rglob("*")):
    if f.is_file() and "__pycache__" not in f.parts and ".pytest_cache" not in f.parts:
        entries.append((f, "code/business_entity_resolution/" + f.relative_to(pkg).as_posix()))
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for src, arc in entries:
        z.write(src, arc)
print(dest, f"{dest.stat().st_size/1e6:.1f} MB,", len(entries), "files")
