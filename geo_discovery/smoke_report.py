# smoke_report.py
from report.builders import build_report_model
from report.render_docx import render_docx

rm = build_report_model("danone", "aptamil", "infant formula")
print("stats:", rm.stats())
print("provisional:", rm.provisional, rm.provisional_reasons)
print("warnings:", rm.warnings)
for s in rm.sections:
    print(f"  {s.id:14} {len(s.blocks):3} blocks  {s.title}")
open("/tmp/geo_test.docx", "wb").write(render_docx(rm))
print("\nwritten /tmp/geo_test.docx")