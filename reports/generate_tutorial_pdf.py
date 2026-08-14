from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = PROJECT_ROOT / "PROJECT_TUTORIAL_12TH_GUIDE.md"
OUTPUT_PDF = PROJECT_ROOT / "reports" / "PROJECT_TUTORIAL_12TH_GUIDE.pdf"


def markdown_line_to_paragraph(line: str, styles):
    text = line.rstrip("\n")

    if not text.strip():
        return Spacer(1, 0.25 * cm)

    if text.startswith("# "):
        return Paragraph(text[2:].strip(), styles["h1"])

    if text.startswith("## "):
        return Paragraph(text[3:].strip(), styles["h2"])

    if text.startswith("- "):
        body = text[2:].strip()
        return Paragraph(f"&#8226; {body}", styles["bullet"])

    if text.startswith("1.") or text.startswith("2.") or text.startswith("3.") or text.startswith("4.") or text.startswith("5.") or text.startswith("6.") or text.startswith("7.") or text.startswith("8.") or text.startswith("9."):
        return Paragraph(text.strip(), styles["body"])

    # Escape angle brackets to avoid XML parser issues in reportlab Paragraph.
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, styles["body"])


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Image Classifier Project Tutorial",
        author="GitHub Copilot",
    )

    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=19,
            leading=23,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#243B53"),
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=14,
            spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=14,
            leftIndent=12,
            spaceAfter=2,
        ),
    }

    lines = INPUT_MD.read_text(encoding="utf-8").splitlines()
    story = []
    for line in lines:
        story.append(markdown_line_to_paragraph(line, styles))

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(f"PDF created: {OUTPUT_PDF}")
