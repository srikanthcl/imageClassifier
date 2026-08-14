"""Generate a PDF tutorial from the GitHub Pages markdown guide."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
INPUT_MD = ROOT / "TUTORIAL_GITHUB_PAGES.md"
OUTPUT_PDF = ROOT / "GITHUB_PAGES_TUTORIAL.pdf"


def make_styles():
    """Create paragraph styles used in the generated PDF."""
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1F3F5B"),
            spaceBefore=6,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=14,
            spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=14,
            leftIndent=12,
            spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=9.2,
            leading=12,
            backColor=colors.HexColor("#EEF3F8"),
            leftIndent=6,
            rightIndent=6,
            spaceAfter=2,
        ),
    }


def sanitize(text: str) -> str:
    """Escape XML-sensitive characters for reportlab Paragraph rendering."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def line_to_flowable(line: str, styles, in_code: bool):
    """Convert one markdown line to a reportlab flowable."""
    text = line.rstrip("\n")

    if text.strip() == "```":
        return None

    if in_code:
        safe = sanitize(text) if text else " "
        return Paragraph(safe, styles["code"])

    if not text.strip():
        return Spacer(1, 0.22 * cm)

    if text.startswith("# "):
        return Paragraph(sanitize(text[2:].strip()), styles["h1"])

    if text.startswith("## "):
        return Paragraph(sanitize(text[3:].strip()), styles["h2"])

    if text.startswith("- "):
        body = sanitize(text[2:].strip())
        return Paragraph(f"&#8226; {body}", styles["bullet"])

    if text[:2].isdigit() and text[1] == ".":
        return Paragraph(sanitize(text), styles["body"])

    return Paragraph(sanitize(text), styles["body"])


def build_pdf():
    """Read the markdown tutorial and build the final PDF file."""
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title="GitHub Pages Tutorial",
        author="GitHub Copilot",
    )

    story = []
    in_code = False
    for line in INPUT_MD.read_text(encoding="utf-8").splitlines():
        if line.strip() == "```":
            in_code = not in_code
            story.append(Spacer(1, 0.12 * cm))
            continue

        flowable = line_to_flowable(line, styles, in_code)
        if flowable is not None:
            story.append(flowable)

    doc.build(story)
    print(f"PDF created: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
