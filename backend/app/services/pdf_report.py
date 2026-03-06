"""PDF report generator using ReportLab."""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from app.core.logging import get_logger

logger = get_logger(__name__)

# Brand colours
PRIMARY = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#e94560")
HEADER_BG = colors.HexColor("#16213e")
ROW_ALT = colors.HexColor("#f5f5f5")
SEVERITY_COLORS = {
    "low": colors.HexColor("#27ae60"),
    "medium": colors.HexColor("#f39c12"),
    "high": colors.HexColor("#e74c3c"),
    "none": colors.grey,
}


def _severity_color(s: str):
    return SEVERITY_COLORS.get(s, colors.grey)


def generate_pdf(
    prediction: Dict[str, Any],
    car_model: Optional[str],
    panel_location: Optional[str],
    notes: Optional[str],
    overlay_image_path: Optional[str] = None,
) -> bytes:
    """
    Build a PDF report for a car damage prediction.

    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], textColor=PRIMARY, fontSize=20, spaceAfter=6
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], textColor=ACCENT, spaceAfter=4
    )
    normal_style = styles["Normal"]
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8)

    elements = []

    # ── Header ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("🚗 Car Damage Assessment Report", title_style))
    elements.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Currency: PKR | Powered by Azeem Aslam",
            small_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    elements.append(Spacer(1, 4 * mm))

    # ── Summary ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("Executive Summary", h2_style))
    summary = prediction.get("summary", {})
    cost = summary.get("estimated_cost_pkr", {})
    overall_sev = prediction.get("overall_severity", "none")

    summary_data = [
        ["Car Model", car_model or "Not specified"],
        ["Panel / Location", (panel_location or "unknown").capitalize()],
        ["Damage Detected", "Yes" if prediction.get("damage_detected") else "No"],
        ["Total Instances", str(summary.get("total_instances", 0))],
        ["Total Damage Area", f"{summary.get('total_damage_percent', 0):.2f}%"],
        ["Overall Severity", overall_sev.upper()],
        ["Estimated Cost (min)", f"PKR {cost.get('min', 0):,.0f}"],
        ["Estimated Cost (max)", f"PKR {cost.get('max', 0):,.0f}"],
    ]

    summary_table = Table(summary_data, colWidths=[65 * mm, 105 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, ROW_ALT]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 6 * mm))

    # ── Overlay image ────────────────────────────────────────────────────────
    if overlay_image_path and os.path.exists(overlay_image_path):
        elements.append(Paragraph("Damage Overlay", h2_style))
        try:
            img = RLImage(overlay_image_path, width=170 * mm, height=95 * mm, kind="proportional")
            elements.append(img)
        except Exception as exc:
            logger.warning(f"Could not embed overlay image: {exc}")
        elements.append(Spacer(1, 4 * mm))

    # ── Detections table ─────────────────────────────────────────────────────
    detections = prediction.get("detections", [])
    if detections:
        elements.append(Paragraph("Detailed Detections", h2_style))
        headers = ["Class", "Confidence", "Area %", "Severity", "Cost Min (PKR)", "Cost Max (PKR)"]
        rows = [headers]
        for det in detections:
            c = det.get("cost_pkr", {})
            rows.append(
                [
                    det.get("class", det.get("class_name", "?")),
                    f"{det.get('confidence', 0):.2f}",
                    f"{det.get('damage_area_percent', 0):.3f}%",
                    det.get("severity", "?").upper(),
                    f"{c.get('min', 0):,.0f}",
                    f"{c.get('max', 0):,.0f}",
                ]
            )

        det_table = Table(rows, colWidths=[35*mm, 25*mm, 20*mm, 25*mm, 35*mm, 35*mm])
        sev_style_cmds: list = []
        for i, det in enumerate(detections, start=1):
            sev = det.get("severity", "none")
            sev_style_cmds.append(
                ("TEXTCOLOR", (3, i), (3, i), _severity_color(sev))
            )

        det_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
                + sev_style_cmds
            )
        )
        elements.append(det_table)
        elements.append(Spacer(1, 6 * mm))

    # ── Notes ────────────────────────────────────────────────────────────────
    if notes:
        elements.append(Paragraph("Additional Notes", h2_style))
        elements.append(Paragraph(notes, normal_style))
        elements.append(Spacer(1, 4 * mm))

    # ── Assumptions ─────────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Spacer(1, 2 * mm))
    elements.append(
        Paragraph(
            "<b>Assumptions:</b> Costs are estimated based on average PKR repair rates in Pakistan. "
            "Actual costs may vary by workshop, city, and material availability. "
            "Severity is determined by the percentage of image area affected by each damage type. "
            "This report is AI-generated and should be verified by a qualified assessor.",
            small_style,
        )
    )

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    logger.info(f"PDF report generated ({len(pdf_bytes)} bytes)")
    return pdf_bytes
