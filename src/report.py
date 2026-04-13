from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import random

def generate_report(image_path, prediction, confidence, patient_name, patient_age, patient_gender, output_path="report.pdf"):

    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        textColor=colors.darkblue,
        alignment=1
    )

    normal_style = styles["Normal"]

    elements = []

    # HEADER
    elements.append(Paragraph("🏥 MEDICAL DIAGNOSIS REPORT", title_style))
    elements.append(Spacer(1, 12))

    report_id = f"MD-{random.randint(1000,9999)}"
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elements.append(Paragraph(f"<b>Report ID:</b> {report_id}", normal_style))
    elements.append(Paragraph(f"<b>Date:</b> {date}", normal_style))
    elements.append(Spacer(1, 12))

    # PATIENT INFO
    elements.append(Paragraph("👤 PATIENT DETAILS", styles["Heading2"]))
    elements.append(Paragraph(f"Name: {patient_name}", normal_style))
    elements.append(Paragraph(f"Age: {patient_age}", normal_style))
    elements.append(Paragraph(f"Gender: {patient_gender}", normal_style))
    elements.append(Spacer(1, 12))

    # DIAGNOSIS
    elements.append(Paragraph("🧠 DIAGNOSIS RESULT", styles["Heading2"]))
    elements.append(Paragraph(f"Disease: <b>{prediction}</b>", normal_style))
    elements.append(Paragraph(f"Confidence: <b>{confidence:.2f}</b>", normal_style))

    if confidence > 0.85:
        severity = "HIGH"
    elif confidence > 0.6:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    elements.append(Paragraph(f"Severity: <b>{severity}</b>", normal_style))
    elements.append(Spacer(1, 12))

    # IMAGE
    elements.append(Paragraph("📷 SCAN IMAGE", styles["Heading2"]))
    elements.append(Image(image_path, width=250, height=250))
    elements.append(Spacer(1, 12))

    # DISCLAIMER
    elements.append(Paragraph("⚠ DISCLAIMER", styles["Heading2"]))
    elements.append(Paragraph(
        "This AI-generated report is for educational purposes only.",
        normal_style
    ))

    doc.build(elements)