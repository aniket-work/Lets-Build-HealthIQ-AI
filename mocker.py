from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_medical_pdf(filename):
    # Create a canvas object
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Set the title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 50, "Medical Knowledge Base")

    # Set the subtitle
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 80, "For Specialized Medical Chatbot")

    # Medical content
    medical_content = [
        ("Common Diseases", [
            "1. Hypertension: A condition in which the blood pressure is consistently high.",
            "2. Diabetes: A chronic condition that affects how your body turns food into energy.",
            "3. Asthma: A condition in which a person's airways become inflamed, narrow, and swell, producing extra mucus."
        ]),
        ("Symptoms and Treatments", [
            "Hypertension: Symptoms include headaches, shortness of breath, and nosebleeds. Treatment involves lifestyle changes and medication.",
            "Diabetes: Symptoms include increased thirst, frequent urination, and fatigue. Treatment involves insulin injections and dietary changes.",
            "Asthma: Symptoms include wheezing, chest tightness, and shortness of breath. Treatment involves inhalers and avoiding triggers."
        ]),
        ("Preventive Measures", [
            "Regular exercise and a balanced diet can help prevent hypertension and diabetes.",
            "Avoiding allergens and irritants can help manage asthma.",
            "Regular check-ups and screenings are essential for early detection and prevention of diseases."
        ])
    ]

    # Draw the medical content
    y = height - 120
    for section, content in medical_content:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, section)
        y -= 20
        c.setFont("Helvetica", 12)
        for line in content:
            c.drawString(70, y, line)
            y -= 20
        y -= 10

    # Save the PDF file
    c.save()

# Create the PDF
create_medical_pdf("medical_knowledge_base.pdf")
