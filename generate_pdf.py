from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Technical Report: Multi-Object Tracking', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    with open("report.md", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple writing
    text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, text)
    
    pdf.output("report.pdf")
    print("PDF generated successfully: report.pdf")

if __name__ == "__main__":
    create_pdf()
