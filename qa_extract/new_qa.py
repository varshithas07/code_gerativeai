import fitz  # PyMuPDF
def extract_titles_from_pdf(pdf_path):
    # Open the PDF document
    document = fitz.open(pdf_path)
    titles = []

    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        # Iterate through each block on the page
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        flags = span["flags"]
                        
                        # Print intermediate data for debugging
                        print(f"Text: {text}, Font Size: {font_size}, Flags: {flags}")
                        
                        # Check for potential title conditions
                        if font_size > 15 or flags & 2:  # Font size threshold or bold text
                            titles.append(text.strip())

    return titles


# Example usage
pdf_path = "./data/e_gov_Policy_Document_GOI.pdf"
titles = extract_titles_from_pdf(pdf_path)
if titles:
    for title in titles:
        print(title)
else:
    print("No titles found.")