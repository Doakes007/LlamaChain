import pdfplumber
from langchain.schema import Document
import os


def extract_tables_from_pdf(file_path):

    documents = []
    filename = os.path.basename(file_path)

    try:
        with pdfplumber.open(file_path) as pdf:

            for page_number, page in enumerate(pdf.pages):

                tables = page.extract_tables()

                if not tables:
                    continue

                for table_id, table in enumerate(tables):

                    try:
                        # Skip invalid tables
                        if not table or len(table) < 2:
                            continue

                        headers = table[0]
                        rows = table[1:]

                        # Clean headers
                        headers = [
                            str(h).strip() if h else f"Column_{i}"
                            for i, h in enumerate(headers)
                        ]

                        table_text = "TABLE FROM DOCUMENT\n\n"

                        # -------------------------------
                        # Column Section
                        # -------------------------------
                        table_text += "Columns:\n"
                        table_text += " | ".join(headers) + "\n\n"

                        # -------------------------------
                        # Row Section
                        # -------------------------------
                        table_text += "Rows:\n"

                        for row in rows:

                            # Pad row if shorter than headers
                            if len(row) < len(headers):
                                row = row + [""] * (len(headers) - len(row))

                            row_dict = dict(zip(headers, row))

                            row_sentence = ", ".join(
                                f"{k}: {v}" for k, v in row_dict.items() if v
                            )

                            if row_sentence:
                                table_text += row_sentence + "\n"

                        # -------------------------------
                        # Metadata
                        # -------------------------------
                        metadata = {
                            "source": filename,
                            "page": page_number + 1,
                            "chunk_type": "table",
                            "table_id": table_id,
                            "preview": table_text[:200]
                        }

                        documents.append(
                            Document(
                                page_content=table_text.strip(),
                                metadata=metadata
                            )
                        )

                    except Exception as e:
                        print(f"Table parsing failed on page {page_number+1}:", e)
                        continue

    except Exception as e:
        print("PDF table extraction failed:", e)

    return documents