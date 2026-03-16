import pdfplumber
from langchain.schema import Document
import os


def extract_tables_from_pdf(file_path):

    documents = []
    filename = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:

        for page_number, page in enumerate(pdf.pages):

            tables = page.extract_tables()

            for table_id, table in enumerate(tables):

                if not table or len(table) < 2:
                    continue

                headers = table[0]
                rows = table[1:]

                table_text = "Table extracted from document.\n\n"

                table_text += "Columns:\n"
                table_text += " | ".join(str(h) for h in headers if h) + "\n\n"

                table_text += "Rows:\n"

                for row in rows:

                    row_dict = dict(zip(headers, row))

                    row_sentence = ", ".join(
                        f"{k}: {v}" for k, v in row_dict.items() if v
                    )

                    table_text += row_sentence + "\n"

                metadata = {
                    "source": filename,
                    "page": page_number + 1,
                    "chunk_type": "table",
                    "table_id": table_id,
                    "preview": table_text[:200]
                }

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata=metadata
                    )
                )

    return documents