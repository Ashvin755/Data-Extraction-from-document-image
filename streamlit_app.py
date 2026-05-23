import os
import json
import time
from io import BytesIO

import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ServerError

PROMPT =  """
You are an expert document processing AI. Analyze the provided image of a transport consignment note/invoice. Your task is to extract specific target fields and return them in a structured JSON format. 

Follow these strict visual processing rules to handle mixed handwriting, printing, and overlapping stamps:

### 1. FIELD EXTRACTION RULES:
- **invoice_number**: 
    * Look for the box labeled "INVOICE NO.". Extract the continuous sequence of numbers written there.
    * The number may physically overflow past the right boundary line of its box; treat the overflowing digits as part of the same continuous number.
    * CRITICAL: The true system invoice number must NEVER contain a forward slash ('/'). If you encounter a slash or a character that looks like a slash, interpret it as the number '1' and merge it seamlessly with the rest of the digits.
    * Ignore any neighboring alphabetical text (like the letter 'y' from the "FROM" box).

- **invoice_value**:
    * Look for the box explicitly labeled "INVOICE VALUE". 
    * Extract the total monetary valuation of the goods listed in this box.
    * If the document is handwritten, ignore trailing currency designators, formatting lines, or suffix dashes (such as "/-" or "—"). Extract only the core numerical amount.
    * Convert the final value to a clean decimal string format (e.g., "8939.76" or "108336"). Do not include currency symbols (Rs, $).

- **date**: 
    * Look for the field explicitly labeled "DATE & TIME" or "BOOKING DATE & TIME". Extract only the date portion in a clean format (e.g., DD/MM/YY or DD-MMM-YYYY).
    * CRITICAL: Do NOT extract the "EXPECTED DELIVERY DATE". Always prioritize the actual booking/creation date.

- **consignor**: 
    * Extract the full company name and shipping address listed under the "CONSIGNOR" . Clean up any trailing text or address details if they bleed across lines.

- **consignee**: 
    * Extract the full company name and shipping address listed under the "CONSIGNEE (SHIPPED TO)" or "TO" label.

- **is_sealed**: 
    * Scan the document for an official inked corporate/security stamp or verification seal (such as the purple "WELLSUN TEXCHEM PVT LTD" or "ACE REALTORS" stamps). 
    * Return `true` if a physical ink stamp is visible anywhere on the document, otherwise return `false`.

- **document_type**:
    * Classify the overall document format. Return the exact string "handwritten" if the primary billing information (Consignor, Consignee, Invoice No, and Date) is written by hand.
    * Return the exact string "printed" if these core transactional fields are filled out via computer typography or machine text.
    * CRITICAL CLASSIFICATION RULE: Ignore the presence of handwritten text inside security or gate entry stamps at the bottom when making this decision. Focus strictly on the primary billing boxes.

### 2. OUTPUT FORMAT CONSTRAINT:
Return the output strictly as a valid JSON object. Do not include any conversational introductions, markdown code block wrappers (like ```json), or trailing explanations. Output the raw JSON text only.

Expected JSON Schema:
{
    "invoice_number": "string",
    "invoice_value": "string",
    "date": "string",
    "consignor": "string",
    "consignee": "string",
    "is_sealed": boolean,
    "document_type": "string"
}
"""


def extract_invoice_json(
    image: Image.Image,
    api_keys: list[str],
    max_retries: int = 3,
) -> dict:
    last_error: Exception | None = None

    for api_key in api_keys:
        client = genai.Client(api_key=api_key)
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[image, PROMPT],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                return json.loads(response.text)
            except ServerError as exc:
                last_error = exc
                if "503" in str(exc) and attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                break
            except Exception as exc:
                last_error = exc
                break

    raise RuntimeError("All API keys failed to extract invoice data.") from last_error


def get_api_keys() -> list[str]:
    keys: list[str] = []

    secret_keys = st.secrets.get("GEMINI_API_KEYS", [])
    if isinstance(secret_keys, str):
        secret_keys = [secret_keys]
    keys.extend([key.strip() for key in secret_keys if str(key).strip()])

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    return keys


def main() -> None:
    st.set_page_config(page_title="Invoice Extraction", layout="wide")
    st.title("Invoice Extraction")

    api_keys = get_api_keys()
    if not api_keys:
        st.warning(
            "Add GEMINI_API_KEYS in .streamlit/secrets.toml or set GEMINI_API_KEY "
            "as an environment variable."
        )

    uploaded = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])
    if not uploaded:
        st.info("Upload an image to get started.")
        return

    image = Image.open(BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded invoice", use_container_width=True)

    if st.button("Extract JSON", type="primary", disabled=not api_keys):
        with st.spinner("Extracting data..."):
            try:
                data = extract_invoice_json(image, api_keys=api_keys)
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                return

        st.subheader("Extracted JSON")
        st.json(data)


if __name__ == "__main__":
    main()
