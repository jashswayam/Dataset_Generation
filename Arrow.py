# 1. Define your file names
input_txt = "a.txt"
output_docx = "a.docx"

try:
    # 2. Read the Base64 text string
    with open(input_txt, "r") as text_file:
        base64_string = text_file.read()
       
    # 3. Decode the string back into raw binary data
    binary_data = base64.b64decode(base64_string)
   
    # 4. Save the binary data as a new DOCX file ('wb' mode)
    with open(output_docx, "wb") as doc_file:
        doc_file.write(binary_data)
       
    print(f"Success! Your file has been rebuilt as {output_docx}.")

except FileNotFoundError:
    print(f"Error: Could not find {input_txt}.")
except base64.binascii.Error:
    print("Error: The text file doesn't contain valid Base64 data. Did you copy all of it?")