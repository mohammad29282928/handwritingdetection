pip install aspose-pdf
import aspose.pdf as ap
import pandas as pd

input_pdf = "C:/Users/RGS/Downloads/Book1.pdf"
output_pdf = "C:/Users/RGS/Downloads/qwe.xlsx"

# Open PDF document
document = ap.Document(input_pdf)

# Set save options
save_option = ap.ExcelSaveOptions()

# Save the file into MS Excel format
document.save(output_pdf, save_option)

# Load the Excel file into a pandas DataFrame without using the first row as header
df = pd.read_excel(output_pdf, header=None)

# Drop the first row (index 0) of the DataFrame
df = df.drop(index=0)

# Save the modified DataFrame back to Excel without the index
df.to_excel(output_pdf, index=False, header=False)