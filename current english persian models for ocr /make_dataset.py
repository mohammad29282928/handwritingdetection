import pandas as pd

# تابعی برای خواندن تصویر به صورت بایت
def image_to_bytes(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

# خواندن داده‌ها از فایل Excel
excel_file_path = 'path/to/your/file.xlsx'
df_excel = pd.read_excel(excel_file_path)

# فرض می‌کنیم فایل Excel دارای دو ستون است: 'image_path' و 'text'
image_paths = df_excel['image_path'].tolist()
texts = df_excel['text'].tolist()

# اطمینان از برابر بودن تعداد تصاویر و متن‌ها
assert len(image_paths) == len(texts), "تعداد تصاویر و متن‌ها باید برابر باشد."

# تبدیل تصاویر به فرمت بایت و ساخت دیتا فریم
data = {
    'image_path': [{'bytes': image_to_bytes(image)} for image in image_paths],
    'text': texts
}

df = pd.DataFrame(data)

# نمایش داده‌ها
print("داده‌های ایجاد شده:")
print(df)

# ذخیره دیتا فریم به صورت فایل Parquet
output_file_path = 'path/to/output/file.parquet'
df.to_parquet(output_file_path, engine='pyarrow')

print(f"فایل Parquet با موفقیت ذخیره شد: {output_file_path}")
