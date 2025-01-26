from PIL import Image
def check_corrupt(df):
    invalid_images = []

    for idx, row in df.iterrows():
        file_path = row["img_path"]
        try:
            with Image.open(file_path) as img:
                img.verify()  # Ensure the file is a valid image
        except Exception as e:
            print(f"Invalid image: {file_path} - Error: {e}")
            invalid_images.append(file_path)

    return invalid_images

