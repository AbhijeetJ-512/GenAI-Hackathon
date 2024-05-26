# image_embedder.py
import torch
from PIL import Image
from torchvision import transforms
from database import batch_data

class ImageEmbedder:
    def __init__(self, clip_model):
        self.clip_model = clip_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image = self.transform(image).unsqueeze(0)
        return image

    def process_and_encode_images(self, data, index):
        print("Started image embedding...")
        embeddings_data = []
        image_counter = 0

        for key, value in data.items():
            pdf_name, page_num = key.split('_page_')
            images = value.get("images", [])

            for i, image_path in enumerate(images):
                image = self.preprocess_image(image_path).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                image_features_list = image_features.squeeze().tolist()
                metadata = {
                    "pdf name": pdf_name,
                    "page no": page_num,
                    "image no": i,
                    "Image path": image_path
                }
                embeddings_data.append({
                    "id": f"{pdf_name}_{page_num}_{i}",
                    "values": image_features_list,
                    "metadata": metadata
                })

                image_counter += 1
                if image_counter % 100 == 0:
                    print(f"{image_counter} images processed so far...")

        print(f"Total {image_counter} images processed.")
        batch_size = 100
        for batch in batch_data(embeddings_data, batch_size=batch_size):
            index.upsert(vectors=batch)
        print("Image embeddings uploaded successfully!")
