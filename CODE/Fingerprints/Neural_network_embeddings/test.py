import torch
from torchvision import transforms
from dataset import FingerprintDatasetDirect
from network import SiameseNetwork
import os
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import argparse

def get_top_matches(test_image_path, model, dataset, transform, top_k=5, device='cpu'):
    test_image = Image.open(test_image_path).convert('RGB')
    test_image = transform(test_image).unsqueeze(0).to(device)

    model.eval()

    similarity_scores = []

    for img_path in tqdm(dataset.image_paths, desc="Matching", unit="img"):
        if os.path.isfile(img_path):
            db_image = Image.open(img_path).convert('RGB')
            db_image = transform(db_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output_db, output_test = model(db_image, test_image)
                output_db = F.normalize(output_db, p=2, dim=1)
                output_test = F.normalize(output_test, p=2, dim=1)

                similarity_score = torch.cosine_similarity(output_test, output_db).item()
                similarity_scores.append((img_path, similarity_score))

    # Sort and get top K
    top_matches = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_k]



    return top_matches


parser = argparse.ArgumentParser(prog='test.py',
                                 description="Identificarea unei amprente într-o colecție existentă",
                                 usage='test.py -m <model file> -df <images directory>',
                                 epilog="Programul extrage embedding-urile dintr-o imagine de test și le compară cu embedding-urile extrase cu aceeași rețea neuronală din imaginile din baza de date")


parser.add_argument('-m', "--model_file", required = True, help="Fișierul care conține modelul")
parser.add_argument('-if', "--input_file", required = True, help="Imaginea de test")
parser.add_argument('-id', "--input_directory", required = True, help="Directorul cu imagini ale bazei de date")
parser.add_argument('-e', "--embedding_size", default=128, help="Dimensiunea stratului de embeddings", type=int)
parser.add_argument('-k', "--top_k", default=5, help="Numărul de imagini afișate în topul celor care sunt asemănătoare", type=int)

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork(embedding_dim=args.embedding_size).to(device)

model.load_state_dict(torch.load(args.model_file, weights_only=True, map_location=torch.device(device)))
    
#data_dir = '/mnt/c/Users/smilutinovici/VisualStudioProjects/UTM/BioMetrics/DATA/fingerprints/NISTDB4_fragment'
dataset = FingerprintDatasetDirect(args.input_directory, transform)

#test_image_path = '/mnt/c/Users/smilutinovici/VisualStudioProjects/UTM/BioMetrics/DATA/fingerprints/NISTDB4_fragment/test_images/class1_Arc/class1_Arc_0010_v3.png'

top_matches = get_top_matches(args.input_file, model, dataset, transform, device=device)

top_k = 5

print(f"\nTop {top_k} matches:")
for i, (path, score) in enumerate(top_matches, 1):
    print(f"{i}. {path} — Score: {score:.4f}")

