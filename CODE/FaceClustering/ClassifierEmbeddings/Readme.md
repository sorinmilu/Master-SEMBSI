# ClassifierEmbeddings

Programul demonstreaza posibilitatea grupării fețelor pe baza vectorului de embeddings extras din penultimul strat al unui model de clasificare. 

Modelele de clasificare au un strat final care conține un număr de neuroni corespunzător cu numărul de categorii poe care dorim să le identificăm. In cazul modelelor utilizate aici, ultimul strat conține un singur neuron, acesta putând lua valori între 0 și 1. Cu cât valoarea acestuia este mai aproape de 1, cu atât imaginea pe care a analizat-o are șanse mai mari să fie o față.

Ultimul strat este calculat pe baza unei sume ponderate, cu ponderi antrenate, a valorilor neuronilor din stratul preliminar. Acesta este tot un strat complet conectat, liniar, care poate avea un număr variabil de neuroni. Acest penultim strat, numit "embeddings" conține o serie de numere care identifică obiectele din imagine. Două imagini similare vor avea valori similare în acest vector. Din acest motiv, acest vector poate fi utilizat pentru identificarea similarității fețelor. 

Odată modelul antrenat, pornind de la o colecție de fețe care aparțin câtorva personaje, utilizând vectorul de embeddings, putem grupa fețele în funcție de personaj. 

Progamul se ruleaza astfel: 

```bash
usage: infer_folder.py -id <input_directory> -m <model file> -e <embedding size>

Program that extracts the embeddings from a folder of images using a CNN model. The model has to be trained with the same architecture as the one used for inference. The program also generates t-SNE and UMAP charts of the embeddings.

options:
  -h, --help            show this help message and exit
  -id INPUT_DIRECTORY, --input_directory INPUT_DIRECTORY
                        Directory with the testing images. Can have subdiorectories
  -e EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
                        size of the last fully connected layer
  -m MODEL, --model MODEL
                        File that holds the model
  -os OUTPUT_TSNE, --output_tsne OUTPUT_TSNE
                        Name of the t-sne chart. No extension (png will be added, together with the size of the embeddings vector)
  -ou OUTPUT_UMAP, --output_umap OUTPUT_UMAP
                        Name of the umap chart. No extension (png will be added, together with the size of the embeddings vector)

The structure of the directory has to be very precise, it has to have one level of subdirectories and within those subdirectories the images.
The images have to be extracted faces (not casual images)
    \../../DATA/face/clustering/short/
    ├── face1
    ├── face2
    └── face3
```


