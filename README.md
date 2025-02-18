# Reconhecimento Facial com FaceNet, MTCNN e SVM

Este repositório contém um projeto de **Reconhecimento Facial** utilizando:
- [FaceNet](https://github.com/nyoki-mtl/keras-facenet) para geração de embeddings (representações vetoriais das faces).
- [MTCNN](https://github.com/ipazc/mtcnn) para detecção de faces em imagens.
- [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (via scikit-learn) para classificação das faces.

---

## Estrutura de Pastas

```
.
├── images
│   ├── Classes
│   │   ├── C_America
│   │   │   ├── imagem1.jpg
│   │   │   ├── imagem2.jpg
│   │   │   └── ...
│   │   └── H_Ferro
│   │       ├── imagem1.jpg
│   │       ├── imagem2.jpg
│   │       └── ...
│   └──  Vingadores-Ultimato.jpg
├── ReconhecimentoFacial.py
├── ExemploReconhecimentoFacial.py
└── README.md
```

- **`./images/Classes`**: Diretório que contém as subpastas, cada subpasta representando uma classe (pessoa).  
  - `C_America` e `H_Ferro` são exemplos de subpastas (classes) no dataset.  
  - Cada subpasta deve conter diversas imagens do rosto dessa pessoa.
- **`./images/Vingadores-Ultimato.jpg`**: Exemplo de imagem de grupo para teste do reconhecimento facial.

---

## Dependências

- Python 3.10.11
- [TensorFlow](https://www.tensorflow.org/install) (>= 2.x)
- [keras_facenet](https://pypi.org/project/keras-facenet/)
- [mtcnn](https://pypi.org/project/mtcnn/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [OpenCV](https://pypi.org/project/opencv-python/) (>= 4.x)
- [numpy](https://pypi.org/project/numpy/)
- [logging](https://docs.python.org/3/library/logging.html) (já incluso na stdlib do Python)

Para instalar rapidamente via `pip`, use:

```bash
pip install tensorflow keras_facenet mtcnn scikit-learn opencv-python numpy
```

> **Observação**: Dependendo da sua placa de vídeo, você pode querer instalar a versão GPU do TensorFlow. Consulte a [documentação oficial](https://www.tensorflow.org/install) para mais detalhes.

---

## Como Executar

1. **Clone ou baixe** este repositório.
2. **Organize** o dataset na pasta `./images/Classes`, com subpastas para cada classe (pessoa).  
   Exemplo:
   ```
   ./images/Classes
   ├── C_America
   │   ├── cap1.jpg
   │   ├── cap2.jpg
   │   └── ...
   └── H_Ferro
       ├── hf1.jpg
       ├── hf2.jpg
       └── ...
   ```
3. **Adicione** a imagem de grupo que deseja reconhecer na pasta `./images/` (ex.: `Vingadores-Ultimato.jpg`).
4. **Instale** as dependências (veja a seção anterior).
5. **Execute** o script principal:
   ```bash
   python main.py
   ```
   - O script:
     1. Lê todas as imagens em `./images/Classes`.
     2. Detecta as faces e extrai embeddings com FaceNet.
     3. Treina um classificador SVM.
     4. Detecta as faces na imagem de grupo (`Vingadores-Ultimato.jpg`) e classifica cada face, exibindo uma janela com o resultado.

---

## Explicação do Código

### `process_dataset(dataset_path, threshold=0.8)`
- Percorre cada subpasta (classe) em `dataset_path`.
- Carrega cada imagem, converte para RGB, e utiliza `FaceNet().extract` para detectar a face e gerar o embedding.
- Armazena o embedding e o rótulo (classe).
- `threshold` controla a sensibilidade da detecção interna do FaceNet (0.7 ou 0.8 costumam ser bons pontos de partida).

### `train_classifier(embeddings, labels)`
- Normaliza os embeddings com **L2** (muito útil para melhorar a separação entre classes).
- Divide os dados em treino e teste (`train_test_split`).
- Treina um **SVM** com kernel linear.
- Exibe um relatório de classificação (`classification_report`) no conjunto de teste.

### `recognize_faces_in_image(image_path, classifier, embedder, normalizer, threshold=0.9)`
- Carrega a imagem de grupo.
- Usa **MTCNN** para detectar todas as faces.
- Para cada face detectada, recorta e redimensiona para 160×160 (dimensão padrão do FaceNet).
- Extrai o embedding com `embedder.embeddings()`, aplica a **mesma normalização** L2 e usa o classificador treinado para prever a classe.
- Desenha retângulos e rótulos na imagem, exibindo-a em uma janela OpenCV.

### `ReconhecimentoFacial()`
- Define os caminhos do dataset (`dataset_path`) e da imagem de grupo (`group_image_path`).
- Extrai embeddings, treina o classificador e, por fim, faz o reconhecimento na imagem de grupo.

---

## Dicas de Uso

1. **Balanceamento das classes**: Se você tiver muitas imagens de uma classe e poucas de outra, o classificador pode tender a classificar tudo como a classe mais frequente.
2. **Variedade de imagens**: Use fotos em diferentes ângulos, expressões e condições de iluminação para melhorar a robustez do modelo.
3. **Qualidade das imagens**: Imagens muito pequenas, desfocadas ou com o rosto coberto podem não ser detectadas ou gerar embeddings ruins.
4. **Ajuste de thresholds**: Caso o detector do FaceNet ou o MTCNN não estejam detectando bem, tente valores de `threshold` diferentes (por exemplo, 0.7, 0.8, 0.9) e compare os resultados.
5. **Normalização**: Se você notar que o classificador confunde muito as classes, verifique se a normalização L2 dos embeddings está sendo aplicada tanto no treino quanto na predição.

---

## Contribuindo

Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias, correções de bugs ou novas funcionalidades. Toda contribuição é bem-vinda!

---
