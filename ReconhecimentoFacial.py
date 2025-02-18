import os
import cv2
import numpy as np
import logging
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = FaceNet()

def process_dataset(dataset_path, threshold=0.8):

    embeddings = []
    labels = []
    
    if not os.path.exists(dataset_path):
        logger.error(f"O caminho do dataset '{dataset_path}' não existe.")
        return np.array(embeddings), np.array(labels)
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        
        image_files = os.listdir(person_path)
        logger.info(f"Processando {len(image_files)} imagens para a classe '{person}'")
        
        for filename in image_files:
            file_path = os.path.join(person_path, filename)
            logger.info(f"Processando: {file_path}")
            img = cv2.imread(file_path)
            if img is None:
                logger.warning(f"Não foi possível carregar a imagem: {file_path}")
                continue
            
            # Converte de BGR para RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extrai os embeddings (o método extract já faz a detecção interna)
            detections = embedder.extract(img_rgb, threshold=threshold)
            if detections and len(detections) > 0:
                embedding = detections[0]['embedding']
                embeddings.append(embedding)
                labels.append(person)
                logger.info(f"Face detectada em {file_path}")
            else:
                logger.warning(f"Nenhuma face detectada em {file_path}")
    return np.array(embeddings), np.array(labels)

def train_classifier(embeddings, labels):
    if len(embeddings) == 0:
        logger.error("Nenhum embedding disponível para treinar o classificador.")
        return None
    
    normalizer = Normalizer(norm='l2')
    embeddings_norm = normalizer.transform(embeddings)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_norm, labels, test_size=0.2, random_state=42
    )
    
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    logger.info("Relatório de Classificação:\n" + classification_report(y_test, y_pred))
    return classifier, normalizer

def recognize_faces_in_image(image_path, classifier, embedder, normalizer, threshold=0.9):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Não foi possível carregar a imagem de grupo: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inicializa o detector MTCNN
    detector = MTCNN()
    detections = detector.detect_faces(image_rgb)
    if len(detections) == 0:
        logger.warning("Nenhuma face detectada na imagem de grupo.")
        return
    
    for face in detections:
        x, y, width, height = face['box']
        # Corrige valores negativos, se houver
        x, y = abs(x), abs(y)
        # Recorta a face detectada
        face_crop = image_rgb[y:y+height, x:x+width]
        # Redimensiona para 160x160 (tamanho esperado pelo FaceNet)
        try:
            face_crop = cv2.resize(face_crop, (160, 160))
        except Exception as e:
            logger.warning(f"Erro ao redimensionar face: {e}")
            continue
        
        face_crop = face_crop.astype('float32') / 255.0
        # Extrai o embedding da face utilizando o método embeddings (passa uma lista de imagens)
        embedding = embedder.embeddings([face_crop])[0]
        # Aplica a normalização (L2) antes de classificar
        embedding_norm = normalizer.transform([embedding])[0]
        predicted_label = classifier.predict([embedding_norm])[0]
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(
            image, predicted_label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
    
    cv2.imshow("Reconhecimento de Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    dataset_path = "./images/Classes"
    group_image_path = "./images/Vingadores-Ultimato.jpg"
    
    logger.info("Extraindo embeddings do dataset...")
    embeddings, labels = process_dataset(dataset_path, threshold=0.8)
    logger.info(f"Extraídos {len(embeddings)} embeddings para {len(np.unique(labels))} classes.")
    
    classes, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(classes, counts):
        logger.info(f"Classe '{cls}': {cnt} imagens")
    
    if len(embeddings) == 0:
        logger.error("Nenhum embedding foi extraído. Verifique o dataset e os parâmetros de detecção.")
        return
    
    logger.info("Treinando classificador...")
    classifier, normalizer = train_classifier(embeddings, labels)
    if classifier is None:
        logger.error("Falha no treinamento do classificador.")
        return
    
    logger.info("Realizando reconhecimento na imagem de grupo...")
    recognize_faces_in_image(group_image_path, classifier, embedder, normalizer, threshold=0.8)

if __name__ == '__main__':
    main()
