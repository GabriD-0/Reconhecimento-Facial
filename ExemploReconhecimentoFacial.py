import cv2
import face_recognition as fr
import numpy as np

# Aplicação apenas para aprendizado

# Carregando imagens e extraindo os rostos/encodigs
imagemP1 = fr.load_image_file('./images/C_America/CapitaoAmerica.jpg')
imagemP1 = cv2.cvtColor(imagemP1, cv2.COLOR_BGR2RGB)
encodingP1 = fr.face_encodings(imagemP1)[0]
# cv2.imshow("Capitao America", imagemP1)

imagemP2 = fr.load_image_file('./images/GaviaoArqueiro.jpg')
imagemP2 = cv2.cvtColor(imagemP2, cv2.COLOR_BGR2RGB)
encodingP2 = fr.face_encodings(imagemP2)[0]
# cv2.imshow("Gavião Arqueiro", imagemP2)

imagemP3 = fr.load_image_file('./images/H_Ferro/HomemFerro.jpg')
imagemP3 = cv2.cvtColor(imagemP3, cv2.COLOR_BGR2RGB)
encodingP3 = fr.face_encodings(imagemP3)[0]
# cv2.imshow("Homem de Ferro", imagemP3)

imagemP4 = fr.load_image_file('./images/HomemFormiga.jpg')
imagemP4 = cv2.cvtColor(imagemP4, cv2.COLOR_BGR2RGB)
encodingP4 = fr.face_encodings(imagemP4)[0]
# cv2.imshow("Homem Formiga", imagemP4)

imagemP5 = fr.load_image_file('./images/JamesRhodey1.png')
imagemP5 = cv2.cvtColor(imagemP5, cv2.COLOR_BGR2RGB)
encodingP5 = fr.face_encodings(imagemP5)[0]
# cv2.imshow("James Rhodey", imagemP5)

# Armazenando os rostos/encodigs e nomes
conhecidos_encodigs = [
    encodingP1,
    encodingP2,
    encodingP3,
    encodingP4,
    encodingP5
]
conhecidos_nomes = [
    "C-America",
    "G-Arqueiro",
    "H-Ferro",
    "H-Formiga",
    "J-Rhodey"
]

# Carregando a imagem para reconhecimento do grupo
imagemGrupo = fr.load_image_file('./images/Vingadores-Ultimato.jpg')
imagemGrupo = cv2.cvtColor(imagemGrupo, cv2.COLOR_BGR2RGB)


# Reconhecendo os rostos na imagem
faces_localizadas = fr.face_locations(imagemGrupo)
encodigs_conhecidos = fr.face_encodings(imagemGrupo, faces_localizadas) 

# Comparação das encodigs / faces encontradas
for (top, right, bottom, left), face_encoding in zip(faces_localizadas, encodigs_conhecidos):
    resultados = fr.compare_faces(conhecidos_encodigs, face_encoding)
    nome = "Desconhecido"
    
    
    distancia = fr.face_distance(conhecidos_encodigs, face_encoding)
    melhor_distancia = np.argmin(distancia)
    
    if resultados[melhor_distancia]:
        nome = conhecidos_nomes[melhor_distancia]
    
    # Desenhando o retangulo e o nome do individuo
    cv2.rectangle(imagemGrupo, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(imagemGrupo, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
cv2.imshow("Reconhecimento de Faces", imagemGrupo)
cv2.waitKey(0)