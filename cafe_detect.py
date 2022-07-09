
# Presionar letra Q para detener
#·····································  Librerias Necesarias ·················································
import cv2
import numpy as np

verdeBajo = np.array([15,20,50],np.uint8) #Rango de color HSV : Vector inicial
verdeAlto = np.array([85,255,255],np.uint8)#Rango de color HSV: Vector final

#·············································· Proceso Deteccion ······································
cap = cv2.VideoCapture(0) #Captura de video
while cap.isOpened():   #Mientras la camara este abierta:
  ret,frame = cap.read() #Lectura de de la variable cap (Captura de video)
  if ret==True:          
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #Transformacion del espacio de color BGR a HSV
    mask = cv2.inRange(frameHSV,verdeBajo,verdeAlto) #Imagen binaria -> Con 1 Donde el color pertenece al rango y 0 en caso contrario) 
    
  
    bordes =  cv2.Canny(mask, 100, 101)                #Funcion Canny -> Algoritmo De deteccion de bordes -> https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

    contornos,_ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Funcion -> Encontrar contornos -> https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
    
    #cv2.drawContours(frame, contornos, -1, (0,0,255), 3)
    #cv2.drawContours(frame, contornos, -1, (255,0,0), 3)
    color = (200,0,0)                                #Color en espacio BGR
    #texto = 'Frutos Verdes: '+ str(len(contornos))    #Texto con el conteo de contornos
    #cv2.putText(frame, texto, (10,20), cv2.FONT_ITALIC, 0.7,color, 2, cv2.LINE_AA)#Impresion del texto en la imagen

    for c in contornos:
      area = cv2.contourArea(c)       #Calculo del area detectada en pixeles
      if area < 3000 and area >300:   #Si el Area esta dentro del rango:

        M = cv2.moments(c)            #Encuentra los momentos o el punto central de los contornos en pixeles https://unipython.com/caracteristicas-los-contornos/

        if (M["m00"]==0): M["m00"]=1  #Cuando el momento sea cero se iguala a uno para el cociente a continuacion
        x = int(M["m10"]/M["m00"])    #Coordenada X de los contornos
        y = int(M['m01']/M['m00'])    #Coordenada Y de los contornos

        cv2.circle(frame, (x,y), 7, (0,255,0), -1) #Se imprime un circulo sobre la imagen en las cooredenadas (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX            #Fuente texto - Estilo
        cv2.putText(frame, '{},{}'.format(x,y),(x+10,y), font, 0.3,(0,0,255),1,cv2.LINE_AA) #Imprime sobre la imagen las coordenadas x,y en la posicion x+10, y, 
        nuevoContorno = cv2.convexHull(c)           #Mejora la visualizacion del contorno
        cv2.drawContours(frame, [nuevoContorno], 0, (255,0,0), 3) #Dibuja el contorno

    cv2.imshow('Deteccion',mask) #Imprimir Imagen binaria 
    cv2.imshow('Rta',frame)    #Imprimir Imagen Con contornos y coordendas
    cv2.imshow('Canny',bordes)    #Imprimir Imagen Con contornos y coordendas
    if cv2.waitKey(1) & 0xFF == ord('q'): #Cuando se Precione la letra 'q' durante 1 ms
      break                               #Cierra el bucle
cap.release()                              #Se detiene la captura
cv2.destroyAllWindows()                    #Se cierran las ventanas