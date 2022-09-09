import statistics as st
import cv2
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import os
#from PyPDF2 import PdfReader
#%matplotlib inline
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\epenago\AppData\Local\Tesseract-OCR\tesseract.exe'


def histograma(datos):
    datos.sort()
    resultantList = []
 
    for element in datos:
        if element not in resultantList:
            resultantList.append(element)
    
    print(resultantList)
    print("tamaño de lista resultante: ", len(resultantList))

          
    datos = np.array(datos)
    hist = np.histogram(datos, bins=len(resultantList))
    print(hist[0])
    print("tamaño histograma en y: ", len(hist[0]))
    print()
    print(hist[1])
    print("tamaño histograma en X: ", len(hist[1]))
    moda = st.mode(datos)
    mean = st.mean(datos) 
    #sd = st.stdev(datos)

    """
    print("la moda de los espacios es: " + str(moda))
    print("la media de los espacios es: " + str(mean))
    print("la SD de los espacios es: " + str(sd))
    #print("los datos ordenados son: ", datos)
    """
    
    
    plt.hist(x=datos, bins= len(resultantList), color='#F2AB6D', rwidth=0.95)
    #plt.hist(hist, bins= len(resultantList))
    
    plt.title('Histograma de frecuencias espacios')
    plt.xlabel('Pixeles')
    plt.ylabel('Frecuencia')
    #plt.xticks(datos)
    plt.show() 
    
    #return (moda-(moda*1), moda+(moda*1))

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


def coordenadas_extraccion(diccionario, indices, intervalo):
    
    datos_extraidos = []
    dict = {}
    coordenadas = []
    suma = 0
    bandera = True
    indice_anterior = -1
    strings = []
    string = ""
    index = []
    line = 0

    for i in range(len(indices)):
             
        if(bandera == True):
            indice_anterior = i
            
            bandera = False
            suma = 0
            suma += int(diccionario["left"][indices[i]])+int(diccionario["width"][indices[i]])

            # Se guarda la palabra y su indice
            string = diccionario["text"][indices[i]]
            index.append(indices[i])
            
            if i < len(indices)-1:
                if (int(diccionario["top"][indices[i]]) >= int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]])+10 or int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) > int(diccionario["top"][indices[i+1]])-10):
                    
                    if (diccionario["left"][indices[i+1]] - suma) < intervalo[0] or (diccionario["left"][indices[i+1]] - suma) > intervalo[1]:
                        bandera = True
                        #print(indices[i])
                        coordenadas.append([diccionario["left"][indices[indice_anterior]], 
                        diccionario["top"][indices[indice_anterior]], 
                        suma,
                        diccionario["top"][indices[i]]+diccionario["height"][indices[i]]])
                        
                        dict["text"] = string
                        dict["line"] = line
                        dict["index"] = index
                        dict["coordinates"] = (diccionario["left"][indices[indice_anterior]], suma)
                        datos_extraidos.append(dict)
                                                                    
                        dict = {}                        
                        string = ""
                        index = []

                        bandera = True
                        indice_anterior = -1
                else:
                    bandera = True
                    #print(indices[i])
                    coordenadas.append([diccionario["left"][indices[indice_anterior]], 
                    diccionario["top"][indices[indice_anterior]], 
                    suma,
                    diccionario["top"][indices[i]]+diccionario["height"][indices[i]]])
                    

                    dict["text"] = string
                    dict["line"] = line
                    dict["index"] = index
                    dict["coordinates"] = (diccionario["left"][indices[indice_anterior]], suma)
                    datos_extraidos.append(dict)
                                                                
                    dict = {}                        
                    string = ""
                    index = []
                    line += 1

                    bandera = True
                    indice_anterior = -1
            else:           
                    coordenadas.append([diccionario["left"][indices[indice_anterior]], 
                    diccionario["top"][indices[indice_anterior]], 
                    suma,
                    diccionario["top"][indices[i]]+diccionario["height"][indices[i]]])

                    dict["text"] = string
                    dict["line"] = line
                    dict["index"] = index
                    dict["coordinates"] = (diccionario["left"][indices[indice_anterior]], suma)
                    datos_extraidos.append(dict)
                                                                
                    dict = {}                        
                    string = ""
                    index = []

                    bandera = True
                    indice_anterior = -1

        elif(i < len(indices)-1):
            string += (" " + diccionario["text"][indices[i]])
            index.append(indices[i])

            suma += int(diccionario["width"][indices[i]] + (diccionario["left"][indices[i]] - suma))
            if (int(diccionario["top"][indices[i]]) >= int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]])+10 or int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) > int(diccionario["top"][indices[i+1]])-10):
                
                if (diccionario["left"][indices[i+1]] - suma) < intervalo[0] or (diccionario["left"][indices[i+1]] - suma) > intervalo[1]:
                    bandera = True
                    #print(indices[i])
                    coordenadas.append([diccionario["left"][indices[indice_anterior]], 
                    diccionario["top"][indices[indice_anterior]], 
                    suma,
                    diccionario["top"][indices[i]]+diccionario["height"][indices[i]]])

                    dict["text"] = string
                    dict["line"] = line
                    dict["index"] = index
                    dict["coordinates"] = (diccionario["left"][indices[indice_anterior]], suma)
                    datos_extraidos.append(dict)
                                                                
                    dict = {}                        
                    string = ""
                    index = []

                    bandera = True
                    indice_anterior = -1
            else:
                bandera = True
                #print(indices[i])
                coordenadas.append([diccionario["left"][indices[indice_anterior]], 
                diccionario["top"][indices[indice_anterior]], 
                suma,
                diccionario["top"][indices[i]]+diccionario["height"][indices[i]]])

                dict["text"] = string
                dict["line"] = line
                dict["index"] = index
                dict["coordinates"] = (diccionario["left"][indices[indice_anterior]], suma)
                datos_extraidos.append(dict)
                                                            
                dict = {}                        
                string = ""
                index = []
                line += 1

                bandera = True
                indice_anterior = -1

        elif(i <= len(indices)-1):
            suma += int(diccionario["width"][indices[i]] + (diccionario["left"][indices[i]] - suma))
            coordenadas.append([diccionario["left"][indices[indice_anterior]], 
            diccionario["top"][indices[indice_anterior]], 
            suma,
            diccionario["top"][indices[i]]+diccionario["height"][indices[i]]])

            string = ""
            index = []

            bandera = True
            indice_anterior = -1

    return (coordenadas, datos_extraidos)


def calcular_espacios(diccionario, indices):
    
    suma = 0
    bandera = True
  
    width_espacios = []
    for i in range(len(indices)):
             
        if(bandera == True):
            
            bandera = False
            suma = 0
            suma += int(diccionario["left"][indices[i]])+int(diccionario["width"][indices[i]])

            if i < len(indices)-1:
                if int(diccionario["top"][indices[i]]) >= int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]])+10 or int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) > int(diccionario["top"][indices[i+1]])-10:
                    pass
                else:
                    bandera = True
            else:           
                    bandera = True
                    

        elif(i < len(indices)-1):
            width_espacios.append(diccionario["left"][indices[i]] - suma)
            suma += int(diccionario["width"][indices[i]] + (diccionario["left"][indices[i]] - suma))
            if int(diccionario["top"][indices[i]]) >= int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]])+10 or int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) > int(diccionario["top"][indices[i+1]])-10:
                pass
            else:
                bandera = True
               

        elif(i <= len(indices)-1):
            width_espacios.append(diccionario["left"][indices[i]] - suma)
            suma += int(diccionario["width"][indices[i]] + (diccionario["left"][indices[i]] - suma))
            bandera = True
            
    #print(width_espacios)
    width_espacios.sort() 
    
    moda = st.mode(width_espacios)
    return (moda-(moda*1), moda+(moda*1))


def num_column(datos):
    
    sum = 0
    line = 0
    arr = []
    for i in range(len(datos)):

        if i != 0 and datos[i]["line"]==line:
            sum += 1
        elif datos[i]["line"] != line:
            arr.append(sum+1)
            sum = 0
            line = datos[i]["line"]
          
    return st.mode(arr)


def built_table(columns, data):

    coordinates = []
    for i in range(len(data)):
       coordinates.append(data[i]["coordinates"])

    coordinates = [(i[0]+i[1])//2 for i in coordinates]
    coordinates.sort()
    #print(coordinates)
    #print("tamaño original: ", len(coordinates))
    histograma(coordinates)
    #coordinates.sort()
    #print(coordinates)
    


def main():
    path = r'C:\Users\epenago\Pictures\reto\4.PNG'
    #path = '.\pagina2.PNG'
    image = cv2.imread(path)
    gray = get_grayscale(image)
    th = thresholding(gray)
    openingg = opening(gray)
    cannyy = canny(gray)
    img = image
    hImg, wImg, _ = img.shape
    
    custom_config = r'-c tessedit_char_blacklist=$¢§_{} --oem 3 --psm 4' 
    data = pytesseract.image_to_data(img, lang= 'eng+spa', config=custom_config, output_type= 'dict')
    data_text = pytesseract.image_to_data(img, lang= 'eng+spa', config=custom_config, output_type= 'string')

    print(data_text)
    #print(data['text'])
    #print(len(data['text']))
    indice_words = []
    for i in range(len(data['text'])):
        if data['text'][i] != '' and data['text'][i] != ' ' and float(data['conf'][i]) >= 50 and data["text"][i] != "-" and data["text"][i] != "--":
            indice_words.append(i)


    #print(len(indice_words))
    #print(indice_words)
    print(data.keys())
    print()
    intervalo = calcular_espacios(data, indice_words)
    print(intervalo)
    coordenadas,data_organizer = coordenadas_extraccion(data,indice_words, intervalo)
    #print("numero de posiciones: " + str(len(coordenadas)))

    for i in data_organizer:
        print(i)
        pass

    print("numero de columnas", num_column(data_organizer))
    

    built_table(num_column(data_organizer), data_organizer)

    for p in coordenadas:
        cv2.rectangle(img, (p[0], p[1]),(p[2], p[3]), (50, 50, 255), 1)

    
    #extraccion_text()
    cv2.imwrite('salida.png',img)
    img = cv2.resize(img, (600, 700))
    
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()








