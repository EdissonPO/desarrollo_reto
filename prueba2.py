import statistics as st
import cv2
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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
                if (int(diccionario["top"][indices[i]]) >= int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]])+13 or int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) > int(diccionario["top"][indices[i+1]])-13):
                    
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
            if (int(diccionario["top"][indices[i]]) >= int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]])+13 or int(diccionario["top"][indices[i]]) < int(diccionario["top"][indices[i+1]]) and int(diccionario["top"][indices[i]]) > int(diccionario["top"][indices[i+1]])-13):
                
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


def extracion_colums(data):

    coordinates = []
    for i in range(len(data)):
       coordinates.append(data[i]["coordinates"])

    arr = np.array(coordinates)
    colums = []
    
    for i in np.arange(len(arr)):
        temp = []
        temp.append(i)
        for j in np.arange(len(arr)):

            if i == j: continue
            interval1 = np.arange(arr[i,0], arr[i,1]+1, dtype=int)
            interval2 = np.arange(arr[j,0], arr[j,1]+1, dtype=int)

            sum = 0
            for k in interval1:
                if k in interval2:
                    sum += 1
    
            tamaño1 = len(interval1)
            tamaño2 = len(interval2)
            percent1 = int((100/tamaño1)*sum )
            percent2 = int((100/tamaño2)*sum )
            """
            print("tamaño1: ", tamaño1)
            print("tamaño2: ", tamaño2)
            print("la suma: ", sum)
            print("percent1: ", percent1, "%")
            print("percent2: ", percent2, "%")
            """
            if percent1 > 55 and percent2 > 55:
                temp.append(j)
        
        colums.append(temp)
        
    cl = []
    descartados = []
    for i in np.arange(len(colums)):
        temp = []
        temp = colums[i]
        if i in descartados: continue
        for j in np.arange(len(colums)):

            if i == j: continue
            interval1 = colums[i]
            interval2 = colums[j]

            sum = 0
            for k in interval1:
                if k in interval2:
                    sum += 1
    
            tamaño1 = len(interval1)
            tamaño2 = len(interval2)
            percent1 = int((100/tamaño1)*sum )
            percent2 = int((100/tamaño2)*sum )
            """
            print("tamaño1: ", tamaño1)
            print("tamaño2: ", tamaño2)
            print("la suma: ", sum)
            print("percent1: ", percent1, "%")
            print("percent2: ", percent2, "%")
            """
            if percent1 > 30 or percent2 > 30:
                temp += colums[j]
                descartados.append(j)
        cl.append(temp)

    colums = []    
    #se eliminan elementos repetidos en las columnas
    for element in cl:
        temp = []
        for i in element:
            if i not in temp:
                temp.append(i)
        colums.append(temp)
    

    print("el tamaño de cl es: ", len(cl))
    #en este bloque se saca la pocision en pixeles de la imagen
    min_max = []
    for element in colums:   
        menor = arr[element[0],0]
        mayor = arr[element[0],1]
        for i in element:
            if arr[i,0] < menor:
                menor = arr[i,0]
            
            if arr[i,1] > mayor:
                mayor = arr[i,1]
        min_max.append((menor,mayor))

    for i in range(len(min_max)):
        print(min_max[i], colums[i])

    cl = []
    descartados = []
    by_delete = []
    for i in range(len(min_max)):
        temp= []
        temp= colums[i]
        if i in descartados: continue
        for j in range(len(min_max)):

            if i==j: continue
            if min_max[i][0]-10 <= min_max[j][0] and min_max[i][1]+10 >= min_max[j][1]:
                temp += colums[j]
                descartados.append(j)
                if i > j:
                    by_delete.append(colums[j])
        cl.append(temp)

    colums = cl
    print("estos son a eliminar", by_delete)
    for k in by_delete:
        colums.remove(k)

    print()
    zises = []
    for element in colums:
        print(element)
        zises.append(len(element))
    
    mean = st.mean(zises)
    print("la media de los len ", mean)
    by_delete = []
    for i in range(len(colums)):
        percent20 = (mean/100)*20
        #print("percent20: ", percent20)
        if len(colums[i]) < percent20:
            by_delete.append(colums[i])

    for element in colums:
        print(element)
        print

    print("elementos a eliminar: ", by_delete)
    for k in by_delete:
        colums.remove(k)

    for element in colums:
        print(element)

    min_max_h= []
    for element in colums:   
        menor = arr[element[0],0]
        mayor = arr[element[0],1]
        for i in element:
            if arr[i,0] < menor:
                menor = arr[i,0]
            
            if arr[i,1] > mayor:
                mayor = arr[i,1]
        min_max_h.append((menor,mayor))

    min_max_h_unOrder = min_max_h.copy()
    min_max_h.sort()

    cl = []
    for i in np.arange(len(colums)):
        for j in np.arange(len(colums)):
            if min_max_h_unOrder[j] == min_max_h[i]:
                cl.append(colums[j])

    colums = cl
    print(min_max_h)
    
    return min_max_h,colums
    
    
def built_table(data, colums):
    

    
    """
    print(coordinates)
    print()
    extracion_colums(coordinates, data)

    coordinates_left = [i[0] for i in coordinates]
    coordinates_righ = [i[1] for i in coordinates]

    frecuency_left = {}
    frecuency_righ = {}
    for i,j in coordinates:
        if i not in frecuency_left:
            frecuency_left[i]=0
        else:
            frecuency_left[i] += 1

        if j not in frecuency_righ:
            frecuency_righ[j]=0
        else:
            frecuency_righ[j] += 1

    frecuency_left = sorted(frecuency_left.items())
    frecuency_left = {i[0]:i[1]+1 for i in frecuency_left}

    frecuency_righ = sorted(frecuency_righ.items())
    frecuency_righ = {i[0]:i[1]+1 for i in frecuency_righ}

    print("---------------------------")
    print(frecuency_left)
    print("---------------------------")
    print(frecuency_righ)

    plt.bar(range(len(frecuency_left)),frecuency_left.values(), edgecolor="black")
    plt.xticks(range(len(frecuency_left.keys())), frecuency_left.keys())
    plt.ylim(min(frecuency_left.values())-1 , max(frecuency_left.values())+1)
    plt.show()
    

    print("tamaño original: ", len(coordinates))
    coordinates.sort()
    #print(coordinates)
    #histograma(coordinates)
    #coordinates.sort()
    #print(coordinates)
    """
    

def main():
    #path = r'./reto/1.PNG'
    path = r'.\reto\4.PNG'
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

    #print(data_text)
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

    min_max_h,colums = extracion_colums(data_organizer)

    
    #Dibujado de los rectangulos en la imagen
    for p in min_max_h:
        cv2.rectangle(img, (p[0], hImg-5),(p[1], 5), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 1)

    for p in coordenadas:
        cv2.rectangle(img, (p[0], p[1]),(p[2], p[3]), (50, 50, 255), 1)
    
    #impresion del texto de las columnas
    for colum in colums:
        print()
        print()
        for i in colum:
            print(data_organizer[i]["text"])
    
    cv2.imwrite('salida.png',img)
    img = cv2.resize(img, (600, 700))
    
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()








