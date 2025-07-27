import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
import requests
import io

def construye_adyacencia(D,m):
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

# Construye matriz de adyacencia simetrica.
def construye_adyacencia_simetrica(D, m):
    A = construye_adyacencia(D, m)
    A_sym = np.ceil((A + A.T) / 2).astype(int)
    np.fill_diagonal(A_sym, 0)
    return A_sym

def calculaLU(matriz):
    #Calcula la factorizacion LU de una matriz
    #matriz: Matriz que factorizar
    #Retorna las matrices L, U cuyo producto es 'matriz'
    dim = len(matriz)

    l = np.identity(dim)

    u = np.array([row[:] for row in matriz])

    for i in range(dim):

        #Vamos a buscar anular la columna i por debajo de la fila i
        for j in range(i + 1, dim):

            #Si nos encontramos que un elemento de la diagonal es nulo no podemos continuar
            #ya que en este metodo no hacemos permutacion de filas
            if u[i][i] == 0:
                raise ZeroDivisionError("No se puede calcular LU porque la diagonal es nula")

            #Calculamos el factor por el que hay que multiplicar la fila i para que al restarsela a j se anule el elemento ji
            factor = u[j][i] / u[i][i]

            l[j][i] = factor

            #Restamos a la fila j el producto del factor por la fila i
            for k in range(dim):
                u[j][k] -= factor * u[i][k]

    return l, u

def resuelveLU(L, U, b):
    # Ly = b. Aplicamos sustitucion hacia adelante.
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = b[i] - L[i, :i] @ y[:i]

    # Ux = y. Aplicamos sustitucion hacia atras.
    x = np.zeros_like(b)
    for i in reversed(range(len(b))):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x

def invertir(m):
    #Invierte una matriz
    #m: Matriz que invertir
    #Retorna la inversa de la matriz 'm'
    l,u = calculaLU(m)

    mi = np.zeros((m.shape[0], m.shape[0]))

    for i in range(0, m.shape[0]):
        e = np.zeros(m.shape[0])
        e[i] = 1
        mi[:, i] = resuelveLU(l,u,e)

    return mi

def calcula_matriz_C(A):
    #Calcula la matriz de transiciones de la formula de PageRank
    #A: Matriz de adyacencias de los museos
    #Retorna la matriz C

    k = np.identity(len(A))

    #Calculamos k como una matriz diagonal donde cada fila tiene la inversa de la suma por fila de A
    for i in range(len(A)):
        k[i][i] = 1/np.sum(A[i])

    a_t = np.transpose(A)

    return a_t @ k

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # alfa: coeficiente de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = len(A)
    M = (N/alfa)*(np.identity(N) - (1 - alfa) * C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y alfa
    b = np.ones(N)
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # D: Matriz de distancias
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)

    k = np.identity(len(D))

    #Calculamos k como una matriz diagonal donde cada fila tiene la inversa de la suma por fila de F
    for i in range(len(D)):
        k[i][i] = 1/np.sum(F[i])

    return F.T @ k

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz C de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matriz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    C = C.copy()
    for i in range(cantidad_de_visitas-1):
        B = B + C
        C = C @ C
    return B

def calcular_ecuacion_5(b, w):
    #Calcula el numero de visitantes inicial en cada museo a partir de la matriz B de la ecuacion (5) y la cantidad total de visitantes que recibio cada
    #museo tras la ejecucion de todo los pasos de recorrido.
    #b: Matriz B
    #w: Vector W tal que Wi contiene el total de visitantes que recibio el museo i tras todos los pasos de recorrido
    #Retorna el vector V tal que Vi contiene la cantidad de visitantes que recibio el museo i en el principio del recorrido

    l, u = calculaLU(b)

    #Resolvemos ecuacion 5 usando la factorizacion LU para no tener que invertir B
    Uv = scipy.linalg.solve_triangular(l, w, lower=True)
    v = scipy.linalg.solve_triangular(u, Uv)

    return v

#Construye la red de museos donde cada nodo tiene su peso establecido en base al PageRank del museo que representa
#a: Matriz de adyacencia de los museos
#museos: Arreglos con los datos de todos los museos relevantes
#barrios: Arreglo con los datos geograficos de los barrios
#page_rank: Arreglo donde el elemento en posicion i es el PageRank del museo i
#network_size: Tamaño del grafico expresado como un arreglo de dos elementos, donde el primer elemento es el ancho y el segundo la altura
#scale: Factor de escala de los nodos
#title: Titulo del grafico
def graficar_red(a, museos, barrios, page_rank, network_size, scale, title, ax=None):
    # Construimos la red a partir de la matriz de adyacencia
    G = nx.from_numpy_array(a)
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i: v for i, v in enumerate(zip(
        museos.to_crs("EPSG:22184").get_coordinates()['x'],
        museos.to_crs("EPSG:22184").get_coordinates()['y']
    ))}

    if ax is None:
        fig, ax = plt.subplots(figsize=(network_size[0], network_size[1]))

    ax.set_title(title)
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    nx.draw_networkx(G, G_layout, ax=ax, node_size=(page_rank * scale))


def graficar_comunidades(A, comunidades, museos, barrios, title, network_size=(10,10)):
    G = nx.from_numpy_array(A)
    layout = {i: v for i, v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],
                                             museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    colores = plt.cm.tab20(np.linspace(0, 1, len(comunidades)))
    fig, ax = plt.subplots(figsize=network_size)
    ax.set_title(title)
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    for i, comunidad in enumerate(comunidades):
        nx.draw_networkx_nodes(G, layout, nodelist=[museos.index.get_loc(c) if isinstance(c, str) else c for c in comunidad],
                               node_color=[colores[i]], node_size=50, ax=ax)
    nx.draw_networkx_edges(G, layout, alpha=0.2, ax=ax)
    plt.axis('off')
    plt.show()

def construir_matriz_de_distancias(museos):
    #Contruye la matriz de distancia para los museos en el arreglo pasado
    #museos: Arreglo con los datos de los museos
    #Retorna una matriz cuadrada donde el elemento ij es la distancia del museo i al j

    d = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
    np.fill_diagonal(d,1)
    return d

#La norma 1 es la maxima suma por columna de los componentes en valor absoluto
def norma1(m):
    return np.max(np.sum(np.abs(m), axis=0))

def calcular_cond(m):
    return norma1(m)*norma1(invertir(m))

def acotar_error(b, w, factor_de_error):
    #Calcula el error porcentual cometido al calcular el vector 'V' de visitantes iniciales cuando se sabe que el vector 'w'
    #tiene un error de a lo sumo 5%
    #b: Matriz B de la ecuacion (5)
    #w: Vector W tal que Wi contiene el total de visitantes que recibio el museo i tras todos los pasos de recorrido
    #factor_de_error: (Cota porcentual del error + 100) / 100
    #Retorna: (Cota porcentual del error en V + 100) / 100

    w = np.array(w)
    cond = calcular_cond(b)
    norma_w = norma1(w)

    #Para calcular la cota superior se toma el máximo entre dos posibles errores relativos, sea error por exceso o por deficiencia
    cota_superior = cond * max(norma1(w - w * factor_de_error), norma1(w - w * (1 - (factor_de_error - 1)))) / norma_w

    return cota_superior

def construir_grafico(x_values, y_values, x_label, y_label, title, graph_size, labels):
    #Construye un grafico de N lineas
    #x_values: Arreglo de valores que se mostraran en el eje x del grafico
    #y_values: Arreglo de N arreglos con los valores de y que toma cada linea del grafico para cada x
    #x_label: Etiqueta que mostrar para el eje x
    #y_label: Etiqueta que mostrar para el eje y
    #title: Titulo del grafico
    #graph_size: Arreglo de 1x2 con el tamaño del grafico
    #labels: Arreglo de N etiquetas que ponerle a cada linea

    plt.figure(figsize=(graph_size[0], graph_size[1]))

    #Dibujamos las lineas y las asociamos con su correspondiente etiqueta
    for i, y in enumerate(y_values):
        plt.plot(x_values, y, marker='o', label=labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()

def construir_grafico_page_rank(x_values, page_ranks, x_label, title, graph_size, nombres_museos, top_museos):
    #Construye un grafico de lineas con los PageRanks de los museos especificados en 'top_museos'
    #x_values: Valores que mostrar en el eje x del grafico
    #page_ranks: Arreglo con los PageRanks para cada valor de x para todos los museos
    #title: Titulo del grafico
    #graph_size: Arreglo de 1x2 con el tamaño del grafico
    #nombres_museos: Arreglo con los nombres de todos los museos
    #top_museos: Arreglo con los indices de los museos que graficar

    #Construimos un arreglo con los PageRanks para cada x de los museos en 'top_museos'
    page_ranks = [[page_ranks[i][museo] for i in range(len(page_ranks))] for museo in top_museos]

    #Construimos un arreglo con los nombres de los museos en 'top_museos'
    nombres_top_museos = [nombres_museos[i] for i in top_museos]

    construir_grafico(
        x_values=x_values,
        y_values=page_ranks,
        x_label=x_label,
        y_label="Pagerank",
        title=title,
        graph_size=graph_size,
        labels=nombres_top_museos)

def construir_tabla(title, graph_size, label_columna_valor, data):
    #Construye una tabla para mostrar los tres museos mas relevantes para visualizar la evolucion del podio a medida que se modifica una variable 
    #title: Titulo de la tabla
    #graph_size: Arreglo de 1x2 con el tamaño del grafico
    #label_columna_valor: Label que colocarle a la columna que muestra el valor de la variable
    #data: Mapa cuya key es el valor de la variable modificada y en el valor contiene el podio para ese valor de la variable

    df = pd.DataFrame.from_dict(data, orient="index")

    # Redondeamos la columna 'valor' a 2 decimales (el índice)
    df.index = df.index.map(lambda x: f"{x:.2f}")

    df.index.name = label_columna_valor
    df.columns = ["Top 1", "Top 2", "Top 3"]

    fig, ax = plt.subplots(figsize=(graph_size[0], graph_size[1]))
    ax.axis('tight')
    ax.axis('off')

    col_labels = [label_columna_valor] + df.columns.to_list()
    table_data = [col_labels] + df.reset_index().values.tolist()

    table = ax.table(cellText=table_data, loc='center', cellLoc='left', colLabels=None)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(title)
    plt.show()

def construir_grafico_barra(v):
    #Construye el grafico de barras de la distribucion inicial de los visitantes
    #v: array de los visitantes
    N = len(v)

    ind = np.arange(N)   
    width = 0.38

    fig = plt.subplots(figsize =(20, 15))
    p1 = plt.bar(ind, v, width,)
  
    plt.xlabel('Museos')
    plt.ylabel('Cantidad de visitantes')
    plt.title('Distribucion incial de los visitantes')
    plt.xticks(ind, rotation = 90)
    plt.yticks(np.arange(0, 1600, 100))
    plt.show()

def construir_grafico_rankings(x_values, y_values,pr_values, x_label, y_label, title, graph_size, nombres_museos, top_museos):
    #Construye un grafico de lineas con los museos ordenado por ranking
    #x_values: Arreglo de valores que se mostraran en el eje x del grafico
    #y_values: Arreglo de N arreglos con los valores de y que toma cada linea del grafico para cada x
    #pr_values:Arreglo con los valores de pagerank
    #x_label: Etiqueta que mostrar para el eje x
    #y_label: Etiqueta que mostrar para el eje y
    #title: Titulo del grafico
    #graph_size: Arreglo de 1x2 con el tamaño del grafico
    #nombres_museos: Arreglo con los nombres de todos los museos
    #top_museos: Arreglo con los indices de los museos que graficar

    page_ranks = [[pr_values[i][museo] for i in range(len(pr_values))] for museo in top_museos]
    y_values = [[y_values[i][museo] for i in range(len(y_values))] for museo in top_museos] 

    labels = [nombres_museos[i] for i in top_museos]

    plt.figure(figsize=(graph_size[0], graph_size[1]))
   
    x_axis= np.arange(0, len(x_values), 1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, y in enumerate(y_values):   
        plt.plot(x_axis, y, marker='o', label=labels[i], linewidth=2)        
        museos_label = []
        for k, j in zip(x_axis, y):
                if(j <= 3 ):
                    plt.text(k - 0.14  , j - 0.1 , labels[i], ha='left',  fontsize=6, bbox=dict(boxstyle='round', facecolor=colors[i]))                     
                    museos_label.append(labels[i])  

    plt.ylim(0.5, 3.5)
    plt.gca().invert_yaxis()
    plt.yticks([1, 2, 3])
    plt.xticks(x_axis, x_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.show()