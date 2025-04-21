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

def calculaLU(matriz):
    dim = len(matriz)

    l = np.identity(dim)

    u = np.array([row[:] for row in matriz])

    for i in range(dim):
        for j in range(i + 1, dim):

            if u[i][i] == 0:
                raise ZeroDivisionError("No se puede calcular LU porque la diagonal es nula")

            factor = u[j][i] / u[i][i]

            l[j][i] = factor

            for k in range(dim):
                u[j][k] -= factor * u[i][k]

    return l, u

def invertir_triangular_superior(u):
    u_t = np.transpose(u)
    u_inv = np.transpose(invertir_triangular_inferior(u_t))
    return u_inv

def invertir_triangular_inferior(l):
    n = len(l)
    inv = np.identity(n)
    for i in range(n):
        for j in range(i + 1):
            sum = 0
            for k in range(j, i):
                sum += l[i][k] * inv[k][j]

            #No asumimos que la l sea unitaria asi que dividimos por el elemento de la diagonal para normalizar
            inv[i][j] = (1 if i == j else 0 - sum) / l[i][i]

    return inv

def invertir_lu(L, U):
    dim = len(L)

    l_inv = invertir_triangular_inferior(L)
    u_inv = invertir_triangular_superior(U)

    return u_inv @ l_inv

def invertir(m):
    l,u = calculaLU(m)
    return invertir_lu(l, u)

def calcula_matriz_C(A):

    k = np.identity(len(A))

    for i in range(len(A)):
        k[i][i] = np.sum(A[i])

    k_inv = invertir(k)

    a_t = np.transpose(A)

    return a_t @ k_inv

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # alfa: coeficientes de damping
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

    for i in range(len(D)):
        k[i][i] = np.sum(F[i])

    k_inv = invertir(k)
    return k_inv @ F.T

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    C = C.copy()
    for i in range(cantidad_de_visitas-1):
        B = B + C
        C = C @ C
    return B

def calcular_ecuacion_5(b, w):
    l, u = calculaLU(b)
    Uv = scipy.linalg.solve_triangular(l, w, lower=True)
    v = scipy.linalg.solve_triangular(u, Uv)

    return v

def graficar_red(a, museos, barrios, page_rank, network_size, scale, title):
    G = nx.from_numpy_array(a)  # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i: v for i, v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'], museos.to_crs("EPSG:22184").get_coordinates()['y']))}

    fig, ax = plt.subplots(figsize=(network_size[0], network_size[1]))  # Visualización de la red en el mapa
    ax.set_title(title)
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)  # Graficamos Los barrios
    nx.draw_networkx(G, G_layout, ax=ax, node_size=(page_rank * scale))  # Graficamos los museos

def construir_matriz_de_distancias(museos):
    d = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
    np.fill_diagonal(d,1)
    return d

def calcular_cond(m, tipo):
    return np.linalg.norm(m, ord=tipo)*np.linalg.norm(invertir(m), ord=tipo)

def acotar_error(b, v, w, factor_de_error):
    w = np.array(w)
    cond = numpy.linalg.cond(b, ord=1)
    norma_w = np.linalg.norm(w, ord=1)
    cota_superior = cond * max(np.linalg.norm(w - w * factor_de_error, ord=1), np.linalg.norm(w - w * (1 - (factor_de_error - 1)), ord=1)) / norma_w

    return cota_superior * np.linalg.norm(v, ord=1)

def construir_grafico(x_values, y_values, x_label, y_label, title, graph_size, labels):
    plt.figure(figsize=(graph_size[0], graph_size[1]))

    for i, y in enumerate(y_values):
        plt.plot(x_values, y, marker='o', label=labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()

def construir_grafico_page_rank(x_values, page_ranks, x_label, title, graph_size, nombres_museos, top_museos):
    page_ranks = [[page_ranks[i][museo] for i in range(len(page_ranks))] for museo in top_museos]
    nombres_top_museos = [nombres_museos[i] for i in top_museos]

    construir_grafico(
        x_values=x_values,
        y_values=page_ranks,
        x_label=x_label,
        y_label="Pagerank",
        title=title,
        graph_size=graph_size,
        labels=nombres_top_museos)

def construir_tabla(title, graph_size, data):
    df = pd.DataFrame.from_dict(data, orient="index")

    # Redondeamos la columna (indice) 'Valor de m' a 2 decimales
    df.index = df.index.map(lambda x: f"{x:.2f}")

    df.index.name = "Valor de m"
    df.columns = ["Top 1", "Top 2", "Top 3"]

    fig, ax = plt.subplots(figsize=(graph_size[0], graph_size[1]))
    ax.axis('tight')
    ax.axis('off')

    col_labels = ["Valor de m"] + df.columns.to_list()
    table_data = [col_labels] + df.reset_index().values.tolist()

    table = ax.table(cellText=table_data, loc='center', cellLoc='left', colLabels=None)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(title)
    plt.show()