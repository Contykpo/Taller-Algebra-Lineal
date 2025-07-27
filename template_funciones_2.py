import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
import requests
import io

import template_funciones as tp1

from scipy.linalg import eigh

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])


def calcular_matriz_de_grado(A):
    #Para calcular la matriz de grado, simplemente ponemos los elementos del vector de grado en una matriz diagonal
    return np.diag(calcular_vector_de_grado(A))

def calcular_vector_de_grado(A):
    #Calculamos el vector de grado contando para cada vector, cuantas aristas incidentes tiene
    return np.sum(A, axis=1)

#Calcula la matriz laplaciana para el grafo descripto por la matriz de adyacencia A.
def calcula_L(A):
    # Calcula la matriz Laplaciana L de la red, dada la matriz de adyacencia A.
    # La matriz L se define como: L = K - A
    # donde K es la matriz diagonal de grados (K[i,i] = grado del nodo i).

    K = calcular_matriz_de_grado(A)

    return K - A

# La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
def calcula_R(A):
    # Calcula la matriz de modularidad R de la red, a partir de la matriz de adyacencia A.
    # La matriz R se define como: R = A - P
    # Donde P = (k * k^T) / (2 * E),
    # k es el vector de grados y
    # E es el número total de aristas E = sum(A) / 2

    #Para calcular todas las aristas sumamos todos los elementos de la matriz de adyacencia y dividimos por dos
    #pues cada arista aparece contada dos veces ya que es un grafo no dirigido
    edge_count = np.sum(A)/2

    K = calcular_vector_de_grado(A)

    #Calculamos P = (k * k^T) / (2 * E)
    P = np.outer(K, K) / (2*edge_count)

    return A - P

def calcula_lambda(L, v):
    # Calcula el corte asociado al vector v, usando la matriz Laplaciana L.
    # La fórmula es: λ = (s^T) * L * s
    # donde s es el vector de signos de v (s_i = sign(v_i)).
    # Obtenemos el vector s de signos: +1 si v_i > 0, -1 si v_i < 0
    s = np.sign(v)
    # Calculamos el corte: s^T * L * s
    lambdon = s.T @ L @ s
    return lambdon

def calcula_Q(R, v):
    # Calcula la modularidad asociada al vector v, usando la matriz de modularidad R.
    # La fórmula es: Q = (s^T) * R * s
    # donde s es el vector de signos de v (s_i = sign(v_i)).
    # Esto representa la diferencia entre el número de aristas internas al grupo y el esperado por azar.
    s = np.sign(v)
    Q = s.T @ R @ s
    return Q

# Metodo de la potencia que encuentra el autovalor de mayor modulo y su correspondiente autovector.
# A: matriz cuadrada de entrada.
# tol: tolerancia para el error relativo entre autovalores estimados consecutivos.
# maxrep: maximo numero de iteraciones.
def metpot1(A, tol=1e-8, maxrep=np.inf):
    # Generamos un vector inicial aleatorio y evitamos autovectores nulos.
    v = np.random.uniform(-1, 1, size=A.shape[0])
    # Normalizacion inicial.
    v = v / np.linalg.norm(v)
    # Aplicamos A: A * v.
    v1 = A @ v
    # Normalizamos para evitar desbordes numericos.
    v1 = v1 / np.linalg.norm(v1)
    # Estimacion inicial del autovalor con el cociente de Rayleigh.
    l = v.T @ A @ v
    # Estimacion siguiente del autovalor.
    l1 = v1.T @ A @ v1
    nrep = 0

    # Iteramos hasta que la diferencia relativa entre autovalores sea pequeña:
    while np.abs(l1 - l) / np.abs(l) > tol and nrep < maxrep:
        v = v1
        l = l1
        # Multiplicacion por A.
        v1 = A @ v
        v1 = v1 / np.linalg.norm(v1)
        # Cociente de Rayleigh para mejorar el autovalor estimado.
        l1 = v1.T @ A @ v1
        nrep += 1

    if not nrep < maxrep:
        print('MaxRep alcanzado')

    # Estimacion final del autovalor.
    l = v1.T @ A @ v1
    # Retorna el autovector, autovalor, y la convergencia.
    return v1, l, nrep < maxrep

# Calcula la deflacion de A usando su primer autovector y autovalor.
# A: matriz de entrada.
def deflaciona(A, tol=1e-8, maxrep=np.inf):
    # Formula de la matriz deflacionada = A - lambdo_1 * v1 v1^t.
    v1, l1, _ = metpot1(A, tol, maxrep)
    # Calculamos la deflacion eliminando el componente de lambdo_1.
    deflA = A - l1 * np.outer(v1, v1)
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   return metpot1(deflA,tol,maxrep)

# Metodo de la potencia inversa con shift que busca el autovalor mas pequeño de A+mu*I.
# A es la matriz.
# mu es el parametro de shift.
def metpotI(A, mu, tol=1e-8, maxrep=np.inf):
    A_shifted = A + mu * np.eye(A.shape[0])
    v, l, converge = metpot1(tp1.invertir(A_shifted))
    l = 1/l - mu
    return v,l,converge

# Metodo de la potencia inversa con deflacion que encuentra el segundo autovalor mas pequeño.
# A: matriz
# mu: es el shift
def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
    # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
    # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
    A_shifted = A + mu * np.identity(A.shape[0])
    iX = tp1.invertir(A_shifted)
    defliX = deflaciona(iX)
    v,l,converge =  metpot1(defliX) # Bu scamos su segundo autovector
    l = 1/l - mu # Reobtenemos el autovalor correcto
    return v,l,converge
      
# Recursivamente realiza cortes y reduce en 1 el numero de niveles hasta llegar a 0 y retornar.
# A: matriz.
# niveles: cantidad de niveles sobre los que hacer cortes.
# nombres_s: los nombres de los nodos.
# Retorna una lista con conjuntos de nodos representando las comunidades.
def laplaciano_iterativo(A, niveles, nombres_s=None):
    # Si no hay nombres asignados, los numeramos
    if nombres_s is None:
        nombres_s = list(range(A.shape[0]))
    # Si solo queda un nodo o no quedan niveles, devolvemos la comunidad actual.
    if A.shape[0] == 1 or niveles == 0:
        return [nombres_s]
    else:
        # Calculamos el laplaciano.
        L = calcula_L(A)
        
        # Segundo vector propio, Fiedler vector, el primero es el vector constante.
        v,l,_ = metpotI2(L, 1)
        
        # Particionamos los nodos segun el signo de las componentes del vector propio.
        idx_pos = [i for i, val in enumerate(v) if val > 0]
        idx_neg = [i for i, val in enumerate(v) if val < 0]
        
        # Creamos subgrafos.
        Ap = A[np.ix_(idx_pos, idx_pos)]
        Am = A[np.ix_(idx_neg, idx_neg)]
        
        # Nombres asociados a cada particion.
        nombres_p = [nombres_s[i] for i in idx_pos]
        nombres_m = [nombres_s[i] for i in idx_neg]
        
        # Llamada recursiva sobre cada subgrafo.
        return (laplaciano_iterativo(Ap, niveles-1, nombres_p) + laplaciano_iterativo(Am, niveles-1, nombres_m))

# A: matriz.
# R: matriz modularidad.
# nombres_s: los nombres de los nodos.
# Retorna una lista con conjuntos de nodos representando las comunidades.
def modularidad_iterativo(A=None,R=None,nombres_s=None):
    if A is None and R is None:
        print('Dame una matriz gordo')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])

    # Si llegamos al último nivel
    if R.shape[0] == 1:
        return [nombres_s]

    # Primer autovector y autovalor de R
    v,l,_ = metpot1(R)

    # Modularidad Actual:
    Q0 = np.sum(R[np.outer(v > 0, v > 0)]) + np.sum(R[np.outer(v < 0, v < 0)])

    # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
    if Q0<=0 or all(v>0) or all(v<0):
        return [nombres_s]

    mask_pos = v > 0
    mask_neg = v < 0

    ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad

    # Parte de R asociada a los valores positivos de v
    Rp = R[np.ix_(mask_pos, mask_pos)]

    # Parte asociada a los valores negativos de v
    Rm = R[np.ix_(mask_neg, mask_neg)]

    # autovector principal de Rp
    vp,lp,_ = metpot1(Rp, 1)

    # autovector principal de Rm
    vm,lm,_ = metpot1(Rm, 1)

    # Calculamos el cambio en Q que se produciría al hacer esta partición
    Q1 = 0
    if not (all(vp>0) or all(vp<0)):
        Q1 = np.sum(Rp[np.outer(vp > 0, vp > 0)]) + np.sum(Rp[np.outer(vp < 0, vp < 0)])

    if not (all(vm>0) or all(vm<0)):
        Q1 += np.sum(Rm[np.outer(vm > 0, vm > 0)]) + np.sum(Rm[np.outer(vm < 0, vm < 0)])

    # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
    if Q0 >= Q1: 
        return [np.array(nombres_s)[mask_pos], np.array(nombres_s)[mask_neg]]

    return (modularidad_iterativo(A, Rp, np.array(nombres_s)[mask_pos]) + modularidad_iterativo(A, Rm, np.array(nombres_s)[mask_neg]))

def graficar_comunidad(layout, G, comunidad, row, column, axs, m, barrios, titulo):
    ax = axs[row, column]
    colores = plt.cm.tab20(np.linspace(0, 1, len(comunidad)))
    ax.set_title(f'm={m} {titulo}')
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    for i, comunidad in enumerate(comunidad):
        nx.draw_networkx_nodes(G, layout, nodelist=comunidad, node_color=[colores[i]], node_size=50, ax=ax)
    nx.draw_networkx_edges(G, layout, alpha=0.2, ax=ax)
    ax.axis('off')