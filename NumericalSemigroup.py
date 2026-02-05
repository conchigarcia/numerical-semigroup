import math
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
from sympy import symbols, groebner

# Funciones auxiliares:
def SetAdd(L1, L2, limit=None):
    """
    Devuelve la lista de todas las sumas posibles a + b con a en L1 y b en L2.
    Si se especifica 'limit', solo se guardan las sumas menores o iguales a ese límite.
    """
    L3 = []
    for a in L1:
        for b in L2:
            s = a + b
            if limit is None or s <= limit:
                L3.append(s)
    return sorted(set(L3)) # Eliminamos duplicados y ordenamos

def minimize_relations(relations_list):
    """
    Elimina elementos redundantes de una lista de polinomios que generan un ideal.
    """
    minimal_gens = list(relations_list)
    
    i = 0
    while i < len(minimal_gens):
        g = minimal_gens[i]
        
        # Creamos la lista de generadores con todos menos el actual
        others = minimal_gens[:i] + minimal_gens[i+1:]
        
        # Si hay otros generadores, comprobamos si g pertenece al ideal que generan
        if others:
            G = groebner(others)
            _, remainder = G.reduce(g)
            
            # Si el resto es cero, g es redundante
            if remainder == 0:
                minimal_gens.pop(i)
                continue
        
        # Si g es esencial, pasamos al siguiente
        i += 1
            
    return minimal_gens

def share_generator(f1, f2):
    """
    Devuelve True si dos factorizaciones usan algún generador común.
    """
    return any(a > 0 and b > 0 for a, b in zip(f1, f2))

def distance(x, y):
    """
    Devuelve la distancia entre x e y: d(x,y) = max(|x|,|y|) - |x ∧ y|
    """
    common = [min(xi, yi) for xi, yi in zip(x, y)]
    return max(sum(x), sum(y)) - sum(common)

class NumericalSemigroup:
    def __init__(self, *args):
        """
        Construye un semigrupo numérico a partir de un conjunto de generadores
        como argumentos: lista o secuencia de enteros positivos cuyo máximo común divisor es 1.
        """

        # Aceptamos tanto listas como varios enteros
        if len(args) == 1 and isinstance(args[0], list):
            gens = args[0]
        else:
            gens = list(args)

        # Validaciones
        L = set(gens)
        L.discard(0)

        if not L:
            raise ValueError("Debe haber al menos un generador.")

        if not all(isinstance(x, int) and x > 0 for x in L):
            raise ValueError("Todos los generadores deben ser enteros positivos.")

        if math.gcd(*L) != 1:
            raise ValueError("El máximo común divisor debe ser 1.")

        # Guardamos los generadores ordenados
        self.generators = tuple(sorted(L))
        self.multiplicity_val = min(self.generators)

        # Inicializamos atributos
        self.minimal_generators_val = None
        self.frobenius_number_val = None
        self.conductor_val = None
        self.gaps_val = None
        self.small_elements_val = None
        self.apery_set_val = {}

        # Casos especiales
        if self.multiplicity_val == 1:
            self.minimal_generators_val = (1)
            self.gaps_val = ()
            self.small_elements_val = (0,)
            self.frobenius_number_val = -1
            self.conductor_val = 0
            self.apery_set_val = {self.multiplicity_val: (0)}
        
        elif len(self.generators) == 2:
            a, b = self.generators
            self.minimal_generators_val = (a, b)
            self.frobenius_number_val = a * b - a - b
            self.conductor_val = self.frobenius_number_val + 1
    
    def __contains__(self, n):
        """
        Permite usar la sintaxis 'n in S'.
        """
        return self.belongs(n)
    
    def multiplicity(self):
        return self.multiplicity_val

    def small_elements(self):
        """
        Devuelve todos los elementos del semigrupo hasta el número de Frobenius + 1.
        """
        # Comprobamos si ya está calculado
        if self.small_elements_val is not None:
            return self.small_elements_val
        
        # Caso 1: tenemos los gaps
        if self.gaps_val is not None:
            if not self.gaps_val: # Caso S = N
                F = -1
                small_elements = [0]
            else:
                F = max(self.gaps_val)
                gaps_set = set(self.gaps_val)
                # Los small_elements son los que NO son gaps hasta el conductor
                small_elements = [x for x in range(F + 2) if x not in gaps_set]
            
            # Actualizamos atributos faltantes y convertimos a tupla
            self.frobenius_number_val = F
            self.conductor_val = F + 1
            self.small_elements_val = tuple(small_elements)

            return tuple(small_elements)
        
        # Caso 2: tenemos calculado un conjunto de Apéry
        if self.apery_set_val:
            n, apery = next(iter(self.apery_set_val.items()))
            F = max(apery) - n
            small_elements = [x for x in range(F + 2) if x >= apery[x % n]]
            
            # Actualizamos atributos
            self.frobenius_number_val = F
            self.conductor_val = F + 1
            self.small_elements_val = tuple(small_elements)

            return tuple(small_elements)
        
        # Caso 3: algoritmo general (Moskowitz)
        # Tomamos un sistema de generadores más pequeño si es posible
        if self.minimal_generators_val is not None:
            reduced_gens = list(self.minimal_generators_val) # Si hemos calculado el mínimo, lo usamos
        else:
            # Eliminamos generadores redundates (múltiplos de otros)
            gens = list(self.generators)
            reduced_gens = []
            for g in gens:
                is_candidate = True
                for h in gens:
                    if g == h:
                        continue
                    if g % h == 0:
                        is_candidate = False
                        break
                if is_candidate:
                    reduced_gens.append(int(g))

        # Construimos el conjunto de elementos sumando combinaciones de generadores
        upTo = min(reduced_gens) * max(reduced_gens)  # Cota superior

        current_layer = reduced_gens.copy()  # Elementos actuales en expansión
        all_elements = set([0] + current_layer)  # Todos los elementos encontrados hasta el momento

        is_expanding = True # Control del bucle: sigue iterando mientras se sigan encontrando nuevos elementos

        while is_expanding:
            new_elements = SetAdd(current_layer, reduced_gens, limit=upTo) # Generamos nuevas sumas hasta 'upTo'
            new_elements = [x for x in new_elements if x not in all_elements] # Nos quedamos solo con los nuevos

            # Si no hay nuevos elementos, detenemos la expansión
            if not new_elements:
                is_expanding = False
            else:
                # Añadimos los nuevos al conjunto total y seguimos
                all_elements.update(new_elements)
                current_layer = new_elements

        # Convertimos a lista ordenada
        all_elements = sorted(all_elements)

        # Calculamos gaps y número de Frobenius
        max_elem = max(all_elements)
        gaps = [n for n in range(max_elem) if n not in all_elements] # Saltos
        F = -1 if len(gaps) == 0 else max(gaps) # Número de Frobenius
        small_elements = [n for n in all_elements if n <= F + 1]

        # Actualizamos atributos
        self.gaps_val = tuple(gaps)
        self.frobenius_number_val = F
        self.conductor_val = F + 1
        self.small_elements_val = tuple(small_elements)

        return tuple(small_elements)

    def gaps(self):
        """
        Devuelve los gaps (saltos) del semigrupo.
        """
        if self.gaps_val is None:
            se = self.small_elements()
            # Calculamos gaps si es que no los tenemos después de obtener small_elements
            if self.gaps_val is None:
                F = self.frobenius_number()
                gaps = sorted(set(range(F + 1)) - set(se))
                self.gaps_val = tuple(gaps)

        return self.gaps_val

    def genus(self):
        """
        Calcula el género (número de saltos) del semigrupo numérico.
        """
        # Caso 1: tenemos los gaps calculados
        if self.gaps_val is not None:
            return len(self.gaps_val)

        # Caso 2: general (usando la Fórmula de Selmer)        
        # Calculamos el Apéry set respecto a la multiplicidad si no lo tenemos
        m = self.multiplicity_val
        apery = self.apery(m)

        # Sumamos los elementos del Apéry set
        sum_apery = sum(apery)
        
        # Aplicamos la fórmula: g(S) = (1/m) * Sum(Ap(S,m)) - (m-1)/2        
        g = (sum_apery / m) - ((m - 1) / 2)
        return int(g)

    def frobenius_number(self):
        """
        Devuelve el número de Frobenius.
        """
        # Si ya está calculado, lo devolvemos directamente
        if self.frobenius_number_val is not None:
            return self.frobenius_number_val

        # Caso especial: S tiene dos generadores, aplicamos la fórmula directa
        if len(self.generators) == 2:
            a, b = self.generators
            self.frobenius_number_val = a * b - a - b
            return self.frobenius_number_val

        # Caso 1, tenemos los gaps: devolvemos el máximo
        if self.gaps_val is not None:
            self.frobenius_number_val = max(self.gaps_val)
            return self.frobenius_number_val

        # Caso 2, tenemos los small_elements: devolvemos el máximo - 1
        if self.small_elements_val is not None:
            self.frobenius_number_val = max(self.small_elements_val) - 1 # small_elements contiene elementos hasta F+1 inclusive.
            return self.frobenius_number_val

        # Caso 3, tenemos algún Apéry set calculado:
        # usamos la fórmula de Selmer: F(S) = max(Ap(S, n)) - n
        if self.apery_set_val:
            n, apery = next(iter(self.apery_set_val.items()))
            self.frobenius_number_val = max(apery) - n
            return self.frobenius_number_val

        # Caso 4: método general (calculamos Apéry respecto a la multiplicidad y usamos Selmer)
        m = self.multiplicity_val
        apery = self.apery(m)
        self.frobenius_number_val = max(apery) - m
        return self.frobenius_number_val

    def conductor(self):
        """
        Devuelve el conductor del semigrupo (F + 1).
        """
        if self.conductor_val is None:
            self.conductor_val = self.frobenius_number() + 1
        return self.conductor_val

    def minimal_generators(self):
        """
        Devuelve el sistema mínimo de generadores del semigrupo numérico.
        """
        if self.minimal_generators_val is not None:
            return self.minimal_generators_val

        gens = list(self.generators)
        m = self.multiplicity_val

        # Caso especial
        if m == 2:
            odd_gen = next(g for g in gens if g % 2 == 1) # Primer generador impar
            self.minimal_generators_val = (2, odd_gen)
            self.generators = (2, odd_gen)  # Actualizamos los generadores al mínimo
            return (2, odd_gen)
        
        # Caso 1: tenemos los small_elements o algún Apéry set calculado
        if self.small_elements_val is not None or self.apery_set_val:
            min_gens = gens[:]
            for g in gens:
                # Si g se puede escribir como suma de otros elementos, se elimina
                if any((g - h) in self for h in gens if h < g): 
                    min_gens.remove(g)
            
            min_gens = tuple(sorted(min_gens))
            self.minimal_generators_val = min_gens
            self.generators = min_gens  # Actualizamos los generadores al mínimo
            return min_gens

        # Caso 2: algoritmo general (GAP)
        # Si hay muchos generadores comparado con la multiplicidad, filtramos por restos.
        if m < len(gens):
            residue_candidates = {}
            for g in gens:
                r = g % m
                if r not in residue_candidates:
                    residue_candidates[r] = g
            
            # Reconstruimos la lista de candidatos
            min_gens = sorted(residue_candidates.values())
        else:
            min_gens = gens

        # Cota superior: no necesitamos mirar sumas mayores al mayor generador
        upTo = max(min_gens)
        # Generamos sumas hasta 'upTo'
        current_layer = set(SetAdd(min_gens, min_gens, limit=upTo))

        while current_layer:
            min_gens = [x for x in min_gens if x not in current_layer]
            current_layer = set(SetAdd(current_layer, min_gens, limit=upTo))

        # Actualizamos atributos y devolvemos
        min_gens = tuple(sorted(min_gens))
        self.minimal_generators_val = min_gens
        self.generators = min_gens  # Actualizamos los generadores al mínimo
        return min_gens
    
    def embedding_dimension(self):
        return len(self.minimal_generators())

    def pseudo_frobenius_numbers(self):
        """
        Devuelve los números pseudo-Frobenius de S.
        """
        # Caso 1: tenemos algún Apéry set calculado
        if self.apery_set_val:
            n, apery = next(iter(self.apery_set_val.items()))

            PF = []
            for w in apery:
                # w es maximal en apery respecto al orden inducido por s
                # si no existe v ≠ w con v - w ∈ S
                if not any((v - w) in self for v in apery if v > w):
                    PF.append(w - n)

            return tuple(sorted(PF))

        # Caso 2: método general. Aplicamos la definición
        else:
            gaps = list(self.gaps())

            # Caso trivial: S = N
            if not gaps:
                return (-1,)

            min_gens = self.minimal_generators()
            F = max(gaps)
            se = {x for x in range(F + 1) if x not in gaps}
            PF = []

            # g es pseudo-Frobenius si g + m ∈ S para todo generador minimal m
            for g in gaps:
                if all((g + m) in se or (g + m) > F for m in min_gens):
                    PF.append(g)

            return tuple(PF)

    def type(self):
        return len(self.pseudo_frobenius_numbers())

    def belongs(self, n):
        """
        Determina si el entero 'n' pertenece al semigrupo numérico.
        """
        if n < 0: return False
        if n == 0: return True
        if self.conductor_val is not None and n >= self.conductor_val:
            return True

        # Caso 1: tenemos small_elements
        if self.small_elements_val is not None:
            if n >= self.conductor():
                return True
            return n in self.small_elements_val

        # Caso 2: tenemos algún Apéry set calculado    
        if self.apery_set_val:
            m, apery = next(iter(self.apery_set_val.items()))
            # n in S <==> n >= w (donde w es el mínimo residuo de n mod m)
            return n >= apery[n % m]
        
        # Caso 3: algoritmo de programación dinámica (general) (GAP)
        # Si n es múltiplo de algún generador, pertenece a S
        gens = self.generators
        for g in gens:
            if n % g == 0:
                return True
            
        # Creamos una lista de falsos del tamaño del número n
        # dp[i] será True si el número i se puede formar
        dp = [False] * (n + 1)
        dp[0] = True  # El 0 siempre se puede formar

        # Primera pasada con la multiplicidad
        # Marcamos todos los múltiplos del primer generador como True
        m = self.multiplicity_val
        for i in range(m, n + 1, m):
            dp[i] = True

        # Pasadas con el resto de generadores
        for g in gens[1:]:
            # Recorremos la lista desde el generador hasta n
            for i in range(g, n + 1):
                # Si dp[i] ya es True, no tocamos nada.
                # Si es False, miramos "g" pasos atrás.
                if not dp[i]: 
                    if dp[i - g]:  # Si se puede formar i - g, entonces se puede formar i
                        dp[i] = True
        return dp[n]

    def apery(self, m=None):
        """
        Calcula el conjunto de Apéry del semigrupo numérico S respecto 
        a un elemento m (que pertenezca o no a S). Si no se indica m, se usa la 
        multiplicidad del semigrupo.
        """
        # Determinar valor de m
        if m is None:
            m = self.multiplicity()

        # Si ya tenemos este conjunto guardado, lo devolvemos directamente    
        if m in self.apery_set_val:
            return self.apery_set_val[m]

        # Caso especial: S tiene dos generadores y m es uno de ellos
        if len(self.generators) == 2 and m in self.generators:
            other_gen = self.generators[1] if m == self.generators[0] else self.generators[0]
            
            # Generamos y ordenamos por residuo
            apery = [0] * m
            for i in range(m):
                val = i * other_gen
                apery[val % m] = val
            
            # Guardamos en el caché y devolvemos
            if not self.apery_set_val or m < min(self.apery_set_val.keys()):
                self.apery_set_val = {m: tuple(apery)}
            return tuple(apery)

        # Caso 1: m in S.
        if m in self:
            # Caso 1.1: tenemos small_elements o algún Apéry set calculado
            if self.small_elements_val is not None or self.apery_set_val:
                apery = [0] * m
                for r in range(m): # Recorremos cada residuo
                    k = 0
                    while True: # Generamos candidatos crecientes: r + k *m con k >= 0
                        x = r + k*m
                        if x in self:
                            apery[r] = x # Guardamos y salimos del bucle porque es el mínimo
                            break
                        k += 1

                # Guardamos en el caché y devolvemos
                if not self.apery_set_val or m < min(self.apery_set_val.keys()):
                    self.apery_set_val = {m: tuple(apery)}
                return tuple(apery)

            # Caso 1.2: no tenemos small_elements (Chris O'Neill)
            else:
                # Preparamos generadores
                gens = list(self.generators)
                non_m = [g for g in gens if g != m] # Quitamos el generador m
                apery = [float("inf")] * m
                apery[0] = 0  # el resto 0 siempre tiene representante 0
                
                # Inicializar con los generadores distintos de m
                for g in non_m:
                    r = g % m
                    apery[r] = min(apery[r], g)

                # Propagar sumas pequeñas
                current = non_m[:] # lista de elementos activos, es decir, aquellos
                                # que han actualizado algún residuo recientemente
                while current:
                    new_values = [] # nuevos candidatos generados en esta iteración
                    for a in current:
                        if apery[a % m] != a: # Solo seguimos si 'a' sigue siendo el mínimo de su clase módulo m.
                            continue          # Si ya fue reemplazado por un valor menor, no lo usamos.
                        for g in non_m:
                            s = a + g # nuevo elemento posible del semigrupo
                            r = s % m # residuo correspondiente
                            if s < apery[r]: # Si encontramos un valor más pequeño para ese resto,
                                apery[r] = s # actualizamos el mínimo y marcamos 's' para seguir propagando.
                                new_values.append(s)
                    current = new_values
                
                # Guardamos en el caché y devolvemos
                if not self.apery_set_val or m < min(self.apery_set_val.keys()):
                    self.apery_set_val = {m: tuple(apery)}
                return tuple(apery)

        # Caso 2: m not in S. Paquete GAP
        else:
            se = list(self.small_elements())
            C = self.frobenius_number() + 1 # Conductor
            
            # Reúne candidatos: small_elements + {C+1, ..., C+m-1}
            candidatos = se + list(range(C + 1, C + m))

            # Filtra: un candidato x pertenece a Apery(m) si x in S x - m not in en S
            apery = [x for x in candidatos if (x - m) not in self]

            return tuple(apery)
        
    def factorizations(self, n):
        """
        Devuelve una tupla de tuplas con todas las formas de escribir n
        usando los generadores mínimos.
        """
        if n not in self:
            raise ValueError(f"El número {n} no está en el semigrupo.")
        
        return tuple(self._factorize(self.minimal_generators(), n))
    
    def _factorize(self, gens, target):
        """
        Método interno recursivo para calcular factorizaciones.
        """
        # Caso base: Solo queda un generador
        if len(gens) == 1:
            gen = gens[0]
            if target % gen == 0:
                return [(target // gen,)] 
            return []

        # Paso recursivo
        sols = []
        last_gen = gens[-1]
        rest_gens = gens[:-1] 
        
        # Probamos cuántas veces cabe el último generador
        max_times = target // last_gen
        
        for i in range(max_times + 1):
            remainder = target - (i * last_gen)
            
            # Recursión: resuelve el resto con los generadores que quedan
            partial_sols = self._factorize(rest_gens, remainder)
            
            # Construimos las soluciones completas
            for partial in partial_sols:
                sols.append(partial + (i,))
                
        return sols

    def get_factorization_graph(self, n):
            """
            Construye el grafo de factorizaciones de n.
            """
            facts = self.factorizations(n)

            # Creamos el grafo con NetworkX
            G = nx.Graph()
            
            # Añadimos nodos
            G.add_nodes_from(facts)
            
            # Añadimos aristas
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    f1 = facts[i]
                    f2 = facts[j]
                    if share_generator(f1, f2):
                        G.add_edge(f1, f2)
            return G   

    def R_classes(self, n):
        """
        Devuelve las componentes conexas del grafo de factorizaciones de n.
        """
        # Obtenemos el grafo y sus componentes conexas
        G = self.get_factorization_graph(n)
        raw_components = list(nx.connected_components(G))
        
        # Convertimos a tuplas y ordenamos
        cleaned_components = [tuple(sorted(comp)) for comp in raw_components]
        cleaned_components.sort()
        
        return tuple(cleaned_components)

    def betti_elements(self):
        """
        Calcula los Elementos de Betti del semigrupo.
        Un número x es Elemento de Betti si su grafo de factorizaciones
        tiene más de una componente conexa.
        """
        m = self.multiplicity_val
        apery = self.apery(m)
        min_gens = self.minimal_generators()

        # Generamos candidatos: w + g para cada w ≠ 0 en Apéry y cada generador minimal g != 0
        gens_to_check = [g for g in min_gens if g != m]
        candidates = {w + g for w in apery if w != 0 for g in gens_to_check}

        betti = []

        # Filtramos los candidatos según la conectividad de su grado de factorizaciones
        for x in sorted(candidates):
            # Si tiene más de una R-clase, es un elemento de Betti
            if len(self.R_classes(x)) > 1:
                betti.append(x)

        return tuple(sorted(betti))

    def plot_factorization_graph(self, n, engine="plotly"):
        """
        Dibuja el grafo de factorizaciones de n.
        
        Opciones de engine:
        'plotly': interactivo
        'pyvis': físico
        """
        # Obtenemos el grafo sus componentes conexas
        G = self.get_factorization_graph(n)
        components = list(nx.connected_components(G)) 
        
        # Posicionamiento de nodos
        pos = nx.spring_layout(G, k=0.15, seed=42)
        
        # Colores para las componentes
        colores = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
        
        if engine == "plotly":
            fig = go.Figure()

            # Dibujamos aristas
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'), # Gris
                hoverinfo='none',                 # Las líneas no muestran texto
                showlegend=False               
            ))

            # Dibujamos nodos por componentes conexas            
            for i, comp_nodes in enumerate(components):
                node_x = []
                node_y = []
                node_text = []
                
                for node in comp_nodes:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(str(node)) # Texto que saldrá al pasar el ratón

                # Añadimos la capa de puntos de esta componente
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',       
                    text=node_text,            
                    textposition="top center", 
                    hoverinfo='text',
                    cliponaxis=False,
                    marker=dict(size=14, color=colores[i % len(colores)], line=dict(width=2, color='DarkSlateGrey')),
                    name=f"R{i}",
                    textfont=dict(size=12, color='black') 
                ))

            # C. Configuración final visual
            fig.update_layout(
                title=dict(
                    text=f"Grafo de factorizaciones de {n} ({len(components)} R-clases)",
                    x=0.5,              # Posición horizontal (0.5 es el centro)
                    xanchor='center'    # Anclaje del texto al centro
                ),
                plot_bgcolor='white',
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), # Quitar ejes
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            fig.show()

        elif engine == "pyvis":

            # Configuración oscura para resaltar
            net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', notebook=False)
            
            # Iteramos componentes y asignamos atributos a los nodos de NetworkX
            for i, comp in enumerate(components):
                color_actual = colores[i % len(colores)]
                for node in comp:
                    # Asignamos color visual
                    G.nodes[node]['color'] = color_actual

            # Convertimos nodos a texto
            G_str = nx.relabel_nodes(G, lambda x: str(x))
            
            # Cargar y mostrar
            net.from_nx(G_str)
            net.toggle_physics(True)
            nombre_archivo = f"grafo_{n}.html"
            net.show(nombre_archivo, notebook=False)
            print(f"Abriendo {nombre_archivo}...")

        else:
            raise ValueError("El engine debe ser 'plotly' o 'pyvis'.")

    def minimal_presentation(self):
        """
        Calcula la presentación minimal del semigrupo usando eliminación 
        de variables y bases de Gröbner.
        """
        # Caso 1: tenemos los small_elements o algún Apéry set calculado
        if self.small_elements_val is not None or self.apery_set_val:
            presentation = []
            betti_elems = self.betti_elements()
            for b in betti_elems:
            # Obtenemos las componentes conexas (R-clases)
                components = self.R_classes(b)
            
                # Conectamos las k componentes con k-1 relaciones
                for i in range(len(components) - 1):                 
                    # Representante de la componente i (tomamos el primero)
                    r1 = components[i][0]
                    
                    # Representante de la componente i+1 (tomamos el primero)
                    r2 = components[i+1][0]
                    
                    # Añadimos la relación (r1, r2)
                    presentation.append((min(r1, r2), max(r1, r2)))
                        
            return tuple(presentation)
        
        # Caso 2: algoritmo general (Eliminación de variables)
        else:
            gens = list(self.minimal_generators())
            p = len(gens)
            t = symbols('t')

            # Creamos las variables xi
            x_vars = symbols(f'x0:{p}')

            # Construimos el ideal <{xi - t^ni}>
            equations = [x_vars[i] - t**gens[i] for i in range(p)]
            
            # Calculamos Base de Gröbner (Eliminación)
            # order='lex' con 't' primero fuerza la eliminación de t
            gb = groebner(equations, t, *x_vars, order='lex')
            
            # Filtramos resultados (intersección)
            # Nos quedamos solo con los polinomios que no contienen t
            dirty_rels = [p for p in gb if t not in p.free_symbols]
            
            # Minimizamos (limpieza de redundancias)
            clean_rels = minimize_relations(dirty_rels)

            # Convertimos los polinomios en pares de tuplas de exponentes
            presentation = []
            for poly in clean_rels:
                # Extraemos los monomios como tuplas de exponentes
                p = poly.as_poly(*x_vars)
                exponents = p.monoms() 
                
                # Esperamos binomios: X^a - X^b
                if len(exponents) == 2:
                    presentation.append(tuple(exponents))
                    
            return tuple(presentation)
    
    def lengths(self, n):
        """
        Calcula el conjunto de longitudes L(n) de las factorizaciones de n.
        Devuelve una tupla ordenada de enteros.
        """
        # Obtenemos todas las factorizaciones
        facts = self.factorizations(n)
        
        # Calculamos la suma de componentes para cada factorización
        L = sorted(set(sum(f) for f in facts))
        
        return tuple(L)

    def elasticity(self, n=None):
        """
        Calcula la elasticidad.
        - Si se da n: calcula rho(n).
        - Si n es None: calcula rho(S).
        """
        # Caso 1: elasticidad del semigrupo
        if n is None:
            min_gens = self.minimal_generators()
            return min_gens[-1] / min_gens[0]

        # Caso 2: elasticidad de un elemento
        else:
            L = self.lengths(n)
            return max(L) / min(L)

    def delta_set(self, n):
        """
        Calcula el conjunto Delta de n.
        """
        L = self.lengths(n)
        if len(L) <= 1:
            return ()
        
        # Diferencias entre longitudes consecutivas: l_{i+1} - l_i
        delta = set(L[i+1] - L[i] for i in range(len(L)-1))
        return tuple(sorted(delta))

    def min_delta(self):
        """
        Calcula el mínimo del conjunto Delta(S).
        """
        pres = self.minimal_presentation()
        
        if not pres: # Caso S = N
            return 0 
            
        # Calculamos |a - b| para cada relación (a, b) en la presentación
        diffs = []
        for a, b in pres:
            len_a = sum(a)
            len_b = sum(b)
            diffs.append(abs(len_a - len_b))
            
        # El mínimo es el GCD de estas diferencias
        if not diffs:
            return 0
            
        return math.gcd(*diffs)

    def max_delta(self):
        """
        Calcula el máximo del conjunto Delta(S).
        """
        betti = self.betti_elements()
        max_d = 0
        
        for b in betti:
            d_b = self.delta_set(b)
            if d_b:
                max_d = max(max_d, max(d_b))
                
        return max_d

    def catenary_degree(self, n=None):
        """
        Calcula el grado de catenariedad.
         - Si se da n: calcula c(n).
         - Si n es None: calcula c(S).
        """
        # Caso 1: grado de catenariedad del semigrupo
        if n is None:
            # Se devuelve elmáximo de los grados de catenariedad de los elementos de Betti
            betti = self.betti_elements()
            return max((self.catenary_degree(b) for b in betti), default=0)
        
        # Caso 2: grado de catenariedad de un elemento
        else:
            # Obtenemos las factorizaciones del elemento n
            Z = self.factorizations(n)
            n_facts = len(Z)
            
            # Si hay 0 o 1 factorización, el grado de catenariedad es 0
            if n_facts <= 1:
                return 0
        
            # Generamos todas las aristas posibles del grafo completo con pesos según la distancia entre factorizaciones
            weighted_edges = []
            for i in range(n_facts):
                for j in range(i + 1, n_facts):
                    d = distance(Z[i], Z[j])
                    weighted_edges.append((i, j, d))

            # Construimos el grafo con pesos
            G = nx.Graph()
            G.add_weighted_edges_from(weighted_edges)

            # Calculamos el árbol generador de peso mínimo usando el algoritmo de Kruskal
            mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
            
            if not mst.edges:
                return 0
                
            # El grado catenario es el peso máximo de ese árbol de expansión mínima
            return max(data['weight'] for u, v, data in mst.edges(data=True))
    
    def plot_catenary_graph(self, n):
        """
        Dibuja el grafo completo de factorizaciones destacando el árbol generador de peso mínimo.
        """
        Z = self.factorizations(n)
        n_facts = len(Z)

        # Construimos el grafo completo con pesos según la distancia entre factorizaciones
        G = nx.Graph()
        for i in range(n_facts):
            G.add_node(i, label=str(Z[i]))
            for j in range(i + 1, n_facts):
                d = distance(Z[i], Z[j])
                G.add_edge(i, j, weight=d)

        # Calculamos el árbol generador de peso mínimo
        arbol = nx.minimum_spanning_tree(G, algorithm='kruskal')
        aristas_arbol = set(frozenset((u, v)) for u, v in arbol.edges())

        # Posicionamiento de nodos
        pos = nx.spring_layout(G, k=0.15, seed=42)
        fig = go.Figure()

        # Listas para separar visualmente: árbol (rojo) vs resto (gris)
        x_rojo, y_rojo, x_gris, y_gris = [], [], [], []
        label_x, label_y, label_text = [], [], []
        hover_texts = []

        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Guardamos coordenadas para la etiqueta del peso
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            label_text.append(str(data['weight']))
            mensaje = f"La distancia entre {Z[u]} y {Z[v]} es {data['weight']}"
            hover_texts.append(mensaje)

            # Separamos las coordenadas según si es arista del árbol generador o no
            if frozenset((u, v)) in aristas_arbol:
                x_rojo.extend([x0, x1, None])
                y_rojo.extend([y0, y1, None])
            else:
                x_gris.extend([x0, x1, None])
                y_gris.extend([y0, y1, None])
      
        # Traza 1: aristas normales (gris)
        fig.add_trace(go.Scatter(x=x_gris, y=y_gris, mode='lines',
                                 line=dict(width=1, color='lightgrey'), hoverinfo='none'))

        # Traza 2: aristas del árbol generador (rojo)
        fig.add_trace(go.Scatter(x=x_rojo, y=y_rojo, mode='lines', name='Árbol Generador',
                                 line=dict(width=2, color='#EF553B'), hoverinfo='name'))

        # Traza 3: etiquetas de peso
        fig.add_trace(go.Scatter(x=label_x, y=label_y, mode='text', text=label_text,
                                 textposition="middle center", hovertext=hover_texts, hoverinfo='text',
                                 textfont=dict(size=14, color='black'), 
                                 showlegend=False))

        # Traza 4: nodos
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [G.nodes[node]['label'] for node in G.nodes()]

        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                 text=node_text, textposition="top center", hoverinfo='text',
                                 marker=dict(size=12, color='#636EFA', line=dict(width=2, color='DarkSlateGrey')),
                                 name='Factorizaciones'))

        fig.update_layout(title=dict(text=f"Grafo de Factorizaciones de {n}", x=0.5, xanchor='center'),
                          plot_bgcolor='white', showlegend=False,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        fig.show()
        
    def tame_degree(self, n=None):
        """
        Calcula el grado de amansamiento.
         - Si se da n: calcula t(n).
         - Si n es None: calcula t(S).
        """  
        min_gens = self.minimal_generators()

        # Caso 1: grado de amansamiento del semigrupo
        if n is None:
            if 1 in min_gens: 
                return 0
            
            # Obtenemos elementos de Apéry de todos los generadores
            apery_union = set()
            for g in min_gens:
                apery_union.update(self.apery(g))
            
            # Generamos candidatos y devolvemos el máximo t(s)
            candidates = {w + g for w in apery_union for g in min_gens}
            if not candidates: 
                return 0
            
            return max((self.tame_degree(c) for c in candidates), default=0)

        # Caso 2: grado de amansamiento de un elemento
        else:
            Z = self.factorizations(n)
            if len(Z) <= 1:
                return 0
                
            max_tame = 0
            num_gens = len(min_gens)
            
            # Recorremos cada generador n_i, y calculamos t_i(n)
            for i in range(num_gens):
                # Dividimos Z: los que no usan el generador i (target) vs los que sí (rest)
                target = [z for z in Z if z[i] == 0]
                rest = [z for z in Z if z[i] != 0]
                
                # Si alguno de los subconjuntos es vacío, t_i(n) = 0
                if not target or not rest: 
                    continue
                    
                # Calculamos t_i(n) definido como el máximo de las distancias de x al conjunto rest.
                current_max = 0
                for x in target:
                    min_dist = min(distance(x, z) for z in rest)
                    # Actualizamos el máximo local encontrado para este generador
                    if min_dist > current_max:
                        current_max = min_dist
                
                # Actualizamos el grado de amansamiento total
                if current_max > max_tame:
                    max_tame = current_max
                    
            return max_tame
