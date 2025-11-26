# cython: language_level=3

# importaciones desde la libreria estandar
from libcpp.string cimport string
from libcpp.vector cimport vector

# importacion de definiciones de c++
cimport grafocore

# clase que expone el nucleo en python
cdef class NeuroNetCore:
    cdef grafocore.GrafoBase *impl

    def __cinit__(self):
        # crear instancia concreta del grafo disperso
        self.impl = grafocore.crearGrafoDisperso()
        if self.impl == NULL:
            raise MemoryError("No se pudo inicializar el grafo")

    def __dealloc__(self):
        # liberar recursos nativos cuando python destruye el objeto
        if self.impl != NULL:
            grafocore.liberarGrafo(self.impl)
            self.impl = NULL

    def cargar_archivo(self, path: str):
        # convierte la ruta de python a std::string
        cdef bytes encoded = path.encode('utf-8')
        cdef string ruta = encoded
        cdef bint ok = self.impl.cargarDatos(ruta)
        if not ok:
            raise RuntimeError("No se pudo cargar el dataset")
        return True

    def total_nodos(self) -> int:
        return <int>self.impl.obtenerTotalNodos()

    def total_aristas(self) -> int:
        return <int>self.impl.obtenerTotalAristas()

    def nodo_mayor_grado(self) -> int:
        return self.impl.obtenerNodoMayorGrado()

    def vecinos(self, nodo: int):
        cdef vector[int] datos = self.impl.obtenerVecinos(nodo)
        return [datos[i] for i in range(datos.size())]

    def memoria_mb(self) -> float:
        return self.impl.estimarMemoriaMB()

    def tiempo_carga_ms(self) -> float:
        return self.impl.obtenerUltimoTiempoCargaMs()

    def bfs(self, origen: int, profundidad: int):
        cdef grafocore.BFSResultado resultado = self.impl.bfsConDetalle(origen, <size_t>max(0, profundidad))
        cdef int nodos = resultado.nodos.size()
        cdef int aristas = resultado.aristas_origen.size()
        nodos_py = [resultado.nodos[i] for i in range(nodos)]
        aristas_py = [(resultado.aristas_origen[i], resultado.aristas_destino[i]) for i in range(aristas)]
        return {"nodos": nodos_py, "aristas": aristas_py}
