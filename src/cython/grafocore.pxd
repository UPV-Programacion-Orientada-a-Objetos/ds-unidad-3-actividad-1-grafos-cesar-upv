from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t

cdef extern from "GrafoBase.hpp" namespace "neuronet":
    cdef cppclass BFSResultado:
        vector[int] nodos
        vector[int] aristas_origen
        vector[int] aristas_destino

    cdef cppclass GrafoBase:
        bint cargarDatos(const string &ruta)
        size_t obtenerTotalNodos() const
        size_t obtenerTotalAristas() const
        int obtenerNodoMayorGrado() const
        vector[int] obtenerVecinos(int nodo) const
        BFSResultado bfsConDetalle(int origen, size_t profundidadMaxima) const
        double estimarMemoriaMB() const
        double obtenerUltimoTiempoCargaMs() const

    GrafoBase *crearGrafoDisperso()
    void liberarGrafo(GrafoBase *grafo)
