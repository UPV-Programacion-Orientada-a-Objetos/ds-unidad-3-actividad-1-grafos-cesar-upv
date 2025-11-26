#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace neuronet {

// estructura para resultados de bfs
struct BFSResultado {
    std::vector<int> nodos;
    std::vector<int> aristas_origen;
    std::vector<int> aristas_destino;
};

// interfaz abstracta para grafos dispersos
class GrafoBase {
public:
    virtual ~GrafoBase() = default;
    virtual bool cargarDatos(const std::string &ruta) = 0;
    virtual size_t obtenerTotalNodos() const = 0;
    virtual size_t obtenerTotalAristas() const = 0;
    virtual int obtenerNodoMayorGrado() const = 0;
    virtual std::vector<int> obtenerVecinos(int nodo) const = 0;
    virtual BFSResultado bfsConDetalle(int origen, size_t profundidadMaxima) const = 0;
    virtual double estimarMemoriaMB() const = 0;
    virtual double obtenerUltimoTiempoCargaMs() const = 0;
};

// implementacion concreta basada en csr
class GrafoDisperso : public GrafoBase {
public:
    GrafoDisperso();
    bool cargarDatos(const std::string &ruta) override;
    size_t obtenerTotalNodos() const override;
    size_t obtenerTotalAristas() const override;
    int obtenerNodoMayorGrado() const override;
    std::vector<int> obtenerVecinos(int nodo) const override;
    BFSResultado bfsConDetalle(int origen, size_t profundidadMaxima) const override;
    double estimarMemoriaMB() const override;
    double obtenerUltimoTiempoCargaMs() const override;

private:
    size_t totalNodos;
    size_t totalAristas;
    double ultimoTiempoCargaMs;
    std::vector<size_t> rowPtr;
    std::vector<int> colIndices;
    std::vector<int> grados;
};

// factorias de objetos para cython
GrafoBase *crearGrafoDisperso();
void liberarGrafo(GrafoBase *grafo);

} // namespace neuronet
