#include "GrafoBase.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>

namespace neuronet {

// constructor basico
GrafoDisperso::GrafoDisperso()
    : totalNodos(0), totalAristas(0), ultimoTiempoCargaMs(0.0) {}

// carga de datos con construccion csr
bool GrafoDisperso::cargarDatos(const std::string &ruta) {
    std::cout << "[C++ Core] Inicializando GrafoDisperso..." << std::endl;
    auto inicio = std::chrono::high_resolution_clock::now();

    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "[C++ Core] Error abriendo archivo: " << ruta << std::endl;
        return false;
    }

    std::vector<std::pair<int, int>> aristas;
    aristas.reserve(100000);

    int maxId = -1;
    std::string linea;
    size_t lineasLeidas = 0;
    size_t lineasValidas = 0;
    const size_t avisoCada = 1'000'000;

    while (std::getline(archivo, linea)) {
        ++lineasLeidas;
        if (linea.empty() || linea[0] == '#') {
            continue;
        }
        std::istringstream parser(linea);
        int origen = 0;
        int destino = 0;
        if (!(parser >> origen >> destino)) {
            continue;
        }
        if (origen < 0 || destino < 0) {
            continue;
        }
        aristas.emplace_back(origen, destino);
        ++lineasValidas;
        if (lineasValidas % avisoCada == 0) {
            std::cout << "[C++ Core] Progreso de lectura | lineas validas: " << lineasValidas
                      << " | aristas acumuladas: " << aristas.size() << std::endl;
        }
        maxId = std::max(maxId, std::max(origen, destino));
    }

    if (maxId < 0) {
        std::cerr << "[C++ Core] Archivo vacio o sin datos validos" << std::endl;
        return false;
    }

    std::cout << "[C++ Core] Lectura completada | lineas insumo: " << lineasLeidas
              << " | lineas validas: " << lineasValidas << std::endl;

    totalNodos = static_cast<size_t>(maxId) + 1;
    totalAristas = aristas.size();

    grados.assign(totalNodos, 0);
    for (const auto &edge : aristas) {
        grados[edge.first]++;
    }
    std::cout << "[C++ Core] Paso 1/3 completado: grados calculados" << std::endl;

    rowPtr.assign(totalNodos + 1, 0);
    for (size_t i = 0; i < totalNodos; ++i) {
        rowPtr[i + 1] = rowPtr[i] + static_cast<size_t>(grados[i]);
    }
    std::cout << "[C++ Core] Paso 2/3 completado: rowPtr listo" << std::endl;

    colIndices.assign(totalAristas, 0);
    std::vector<size_t> offset(totalNodos, 0);
    for (const auto &edge : aristas) {
        size_t posicion = rowPtr[edge.first] + offset[edge.first];
        colIndices[posicion] = edge.second;
        offset[edge.first]++;
    }
    std::cout << "[C++ Core] Paso 3/3 completado: columnas cargadas" << std::endl;

    auto fin = std::chrono::high_resolution_clock::now();
    ultimoTiempoCargaMs = std::chrono::duration<double, std::milli>(fin - inicio).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[C++ Core] Carga completa. Nodos: " << totalNodos << " | Aristas: " << totalAristas
              << std::endl;
    std::cout << "[C++ Core] Estructura CSR construida. Memoria estimada: " << estimarMemoriaMB() << " MB"
              << std::endl;
    std::cout << "[C++ Core] Tiempo de carga: " << ultimoTiempoCargaMs << " ms" << std::endl;

    return true;
}

// total de nodos cargados
size_t GrafoDisperso::obtenerTotalNodos() const { return totalNodos; }

// total de aristas cargadas
size_t GrafoDisperso::obtenerTotalAristas() const { return totalAristas; }

// nodo con mayor grado de salida
int GrafoDisperso::obtenerNodoMayorGrado() const {
    if (grados.empty()) {
        return -1;
    }

    int mejorNodo = 0;
    int mejorGrado = grados[0];

    for (size_t i = 1; i < grados.size(); ++i) {
        if (grados[i] > mejorGrado) {
            mejorNodo = static_cast<int>(i);
            mejorGrado = grados[i];
        }
    }

    return mejorNodo;
}

// vecinos directos de un nodo
std::vector<int> GrafoDisperso::obtenerVecinos(int nodo) const {
    std::vector<int> vecinos;
    if (nodo < 0 || static_cast<size_t>(nodo) >= totalNodos) {
        return vecinos;
    }

    size_t inicio = rowPtr[nodo];
    size_t fin = rowPtr[nodo + 1];
    vecinos.insert(vecinos.end(), colIndices.begin() + inicio, colIndices.begin() + fin);
    return vecinos;
}

// bfs con profundidad limitada
BFSResultado GrafoDisperso::bfsConDetalle(int origen, size_t profundidadMaxima) const {
    BFSResultado resultado;

    if (totalNodos == 0 || origen < 0 || static_cast<size_t>(origen) >= totalNodos) {
        return resultado;
    }

    std::vector<char> visitado(totalNodos, 0);
    std::queue<std::pair<int, size_t>> cola;

    visitado[origen] = 1;
    cola.emplace(origen, 0);
    resultado.nodos.push_back(origen);

    std::cout << "[C++ Core] BFS nativo | origen: " << origen << " | profundidad solicitada: "
              << profundidadMaxima << std::endl;

    size_t nivelReportado = static_cast<size_t>(-1);
    size_t nivelMaxExplorado = 0;

    while (!cola.empty()) {
        const auto [nodo, nivel] = cola.front();
        cola.pop();

        if (nivel != nivelReportado) {
            nivelReportado = nivel;
            std::cout << "[C++ Core] Explorando nivel " << nivel << "..." << std::endl;
        }

        nivelMaxExplorado = std::max(nivelMaxExplorado, nivel);

        if (nivel >= profundidadMaxima) {
            continue;
        }

        size_t inicio = rowPtr[nodo];
        size_t fin = rowPtr[nodo + 1];
        for (size_t i = inicio; i < fin; ++i) {
            int vecino = colIndices[i];
            resultado.aristas_origen.push_back(nodo);
            resultado.aristas_destino.push_back(vecino);

            if (!visitado[vecino]) {
                visitado[vecino] = 1;
                resultado.nodos.push_back(vecino);
                cola.emplace(vecino, nivel + 1);
            }
        }
    }

    std::cout << "[C++ Core] BFS finalizado | niveles explorados: " << (nivelMaxExplorado + 1)
              << " | nodos visitados: " << resultado.nodos.size()
              << " | aristas en subgrafo: " << resultado.aristas_origen.size() << std::endl;
    return resultado;
}

// memoria total del csr en mb
double GrafoDisperso::estimarMemoriaMB() const {
    double bytes = static_cast<double>(rowPtr.size() * sizeof(size_t));
    bytes += static_cast<double>(colIndices.size() * sizeof(int));
    bytes += static_cast<double>(grados.size() * sizeof(int));
    return bytes / (1024.0 * 1024.0);
}

// tiempo de carga del ultimo dataset
double GrafoDisperso::obtenerUltimoTiempoCargaMs() const { return ultimoTiempoCargaMs; }

// factorias para integracion
GrafoBase *crearGrafoDisperso() { return new GrafoDisperso(); }

void liberarGrafo(GrafoBase *grafo) { delete grafo; }

} // namespace neuronet
