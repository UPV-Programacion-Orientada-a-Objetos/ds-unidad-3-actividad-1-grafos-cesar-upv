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

    FILE* archivo = fopen(ruta.c_str(), "r");
    if (!archivo) {
        std::cerr << "[C++ Core] Error abriendo archivo: " << ruta << std::endl;
        return false;
    }

    std::vector<std::pair<int, int>> aristas;
    // Reserva optimista para evitar reallocs frecuentes (ej. 5 millones de aristas)
    aristas.reserve(5000000);

    int maxId = -1;
    char buffer[1024];
    size_t lineasLeidas = 0;
    size_t lineasValidas = 0;
    const size_t avisoCada = 1000000;

    int origen, destino;
    while (fgets(buffer, sizeof(buffer), archivo)) {
        ++lineasLeidas;
        if (buffer[0] == '#' || buffer[0] == '\n') {
            continue;
        }
        
        // Parsing manual basico es mas rapido que stringstream
        if (sscanf(buffer, "%d %d", &origen, &destino) != 2) {
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
        if (origen > maxId) maxId = origen;
        if (destino > maxId) maxId = destino;
    }
    fclose(archivo);

    if (maxId < 0) {
        std::cerr << "[C++ Core] Archivo vacio o sin datos validos" << std::endl;
        return false;
    }

    std::cout << "[C++ Core] Lectura completada | lineas insumo: " << lineasLeidas
              << " | lineas validas: " << lineasValidas << std::endl;

    totalNodos = static_cast<size_t>(maxId) + 1;
    totalAristas = aristas.size();

    // Optimizacion: Usar vector::assign y acceso directo para conteo de grados
    grados.assign(totalNodos, 0);
    for (const auto &edge : aristas) {
        grados[edge.first]++;
    }
    std::cout << "[C++ Core] Paso 1/3 completado: grados calculados" << std::endl;

    rowPtr.assign(totalNodos + 1, 0);
    // Calculo de prefijos acumulados
    for (size_t i = 0; i < totalNodos; ++i) {
        rowPtr[i + 1] = rowPtr[i] + static_cast<size_t>(grados[i]);
    }
    std::cout << "[C++ Core] Paso 2/3 completado: rowPtr listo" << std::endl;

    colIndices.assign(totalAristas, 0);
    std::vector<size_t> offset(totalNodos, 0);
    
    // Llenado de CSR
    for (const auto &edge : aristas) {
        int u = edge.first;
        int v = edge.second;
        size_t posicion = rowPtr[u] + offset[u];
        colIndices[posicion] = v;
        offset[u]++;
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

// dfs con profundidad limitada (iterativo para evitar stack overflow)
BFSResultado GrafoDisperso::dfsConDetalle(int origen, size_t profundidadMaxima) const {
    BFSResultado resultado;

    if (totalNodos == 0 || origen < 0 || static_cast<size_t>(origen) >= totalNodos) {
        return resultado;
    }

    std::vector<char> visitado(totalNodos, 0);
    // Stack almacena: {nodo, nivel}
    std::vector<std::pair<int, size_t>> stack;
    stack.reserve(1000);

    stack.emplace_back(origen, 0);
    
    // DFS iterativo no marca visitado al empujar, sino al procesar, 
    // pero para evitar ciclos y duplicados en stack, podemos marcar al empujar 
    // o usar un set. Para eficiencia en grafos densos, mejor marcar al visitar 
    // pero cuidando no re-empujar demasiado.
    // Estrategia comun: marcar al sacar del stack (visitado real) 
    // O marcar al meter (para evitar meterlo multiples veces).
    // En DFS estricto se marca al entrar.
    
    // Vamos a usar la estrategia de marcar al procesar para permitir caminos alternativos 
    // si fuera necesario, pero aqui solo queremos el arbol de expansion.
    // Asi que marcaremos al sacar para asegurar el orden DFS correcto.
    
    // Revision: Para evitar que el stack crezca exponencialmente en grafos densos,
    // es mejor marcar al ver. Pero eso es mas parecido a BFS.
    // DFS puro: marcar al visitar.
    
    std::cout << "[C++ Core] DFS nativo | origen: " << origen << " | profundidad solicitada: "
              << profundidadMaxima << std::endl;

    size_t nivelMaxExplorado = 0;

    while (!stack.empty()) {
        auto [nodo, nivel] = stack.back();
        stack.pop_back();

        if (visitado[nodo]) {
            continue;
        }
        visitado[nodo] = 1;
        resultado.nodos.push_back(nodo);
        
        nivelMaxExplorado = std::max(nivelMaxExplorado, nivel);

        if (nivel >= profundidadMaxima) {
            continue;
        }

        size_t inicio = rowPtr[nodo];
        size_t fin = rowPtr[nodo + 1];
        
        // Para simular orden recursivo (izquierda a derecha), empujamos en orden inverso
        // O empujamos normal y procesamos derecha a izquierda.
        // El orden de vecinos en CSR es arbitrario (segun input).
        // Empujamos normal:
        for (size_t i = inicio; i < fin; ++i) {
            int vecino = colIndices[i];
            if (!visitado[vecino]) {
                // Solo agregamos arista si el vecino no ha sido visitado aun
                // (Arista de arbol). Si quisieramos back-edges, seria diferente.
                // Nota: En DFS iterativo, la arista se confirma cuando visitamos al hijo.
                // Pero aqui necesitamos guardar la arista para visualizar.
                // Si lo agregamos aqui, podriamos tener aristas a nodos que luego no visitamos
                // si ya fueron visitados por otro camino en el stack.
                // Correccion: Guardar padre en stack para reconstruir arista al visitar.
            }
        }
        
        // Re-implementacion con padre en stack para capturar aristas correctamente
    }
    
    // Limpiamos y reiniciamos con logica corregida
    resultado.nodos.clear();
    std::fill(visitado.begin(), visitado.end(), 0);
    stack.clear();
    
    // Stack: {nodo, nivel, padre}
    // Usamos un vector de tuplas o 3 vectores paralelos. O struct simple.
    struct Frame {
        int nodo;
        size_t nivel;
        int padre;
    };
    std::vector<Frame> pila;
    pila.reserve(1000);
    
    pila.push_back({origen, 0, -1});
    
    while(!pila.empty()){
        Frame f = pila.back();
        pila.pop_back();
        
        if(visitado[f.nodo]) continue;
        
        visitado[f.nodo] = 1;
        resultado.nodos.push_back(f.nodo);
        if(f.padre != -1){
            resultado.aristas_origen.push_back(f.padre);
            resultado.aristas_destino.push_back(f.nodo);
        }
        
        nivelMaxExplorado = std::max(nivelMaxExplorado, f.nivel);
        
        if(f.nivel >= profundidadMaxima) continue;
        
        size_t inicio = rowPtr[f.nodo];
        size_t fin = rowPtr[f.nodo + 1];
        
        // Empujamos en reverso para visitar en orden de indice (opcional, estetico)
        // CSR tiene vecinos contiguos.
        if (fin > inicio) {
            for (size_t i = fin; i > inicio; --i) {
                int vecino = colIndices[i - 1];
                if (!visitado[vecino]) {
                    pila.push_back({vecino, f.nivel + 1, f.nodo});
                }
            }
        }
    }

    std::cout << "[C++ Core] DFS finalizado | niveles explorados: " << (nivelMaxExplorado + 1)
              << " | nodos visitados: " << resultado.nodos.size()
              << " | aristas en subgrafo: " << resultado.aristas_origen.size() << std::endl;
    return resultado;
}

} // namespace neuronet
