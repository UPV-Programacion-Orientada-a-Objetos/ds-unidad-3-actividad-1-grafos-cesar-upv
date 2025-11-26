import math
import random
import sys
import webbrowser
from collections import deque

try:
    from fa2 import ForceAtlas2
except ImportError:  # pragma: no cover
    ForceAtlas2 = None
from datetime import datetime
from pathlib import Path

import networkx as nx
from PyQt6 import QtCore, QtGui, QtWidgets

from neuronet_core import NeuroNetCore


# senales livianas para comunicar hilos
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(str, object)


# ejecutor reutilizable para tareas pesadas
class Worker(QtCore.QRunnable):
    def __init__(self, task: str, fn, *args, **kwargs):
        super().__init__()
        self.task = task
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            data = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(self.task, data)
        finally:
            self.signals.finished.emit(self.task)


# vista grafica con zoom fluido
class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__(scene)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.TextAntialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom_steps = 0

    def wheelEvent(self, event: QtGui.QWheelEvent):  # noqa: D401
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        if self._zoom_steps <= -15 and factor < 1:
            return
        if self._zoom_steps >= 60 and factor > 1:
            return
        self._zoom_steps += 1 if factor > 1 else -1
        self.scale(factor, factor)

    def reset_zoom(self):
        self._zoom_steps = 0
        self.resetTransform()


# ventana dedicada para el subgrafo interactivo
class GraphWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subgrafo NeuroNet")
        self.resize(1100, 760)
        layout = QtWidgets.QVBoxLayout(self)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color:#f5f5f7;font-size:12px")
        controls = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Ajustar vista")
        reset_btn.clicked.connect(self._fit_view)
        controls.addWidget(reset_btn)
        controls.addStretch()
        layout.addWidget(self.info_label)
        layout.addLayout(controls)
        layout.addWidget(self.view)

    # dibuja datos recibidos aplicando zoom interactivo
    def render(self, graph: nx.DiGraph, layout_positions: dict[int, tuple[float, float]], origen: int):
        self.scene.clear()
        if not layout_positions:
            return
        xs = [pos[0] for pos in layout_positions.values()]
        ys = [pos[1] for pos in layout_positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(max_x - min_x, 1e-3)
        height = max(max_y - min_y, 1e-3)
        scale = 520.0 / max(width, height)
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        positions = {}
        for node, (x, y) in layout_positions.items():
            positions[node] = QtCore.QPointF((x - center_x) * scale, (y - center_y) * scale)
        max_degree = max((deg for _, deg in graph.degree()), default=1)
        pen_edge = QtGui.QPen(QtGui.QColor("#6c63ff"))
        pen_edge.setWidthF(0.9)
        pen_edge.setCosmetic(True)
        for u, v in graph.edges():
            if u not in positions or v not in positions:
                continue
            line = self.scene.addLine(
                positions[u].x(),
                positions[u].y(),
                positions[v].x(),
                positions[v].y(),
                pen_edge,
            )
            line.setOpacity(0.55)
        for node, point in positions.items():
            grado = graph.degree(node)
            intensidad = grado / max_degree if max_degree else 0
            hue = 0.66 - 0.5 * intensidad
            base_color = QtGui.QColor.fromHsvF(max(0.0, hue), 0.55, 1.0)
            brush = QtGui.QBrush(base_color)
            pen = QtGui.QPen(QtGui.QColor("#101321"))
            pen.setWidthF(1.1)
            radius = 9 + (math.log1p(grado) * 2.5)
            ellipse = self.scene.addEllipse(
                point.x() - radius,
                point.y() - radius,
                radius * 2,
                radius * 2,
                pen,
                brush,
            )
            if node == origen:
                ellipse.setBrush(QtGui.QBrush(QtGui.QColor("#ffbd39")))
            label = self.scene.addSimpleText(str(node))
            label.setBrush(QtGui.QBrush(QtGui.QColor("#f5f5f7")))
            label.setPos(point.x() + radius + 4, point.y() + radius + 4)
        self.info_label.setText(
            f"Nodos mostrados: {graph.number_of_nodes()} | Aristas: {graph.number_of_edges()} | Origen BFS: {origen}"
        )
        self._fit_view()
        self.view.scale(0.65, 0.65)
        self.setModal(False)
        self.show()
        self.raise_()
        self.activateWindow()

    def _fit_view(self):
        self.view.reset_zoom()
        rect = self.scene.itemsBoundingRect()
        if rect.isValid():
            self.view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)


# ventana principal con estilo moderno
class NeuroNetWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroNet Explorer")
        self.resize(1320, 780)

        # estado compartido de la sesion
        self.core = NeuroNetCore()
        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(1)
        self.dataset_dir = Path(__file__).parent / "dataset"
        self.dataset_loaded = False
        self.total_nodes = 0
        self.graph_window = None
        self.max_visual_nodes = 600
        self.max_visual_edges = 1200
        self.last_visual = None

        self._init_palette()
        self._build_ui()
        self._refresh_dataset_options()
        self._set_progress("Listo", False)

    # paleta oscura y estilos base
    def _init_palette(self):
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0f111a"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#161927"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#f5f5f7"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#1f2231"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#f5f5f7"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#6c63ff"))
        self.setPalette(palette)
        self.setStyleSheet(
            "QWidget {font-family: 'Inter','Segoe UI',sans-serif; font-size: 13px;}"
            "QPushButton {background-color:#6c63ff;border:none;border-radius:8px;padding:10px 16px;color:white;}"
            "QPushButton:disabled {background-color:#3c3f56;color:#999cb5;}"
            "QGroupBox {border:1px solid #2a2d3f;border-radius:10px;margin-top:14px;padding:14px;}"
            "QLineEdit,QSpinBox,QSlider,QComboBox {background-color:#161927;border:1px solid #2a2d3f;border-radius:8px;color:#f5f5f7;padding:6px;}"
            "QTextEdit {background-color:#0f111a;border:1px solid #2a2d3f;border-radius:10px;color:#e6e8ef;}"
        )

    # armado de la interfaz general
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        self._build_top_bar(layout)
        self._build_metrics(layout)
        self._build_graph_hint(layout)
        self._build_log_panel(layout)

    # barra superior con seleccion de dataset
    def _build_top_bar(self, parent_layout):
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        label = QtWidgets.QLabel("Dataset activo")
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Selecciona archivo SNAP o usa la lista rápida")

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._combo_selected)

        browse_btn = QtWidgets.QPushButton("Buscar archivo")
        browse_btn.clicked.connect(self._browse_dataset)

        self.load_btn = QtWidgets.QPushButton("Cargar núcleo C++")
        self.load_btn.clicked.connect(self._trigger_load)

        grid.addWidget(label, 0, 0)
        grid.addWidget(self.path_edit, 0, 1)
        grid.addWidget(self.dataset_combo, 0, 2)
        grid.addWidget(browse_btn, 0, 3)
        grid.addWidget(self.load_btn, 0, 4)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 2)

        parent_layout.addWidget(container)

    # tarjetas de metricas y controles bfs
    def _build_metrics(self, parent_layout):
        metrics_group = QtWidgets.QGroupBox("Métricas del grafo")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)
        metrics_layout.setHorizontalSpacing(22)

        self.nodes_label = QtWidgets.QLabel("Nodos: -")
        self.edges_label = QtWidgets.QLabel("Aristas: -")
        self.memory_label = QtWidgets.QLabel("Memoria: - MB")
        self.time_label = QtWidgets.QLabel("Carga: - ms")
        self.degree_label = QtWidgets.QLabel("Nodo crítico: -")

        for idx, widget in enumerate(
            [self.nodes_label, self.edges_label, self.memory_label, self.time_label, self.degree_label]
        ):
            metrics_layout.addWidget(widget, 0, idx)

        parent_layout.addWidget(metrics_group)

        bfs_group = QtWidgets.QGroupBox("Simulación BFS")
        bfs_layout = QtWidgets.QGridLayout(bfs_group)
        bfs_layout.setHorizontalSpacing(12)
        bfs_layout.setVerticalSpacing(10)

        self.origin_spin = QtWidgets.QSpinBox()
        self.origin_spin.setRange(0, 0)
        self.origin_spin.setPrefix("Origen ")

        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(1, 10)
        self.depth_spin.setPrefix("Prof ")

        self.depth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(2)
        self.depth_slider.valueChanged.connect(self.depth_spin.setValue)
        self.depth_spin.valueChanged.connect(self.depth_slider.setValue)

        self.random_btn = QtWidgets.QPushButton("Parámetros aleatorios")
        self.random_btn.clicked.connect(self._randomize_params)

        self.bfs_btn = QtWidgets.QPushButton("Generar subgrafo")
        self.bfs_btn.clicked.connect(self._trigger_bfs)

        self.pyvis_btn = QtWidgets.QPushButton("Exportar a PyVis")
        self.pyvis_btn.clicked.connect(self._export_pyvis)
        self.pyvis_btn.setDisabled(True)

        self.progress_label = QtWidgets.QLabel("Listo")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet(
            "QProgressBar {background-color:#1b1f2d;border:1px solid #2a2d3f;border-radius:10px;color:#f5f5f7;}"
            "QProgressBar::chunk {background-color:#6c63ff;border-radius:9px;}"
        )
        self.progress_bar.setFormat("Listo")
        self.progress_bar.hide()

        bfs_layout.addWidget(QtWidgets.QLabel("Nodo origen"), 0, 0)
        bfs_layout.addWidget(self.origin_spin, 0, 1)
        bfs_layout.addWidget(self.random_btn, 0, 2)
        bfs_layout.addWidget(QtWidgets.QLabel("Profundidad"), 1, 0)
        bfs_layout.addWidget(self.depth_spin, 1, 1)
        bfs_layout.addWidget(self.depth_slider, 1, 2)
        bfs_layout.addWidget(self.bfs_btn, 0, 3, 2, 1)
        bfs_layout.addWidget(self.progress_label, 2, 0, 1, 2)
        bfs_layout.addWidget(self.progress_bar, 2, 2, 1, 2)
        bfs_layout.addWidget(self.pyvis_btn, 3, 0, 1, 4)

        parent_layout.addWidget(bfs_group)

    # panel informativo del subgrafo
    def _build_graph_hint(self, parent_layout):
        group = QtWidgets.QGroupBox("Explorador visual")
        vbox = QtWidgets.QVBoxLayout(group)
        hint = QtWidgets.QLabel(
            "Los subgrafos se abren en una ventana dedicada con zoom, paneo y ajuste automático"
        )
        hint.setWordWrap(True)
        vbox.addWidget(hint)
        parent_layout.addWidget(group)

    # panel de bitacora
    def _build_log_panel(self, parent_layout):
        log_group = QtWidgets.QGroupBox("Bitácora del sistema")
        vbox = QtWidgets.QVBoxLayout(log_group)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(170)
        vbox.addWidget(self.log_view)
        parent_layout.addWidget(log_group)

    # sincroniza combo de datasets
    def _refresh_dataset_options(self):
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem("Datasets locales", "")
        if self.dataset_dir.exists():
            for file in sorted(self.dataset_dir.glob("*.txt")):
                self.dataset_combo.addItem(file.stem, str(file))
        self.dataset_combo.blockSignals(False)

    # dialogo de seleccion manual
    def _browse_dataset(self):
        dialog = QtWidgets.QFileDialog(self, "Selecciona dataset SNAP", str(self.dataset_dir))
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Textos (*.txt)")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                self.path_edit.setText(selected[0])
                self._log(f"dataset seleccionado: {selected[0]}")

    # cuando cambia elemento rapido
    def _combo_selected(self, index: int):
        path = self.dataset_combo.itemData(index)
        if path:
            self.path_edit.setText(path)
            self._log(f"dataset elegido: {path}")

    # activa carga en hilo
    def _trigger_load(self):
        path = self.path_edit.text().strip()
        if not path:
            self._log("selecciona un archivo antes de cargar", accent="#f2a365")
            return
        file_path = Path(path)
        if not file_path.exists():
            self._log("el archivo indicado no existe", accent="#f87171")
            return
        self._set_busy(True)
        self._set_progress("Cargando dataset...", True)
        worker = Worker("load", self._load_dataset_task, str(file_path))
        worker.signals.result.connect(self._handle_result)
        worker.signals.error.connect(self._handle_error)
        worker.signals.finished.connect(self._task_finished)
        self.thread_pool.start(worker)
        self._log(f"cargando dataset: {file_path}")

    # activa bfs en hilo
    def _trigger_bfs(self):
        if not self.dataset_loaded:
            self._log("carga un dataset antes de simular", accent="#f2a365")
            return
        origen = self.origin_spin.value()
        profundidad = self.depth_spin.value()
        self._set_busy(True)
        self._set_progress("Ejecutando BFS...", True)
        worker = Worker("bfs", self._bfs_task, origen, profundidad)
        worker.signals.result.connect(self._handle_result)
        worker.signals.error.connect(self._handle_error)
        worker.signals.finished.connect(self._task_finished)
        self.thread_pool.start(worker)
        self.pyvis_btn.setDisabled(True)
        self._log(f"bfs solicitado desde {origen} con profundidad {profundidad}")

    # parametrizacion aleatoria suave
    def _randomize_params(self):
        if not self.dataset_loaded or self.total_nodes == 0:
            self._log("aun no hay datos cargados", accent="#f2a365")
            return
        self.origin_spin.setValue(random.randint(0, max(0, self.total_nodes - 1)))
        self.depth_spin.setValue(random.randint(1, self.depth_slider.maximum()))
        self._log("parametros aleatorios aplicados", accent="#6ee7b7")

    # trabajo real de carga
    def _load_dataset_task(self, path: str):
        self.core.cargar_archivo(path)
        return {
            "path": path,
            "nodes": self.core.total_nodos(),
            "edges": self.core.total_aristas(),
            "memory": self.core.memoria_mb(),
            "load_time": self.core.tiempo_carga_ms(),
            "critical": self.core.nodo_mayor_grado(),
        }

    # trabajo real de bfs
    def _bfs_task(self, origen: int, profundidad: int):
        resultado = self.core.bfs(origen, profundidad)
        resultado["origen"] = origen
        resultado["profundidad"] = profundidad
        return resultado

    # router de resultados por tipo
    def _handle_result(self, task: str, data: object):
        if task == "load":
            self.dataset_loaded = True
            self.total_nodes = data["nodes"]
            self.origin_spin.setRange(0, max(0, self.total_nodes - 1))
            self._update_metrics(data)
            self._set_progress("Dataset cargado", False)
            self._log(
                f"dataset cargado | nodos {data['nodes']:,} | aristas {data['edges']:,} | memoria {data['memory']:.2f} MB",
                accent="#6ee7b7",
            )
        elif task == "bfs":
            self._set_progress("BFS completado, renderizando...", True)
            self._render_graph(data)
            self._set_progress("Listo", False)
            self._log(
                f"bfs finalizado | nodos {len(data['nodos'])} | aristas {len(data['aristas'])} | origen {data['origen']}",
                accent="#6ee7b7",
            )

    # captura errores del backend
    def _handle_error(self, message: str):
        self._set_progress("Error", False)
        self._log(f"error: {message}", accent="#f87171")

    # actualizacion de metricas en pantalla
    def _update_metrics(self, data: dict):
        self.nodes_label.setText(f"Nodos: {data['nodes']:,}")
        self.edges_label.setText(f"Aristas: {data['edges']:,}")
        self.memory_label.setText(f"Memoria: {data['memory']:.2f} MB")
        self.time_label.setText(f"Carga: {data['load_time']:.2f} ms")
        self.degree_label.setText(f"Nodo crítico: {data['critical']}")

    # renderizado del subgrafo resultante
    def _render_graph(self, data: dict):
        nodos = data["nodos"]
        aristas = data["aristas"]
        if not nodos:
            self._log("no hay nodos que visualizar", accent="#f2a365")
            return
        if len(nodos) > self.max_visual_nodes:
            nodos = nodos[: self.max_visual_nodes]
            filtro = set(nodos)
            aristas = [(u, v) for (u, v) in aristas if u in filtro and v in filtro]
            aristas = aristas[: self.max_visual_edges]
            self._log(
                f"vista limitada a {len(nodos)} nodos y {len(aristas)} aristas para mantener fluidez",
                accent="#fde047",
            )
        self._set_progress("Calculando layout del subgrafo...", True)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodos)
        graph.add_edges_from(aristas)
        if graph.number_of_nodes() == 0:
            self._log("no fue posible construir el subgrafo", accent="#f87171")
            return
        layout = self._compute_layout(graph, data.get("origen", -1))
        self._set_progress("Dibujando subgrafo interactivo...", True)
        if self.graph_window is None:
            self.graph_window = GraphWindow(self)
        self.graph_window.render(graph, layout, data.get("origen", -1))
        self.last_visual = {
            "nodes": list(graph.nodes()),
            "edges": list(graph.edges()),
            "origin": data.get("origen", -1),
        }
        self.pyvis_btn.setDisabled(False)

    def _compute_layout(self, graph: nx.DiGraph, origin: int):
        layout = {}
        if ForceAtlas2 is not None:
            layout = self._forceatlas_layout(graph)
        if not layout and origin in graph:
            layout = self._radial_layout(graph, origin)
        missing = [node for node in graph.nodes() if node not in layout]
        if missing:
            repulsion = max(0.15, 1.2 / max(1.0, math.sqrt(graph.number_of_nodes())))
            fallback = nx.spring_layout(graph, seed=24, k=repulsion, iterations=40, scale=8.0)
            for node in missing:
                layout[node] = fallback[node]
        return layout

    def _forceatlas_layout(self, graph: nx.DiGraph):
        try:
            forceatlas = ForceAtlas2(
                outboundAttractionDistribution=True,
                linLogMode=False,
                adjustSizes=True,
                edgeWeightInfluence=0.8,
                jitterTolerance=1.0,
                barnesHutOptimize=True,
                scalingRatio=8.0,
                gravity=1.0,
            )
            undirected = graph.to_undirected()
            return forceatlas.forceatlas2_networkx_layout(undirected, pos=None, iterations=200)
        except Exception:  # pragma: no cover
            return {}

    def _radial_layout(self, graph: nx.DiGraph, origin: int):
        positions = {}
        levels = {}
        queue = deque([(origin, 0)])
        visited = {origin}
        max_level = 0
        while queue:
            node, level = queue.popleft()
            levels.setdefault(level, []).append(node)
            max_level = max(max_level, level)
            for vecino in graph.neighbors(node):
                if vecino not in visited:
                    visited.add(vecino)
                    queue.append((vecino, level + 1))
        # asigna nodos no visitados a un nivel exterior
        outer_level = max_level + 1
        for node in graph.nodes():
            if node not in visited:
                levels.setdefault(outer_level, []).append(node)
        radio_base = 1.5
        radio_step = 1.4
        for level, nodos in levels.items():
            if not nodos:
                continue
            radius = radio_base + (level * radio_step)
            angle_step = (2 * math.pi) / max(1, len(nodos))
            for idx, node in enumerate(nodos):
                angle = idx * angle_step
                jitter = random.uniform(-0.08, 0.08)
                x = (radius * math.cos(angle)) + jitter
                y = (radius * math.sin(angle)) + jitter
                positions[node] = (x, y)
        return positions

    # texto bonito para bitacora
    def _export_pyvis(self):
        if not self.last_visual:
            self._log("no hay subgrafo para exportar", accent="#f2a365")
            return
        try:
            from pyvis.network import Network
        except ImportError:
            self._log("pyvis no esta instalado en el entorno", accent="#f87171")
            return
        destino = Path.cwd() / "neuronet_subgraph.html"
        net = Network(height="100vh", width="100%", bgcolor="#0f111a", font_color="#f5f5f7", directed=True)
        for node in self.last_visual["nodes"]:
            color = "#ffbd39" if node == self.last_visual["origin"] else "#6c63ff"
            net.add_node(int(node), label=str(node), color=color)
        for u, v in self.last_visual["edges"]:
            net.add_edge(int(u), int(v), color="#8888ff")
        net.write_html(str(destino), notebook=False)
        css = """
<style>
  html, body { margin: 0; height: 100%; background-color: #0f111a; }
  #mynetwork { height: 100vh !important; width: 100% !important; }
</style>
"""
        contenido = destino.read_text()
        if "</head>" in contenido:
            contenido = contenido.replace("</head>", f"{css}\n</head>", 1)
            destino.write_text(contenido)
        webbrowser.open(destino.as_uri(), new=2)
        self._log(f"subgrafo exportado a {destino}", accent="#6ee7b7")

    def _log(self, message: str, accent: str = "#f5f5f7"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"<span style='color:#7dd3fc'>[{timestamp}]</span> <span style='color:{accent}'>{message}</span>"
        self.log_view.append(formatted)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    # control de botones y barra
    def _set_busy(self, busy: bool):
        self.load_btn.setDisabled(busy)
        self.bfs_btn.setDisabled(busy)
        self.random_btn.setDisabled(busy)
        self.pyvis_btn.setDisabled(busy or not self.last_visual)

    # estado de progreso visual
    def _set_progress(self, text: str, active: bool):
        self.progress_label.setText(text)
        if active:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(text)
            self.progress_bar.show()
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.progress_bar.hide()

    # callback comun al terminar tareas
    def _task_finished(self, _task: str):
        self._set_busy(False)


# arranque de la aplicacion qt
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("NeuroNet")
    app.setStyle("Fusion")
    window = NeuroNetWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
