import math
import random
import sys
import webbrowser
from collections import deque, defaultdict

try:
    from fa2 import ForceAtlas2
except ImportError:  # pragma: no cover
    ForceAtlas2 = None
from datetime import datetime
from pathlib import Path

import networkx as nx
from PyQt6 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from neuronet_core import NeuroNetCore


# senales livianas para comunicar hilos
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(str, object)
    progress = QtCore.pyqtSignal(str) # Nuevo signal para mensajes de progreso


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
            # Pasar callback de progreso si la funcion lo acepta
            if "progress_callback" in self.kwargs:
                self.kwargs["progress_callback"] = self.signals.progress.emit
            else:
                # Inyectarlo como ultimo argumento si no esta en kwargs
                pass
                
            data = self.fn(*self.args, **self.kwargs, progress_callback=self.signals.progress.emit)
        except TypeError:
            # Si falla por argumentos, intentar sin callback (compatibilidad)
            data = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(self.task, data)
        finally:
            self.signals.finished.emit(self.task)


# ventana dedicada para el subgrafo interactivo con Matplotlib
class GraphWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subgrafo NeuroNet")
        self.resize(1200, 800)
        layout = QtWidgets.QVBoxLayout(self)
        
        # Matplotlib setup
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Estilo de la toolbar para que coincida con el tema (Cyan para botones, fondo oscuro)
        self.toolbar.setStyleSheet("background-color: #4ECDC4; color: #1e1e1e; border: none;")
        
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("color:#e0e0e0;font-size:14px;font-weight:bold;font-family:'Segoe UI'")
        
        layout.addWidget(self.info_label)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def render(self, graph: nx.DiGraph, layout_positions: dict[int, tuple[float, float]], origen: int):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        
        if not layout_positions:
            self.canvas.draw()
            return

        # Styling
        node_colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6', '#3498DB', '#E67E22', '#2ECC71', '#F1C40F', '#E74C3C', '#1ABC9C', '#8E44AD', '#34495E']
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, 
            pos=layout_positions, 
            ax=ax, 
            edge_color='#555555', 
            width=1.5, 
            alpha=0.7, 
            arrows=True, 
            arrowstyle='-|>', 
            arrowsize=12
        )
        
        # Assign colors
        node_colors = []
        for i, node in enumerate(graph.nodes()):
            if node == origen:
                node_colors.append('#ffffff') # Origin distinct
            else:
                node_colors.append(node_colors_list[i % len(node_colors_list)])
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, 
            pos=layout_positions, 
            ax=ax, 
            node_size=500, 
            node_color=node_colors, 
            edgecolors='#ffffff', 
            linewidths=1.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, 
            pos=layout_positions, 
            ax=ax, 
            font_size=10, 
            font_color='white', 
            font_family='sans-serif', 
            font_weight='bold'
        )
        
        self.info_label.setText(
            f"Nodos: {graph.number_of_nodes()} | Aristas: {graph.number_of_edges()} | Origen: {origen}"
        )
        self.canvas.draw()
        self.show()
        self.raise_()
        self.activateWindow()


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
        self.showMaximized()
        self.dataset_loaded = False
        self.total_nodes = 0
        self.graph_window = None
        self.max_visual_nodes = 100
        self.max_visual_edges = 200
        self.last_visual = None

        self._init_palette()
        self._build_ui()
        self._refresh_dataset_options()
        self._set_progress("Listo", False)

    # paleta dark mode solicitada
    def _init_palette(self):
        palette = QtGui.QPalette()
        # Fondo oscuro profundo
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#121212"))
        # Base para inputs
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#1e1e1e"))
        # Texto principal
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e0e0e0"))
        # Botones base
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#4ECDC4"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#1e1e1e"))
        # Acento
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#4ECDC4"))
        self.setPalette(palette)
        
        # Estilos CSS modernos dark mode
        self.setStyleSheet(
            """
            QWidget {
                font-family: 'Segoe UI', sans-serif; 
                font-size: 13px;
                color: #e0e0e0;
            }
            QMainWindow {
                background-color: #121212;
            }
            QPushButton {
                background-color: #4ECDC4;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                color: #1e1e1e;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3db9b0;
            }
            QPushButton:pressed {
                background-color: #2a9d8f;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #555555;
            }
            /* Botones de accion principal con mismo estilo (flat cyan) */
            QPushButton#ActionBtn {
                background-color: #4ECDC4;
                color: #1e1e1e;
            }
            QPushButton#ActionBtn:hover {
                background-color: #3db9b0;
            }
            QPushButton#ActionBtn:disabled {
                background-color: #2d2d2d;
                color: #555555;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 8px;
                margin-top: 14px;
                padding: 16px;
                background-color: #1e1e1e;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #121212;
            }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                selection-background-color: #4ECDC4;
                selection-color: #1e1e1e;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #4ECDC4;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                border-radius: 8px;
                color: #e0e0e0;
                padding: 8px;
            }
            QProgressBar {
                background-color: #2d2d2d;
                border: none;
                border-radius: 6px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #4ECDC4;
                border-radius: 6px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QRadioButton {
                color: #e0e0e0;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #444444;
                background-color: #2d2d2d;
            }
            QRadioButton::indicator:checked {
                background-color: #4ECDC4;
                border-color: #4ECDC4;
            }
            QRadioButton::indicator:hover {
                border-color: #4ECDC4;
            }
            """
        )

    # ... (omitted code) ...

    def _tidy_tree_positions(
        self,
        children: defaultdict[int, list[int]],
        depth: dict[int, int],
        origin: int,
    ) -> dict[int, tuple[float, float]]:
        if origin not in depth:
            return {}
        subtree: dict[int, float] = {}

        def _size(node: int) -> float:
            hijos = children.get(node, [])
            if not hijos:
                subtree[node] = 1.0
                return 1.0
            total = 0.0
            for hijo in hijos:
                total += _size(hijo)
            subtree[node] = max(total, 1.0)
            return subtree[node]

        def _assign(node: int, start: float, positions: dict[int, float]):
            hijos = children.get(node, [])
            if not hijos:
                positions[node] = start + 0.5
                return positions[node]
            cursor = start
            # Reducimos la separacion base para que no esten tan lejos
            separacion = 1.0 + min(0.5, max(0, len(hijos) - 1) * 0.05)
            for hijo in hijos:
                _assign(hijo, cursor, positions)
                cursor += subtree[hijo] * separacion
            primero = hijos[0]
            ultimo = hijos[-1]
            positions[node] = (positions[primero] + positions[ultimo]) / 2.0
            return positions[node]

        _size(origin)
        x_positions: dict[int, float] = {}
        _assign(origin, 0.0, x_positions)

        min_x = min(x_positions.values(), default=0.0)
        max_depth = max(depth.values(), default=0)
        
        # Calculo dinamico de espaciado basado en la densidad del nivel mas ancho
        level_counts: defaultdict[int, int] = defaultdict(int)
        for node, level in depth.items():
            level_counts[level] += 1
        max_level_width = max(level_counts.values(), default=1)
        
        # Formula logaritmica para suavizar el espaciado
        # Si hay pocos nodos, espaciado generoso (80). Si hay muchos, se compacta un poco pero no demasiado.
        base_spacing = 80.0
        dynamic_factor = 150.0 / (1.0 + math.log10(max_level_width))
        spacing_x = base_spacing + dynamic_factor
        
        spacing_y = 90.0 # Altura fija entre niveles, extremadamente compacta
        
        layout: dict[int, tuple[float, float]] = {}
        for node, x_val in x_positions.items():
            centered_x = (x_val - min_x) * spacing_x
            layout[node] = (centered_x, depth[node] * spacing_y)
        min_y = min((pos[1] for pos in layout.values()), default=0.0)
        if min_y != 0.0:
            for node, (x, y) in layout.items():
                layout[node] = (x, y - min_y)
        return layout

    # armado de la interfaz general
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        self._build_top_bar(layout)
        self._build_metrics(layout)

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
        self.load_btn.setObjectName("ActionBtn")
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
        self.origin_spin.valueChanged.connect(lambda v: self._log(f"Origen cambiado a {v}"))

        self.origin_one_btn = QtWidgets.QPushButton("Origen 1")
        self.origin_one_btn.setFixedWidth(80)
        self.origin_one_btn.clicked.connect(lambda: self.origin_spin.setValue(1))
        self.origin_one_btn.clicked.connect(lambda: self._log("Origen restablecido a 1"))
        self.origin_one_btn.setDisabled(True)

        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(1, 10)
        self.depth_spin.setPrefix("Prof ")
        self.depth_spin.valueChanged.connect(lambda v: self._log(f"Profundidad cambiada a {v}"))

        self.depth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(2)
        self.depth_slider.valueChanged.connect(self.depth_spin.setValue)
        self.depth_spin.valueChanged.connect(self.depth_slider.setValue)

        self.unlimited_check = QtWidgets.QCheckBox("Ilimitada")
        self.unlimited_check.toggled.connect(lambda checked: self.depth_spin.setDisabled(checked))
        self.unlimited_check.toggled.connect(lambda checked: self.depth_slider.setDisabled(checked))
        self.unlimited_check.toggled.connect(lambda checked: self._log(f"Profundidad ilimitada: {'Activada' if checked else 'Desactivada'}", type="WARNING" if checked else "INFO"))

        self.random_btn = QtWidgets.QPushButton("Aleatorio")
        self.random_btn.clicked.connect(self._randomize_params)
        self.random_btn.setDisabled(True) # Deshabilitado hasta cargar dataset

        self.bfs_btn = QtWidgets.QPushButton("Generar subgrafo")
        self.bfs_btn.setObjectName("ActionBtn")
        self.bfs_btn.clicked.connect(lambda: self._trigger_bfs("render"))
        self.bfs_btn.setDisabled(True) # Deshabilitado explicitamente al inicio

        self.pyvis_btn = QtWidgets.QPushButton("Exportar a PyVis")
        self.pyvis_btn.clicked.connect(lambda: self._trigger_bfs("pyvis"))
        self.pyvis_btn.setDisabled(True)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet(
            "QProgressBar {background-color:#2d2d2d;border:1px solid #444444;border-radius:10px;color:#e0e0e0;}"
            "QProgressBar::chunk {background-color:#4ECDC4;border-radius:9px;}"
        )
        self.progress_bar.setFormat("%p%")
        self.progress_bar.hide()

        # Row 0: Origin controls
        bfs_layout.addWidget(QtWidgets.QLabel("Nodo origen"), 0, 0)
        bfs_layout.addWidget(self.origin_spin, 0, 1)
        bfs_layout.addWidget(self.random_btn, 0, 2)
        bfs_layout.addWidget(self.origin_one_btn, 0, 3)

        # Row 1: Depth controls
        bfs_layout.addWidget(QtWidgets.QLabel("Profundidad"), 1, 0)
        bfs_layout.addWidget(self.depth_spin, 1, 1)
        bfs_layout.addWidget(self.depth_slider, 1, 2)
        bfs_layout.addWidget(self.unlimited_check, 1, 3)

        # Row 2: Algorithm
        algo_layout = QtWidgets.QHBoxLayout()
        self.radio_bfs = QtWidgets.QRadioButton("BFS")
        self.radio_dfs = QtWidgets.QRadioButton("DFS")
        self.radio_bfs.setChecked(True)
        self.radio_bfs.toggled.connect(lambda c: self._log("Algoritmo cambiado a BFS") if c else None)
        self.radio_dfs.toggled.connect(lambda c: self._log("Algoritmo cambiado a DFS") if c else None)
        algo_layout.addWidget(self.radio_bfs)
        algo_layout.addWidget(self.radio_dfs)
        bfs_layout.addLayout(algo_layout, 2, 1, 1, 2)

        # Row 3: Visual Settings (Integrated)
        settings_group = QtWidgets.QGroupBox("Límites Visuales")
        settings_layout = QtWidgets.QGridLayout(settings_group)
        
        self.spin_nodes = QtWidgets.QSpinBox()
        self.spin_nodes.setRange(10, 5000)
        self.spin_nodes.setValue(self.max_visual_nodes)
        self.spin_nodes.setPrefix("Nodos: ")
        self.spin_nodes.valueChanged.connect(self._update_visual_limits)
        
        self.spin_edges = QtWidgets.QSpinBox()
        self.spin_edges.setRange(10, 10000)
        self.spin_edges.setValue(self.max_visual_edges)
        self.spin_edges.setPrefix("Aristas: ")
        self.spin_edges.valueChanged.connect(self._update_visual_limits)
        
        self.check_unlimited_visual = QtWidgets.QCheckBox("Sin límites")
        self.check_unlimited_visual.toggled.connect(self._toggle_unlimited_visual)
        
        settings_layout.addWidget(self.spin_nodes, 0, 0)
        settings_layout.addWidget(self.spin_edges, 0, 1)
        settings_layout.addWidget(self.check_unlimited_visual, 0, 2)
        
        bfs_layout.addWidget(settings_group, 3, 0, 1, 4)

        # Row 4: Actions (Stacked)
        bfs_layout.addWidget(self.bfs_btn, 4, 0, 1, 4)
        bfs_layout.addWidget(self.pyvis_btn, 5, 0, 1, 4)
        
        # Row 6: Progress
        bfs_layout.addWidget(self.progress_bar, 6, 0, 1, 4)

        parent_layout.addWidget(bfs_group)


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
                self._log(f"Dataset seleccionado: {selected[0]}")

    # cuando cambia elemento rapido
    def _combo_selected(self, index: int):
        path = self.dataset_combo.itemData(index)
        if path:
            self.path_edit.setText(path)
            self._log(f"Dataset elegido: {path}")

    # activa carga en hilo
    def _trigger_load(self):
        path = self.path_edit.text().strip()
        if not path:
            self._log("Selecciona un archivo antes de cargar", type="WARNING")
            return
        file_path = Path(path)
        if not file_path.exists():
            self._log("El archivo indicado no existe", type="ERROR")
            return
        self._set_busy(True)
        self._set_progress("Cargando dataset en C++ (Espere)...", True)
        worker = Worker("load", self._load_dataset_task, str(file_path))
        worker.signals.result.connect(self._handle_result)
        worker.signals.error.connect(self._handle_error)
        worker.signals.finished.connect(self._task_finished)
        self.thread_pool.start(worker)
        self._log(f"Cargando dataset: {file_path}")

    # iniciar busqueda (visual o exportacion)
    def _trigger_bfs(self, target: str = "render"):
        if not self.dataset_loaded:
            self._log("Carga un dataset antes de simular", type="WARNING")
            return
        origen = self.origin_spin.value()
        profundidad = 1000000 if self.unlimited_check.isChecked() else self.depth_spin.value()
        algo = "dfs" if self.radio_dfs.isChecked() else "bfs"
        
        self._set_busy(True)
        msg = f"Ejecutando {algo.upper()}..." if target == "render" else f"Generando {algo.upper()} para PyVis..."
        self._set_progress(msg, True)
        
        worker = Worker("search", self._search_task, origen, profundidad, algo, target)
        worker.signals.result.connect(self._handle_result)
        worker.signals.error.connect(self._handle_error)
        worker.signals.finished.connect(self._task_finished)
        worker.signals.progress.connect(lambda msg: self._log(msg, type="INFO")) # Conectar progreso a logs
        self.thread_pool.start(worker)
        self._log(f"{algo.upper()} solicitado desde {origen} (Target: {target})")

    def _update_visual_limits(self):
        if not self.check_unlimited_visual.isChecked():
            self.max_visual_nodes = self.spin_nodes.value()
            self.max_visual_edges = self.spin_edges.value()

    def _toggle_unlimited_visual(self, checked: bool):
        self.spin_nodes.setDisabled(checked)
        self.spin_edges.setDisabled(checked)
        if checked:
            self.max_visual_nodes = 10000000
            self.max_visual_edges = 20000000
        else:
            self.max_visual_nodes = self.spin_nodes.value()
            self.max_visual_edges = self.spin_edges.value()

    # parametrizacion aleatoria suave
    def _randomize_params(self):
        if not self.dataset_loaded or self.total_nodes == 0:
            self._log("Aun no hay datos cargados", type="WARNING")
            return
        self.origin_spin.setValue(random.randint(0, max(0, self.total_nodes - 1)))
        self.depth_spin.setValue(random.randint(1, self.depth_slider.maximum()))
        self._log("Parametros aleatorios aplicados", type="SUCCESS")

    # tarea de carga en hilo
    def _load_dataset_task(self, path: str, progress_callback=None):
        if progress_callback:
            progress_callback("Iniciando carga en C++...")
            
        # Nota: cargar_archivo lanza RuntimeError si falla, asi que el try/except del worker lo capturara
        self.core.cargar_archivo(path)
            
        return {
            "path": path,
            "nodes": self.core.total_nodos(),
            "edges": self.core.total_aristas(),
            "memory": self.core.memoria_mb(),
            "load_time": self.core.tiempo_carga_ms(),
            "critical": self.core.nodo_mayor_grado(),
        }

    # trabajo real de busqueda (bfs/dfs)
    def _search_task(self, origen: int, profundidad: int, algo: str, target: str, progress_callback=None):
        if progress_callback:
            progress_callback(f"Iniciando {algo.upper()} en C++...")
            
        if algo == "dfs":
            resultado = self.core.dfs(origen, profundidad)
        else:
            resultado = self.core.bfs(origen, profundidad)
            
        if progress_callback:
            progress_callback(f"{algo.upper()} completado. Procesando resultados...")
        
        resultado["origen"] = origen
        resultado["profundidad"] = profundidad
        resultado["algo"] = algo
        resultado["target"] = target
        
        # Optimizacion: Calcular layout en segundo plano si es para renderizar
        if target == "render":
            nodos = resultado["nodos"]
            aristas = resultado["aristas"]
            
            # Aplicar limites visuales aqui para no calcular layout de mas
            if len(nodos) > self.max_visual_nodes:
                nodos = nodos[: self.max_visual_nodes]
                filtro = set(nodos)
                aristas = [(u, v) for (u, v) in aristas if u in filtro and v in filtro]
                resultado["limited"] = True
                resultado["visual_nodes"] = nodos
                resultado["visual_edges"] = aristas
            else:
                resultado["limited"] = False
                resultado["visual_nodes"] = nodos
                resultado["visual_edges"] = aristas

            # Construir grafo temporal para layout
            graph = nx.DiGraph()
            graph.add_nodes_from(resultado["visual_nodes"])
            graph.add_edges_from(resultado["visual_edges"])
            
            if graph.number_of_nodes() > 0:
                if progress_callback:
                    progress_callback("Calculando layout (0%)...")
                # Calcular layout pesado aqui
                layout = self._compute_layout(graph, origen)
                if progress_callback:
                    progress_callback("Calculando layout (100%)...")
                resultado["layout"] = layout
            else:
                resultado["layout"] = {}
        
        # Si el target es PyVis, generamos el HTML aqui mismo en el hilo para reportar progreso
        elif target == "pyvis":
             if progress_callback:
                progress_callback("Generando estructura PyVis (0%)...")
             self._generate_pyvis_content(resultado, progress_callback)
             if progress_callback:
                progress_callback("Generando estructura PyVis (100%)...")
                
        return resultado

    # router de resultados por tipo
    def _handle_result(self, task: str, data: object):
        if task == "load":
            self.dataset_loaded = True
            self.total_nodes = data["nodes"]
            self.origin_spin.setRange(0, max(0, self.total_nodes - 1))
            self._update_metrics(data)
            self._set_progress("Dataset cargado", False)
            self.pyvis_btn.setDisabled(False) # Habilitar exportacion directa
            self.bfs_btn.setDisabled(False) # Habilitar generacion
            self.origin_one_btn.setDisabled(False) # Habilitar origen 1
            self.random_btn.setDisabled(False) # Habilitar random
            self._log(
                f"Dataset cargado | Nodos {data['nodes']:,} | Aristas {data['edges']:,} | Memoria {data['memory']:.2f} MB",
                type="SUCCESS",
            )
        elif task == "search":
            target = data.get("target", "render")
            if target == "render":
                self._set_progress("Búsqueda completada, renderizando...", True)
                self._render_graph(data)
                self._set_progress("Listo", False)
            elif target == "pyvis":
                # El HTML ya se genero en el hilo, solo abrimos el navegador
                self._set_progress("Abriendo navegador...", True)
                destino = Path.cwd() / "neuronet_subgraph.html"
                if destino.exists():
                    webbrowser.open(destino.as_uri(), new=2)
                    self._log(f"Subgrafo exportado a {destino}", type="SUCCESS")
                self._set_progress("Exportado", False)
            
            self._log(
                f"{data.get('algo', 'bfs').upper()} finalizado | Nodos {len(data['nodos'])} | Aristas {len(data['aristas'])}",
                type="SUCCESS",
            )

    # captura errores del backend
    def _handle_error(self, message: str):
        self._set_progress("Error", False)
        self._log(f"Error: {message}", type="ERROR")

    # actualizacion de metricas en pantalla
    def _update_metrics(self, data: dict):
        self.nodes_label.setText(f"Nodos: {data['nodes']:,}")
        self.edges_label.setText(f"Aristas: {data['edges']:,}")
        self.memory_label.setText(f"Memoria: {data['memory']:.2f} MB")
        self.time_label.setText(f"Carga: {data['load_time']:.2f} ms")
        self.degree_label.setText(f"Nodo crítico: {data['critical']}")

    # renderizado del subgrafo resultante
    def _render_graph(self, data: dict):
        # Usar datos pre-procesados en el hilo
        nodos = data.get("visual_nodes", [])
        aristas = data.get("visual_edges", [])
        layout = data.get("layout", {})
        
        if not nodos:
            self._log("No hay nodos que visualizar", type="WARNING")
            return
            
        if data.get("limited", False):
            self._log(
                f"Vista limitada a {len(nodos)} nodos y {len(aristas)} aristas para mantener fluidez",
                type="WARNING",
            )
            
        self._set_progress("Dibujando subgrafo interactivo...", True)
        
        # Reconstruir grafo ligero solo para dibujo
        graph = nx.DiGraph()
        graph.add_nodes_from(nodos)
        graph.add_edges_from(aristas)
        
        if self.graph_window is None:
            self.graph_window = GraphWindow(self)
            
        # Pasar layout pre-calculado
        self.graph_window.render(graph, layout, data.get("origen", -1))
        
        self.last_visual = {
            "nodes": list(graph.nodes()),
            "edges": list(graph.edges()),
            "origin": data.get("origen", -1),
        }
        self.pyvis_btn.setDisabled(False)

    def _compute_layout(self, graph: nx.DiGraph, origin: int):
        layout = {}
        if origin in graph:
            layout = self._tree_layout(graph, origin)
        if not layout and ForceAtlas2 is not None:
            layout = self._forceatlas_layout(graph)
        if not layout and origin in graph:
            layout = self._radial_layout(graph, origin)
        missing = [node for node in graph.nodes() if node not in layout]
        if missing:
            # Aumentar repulsion para evitar superposicion
            repulsion = max(0.3, 2.0 / max(1.0, math.sqrt(graph.number_of_nodes())))
            fallback = nx.spring_layout(graph, seed=24, k=repulsion, iterations=50, scale=10.0)
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

    def _tree_layout(self, graph: nx.DiGraph, origin: int):
        if origin not in graph:
            return {}
        children, depth, _visited = self._bfs_tree(graph, origin)
        if not depth:
            return {}
        node_count = graph.number_of_nodes()
        # SIEMPRE usar tidy tree layout para garantizar estructura de arbol bonita
        # sin importar el tamano (hasta el limite visual)
        layout = self._tidy_tree_positions(children, depth, origin)
        
        missing = [node for node in graph.nodes() if node not in layout]
        if missing:
            layout.update(self._grid_fallback_positions(missing, depth))
        return layout

 
    def _bfs_tree(self, graph: nx.DiGraph, origin: int):
        children: defaultdict[int, list[int]] = defaultdict(list)
        depth: dict[int, int] = {}
        visited: set[int] = set()
        if origin not in graph:
            return children, depth, visited
        undirected = graph.to_undirected()
        queue = deque([origin])
        depth[origin] = 0
        visited.add(origin)
        while queue:
            node = queue.popleft()
            vecinos_dir = list(graph.successors(node))
            vecinos = vecinos_dir if vecinos_dir else list(undirected.neighbors(node))
            for vecino in sorted(vecinos):
                if vecino in visited:
                    continue
                visited.add(vecino)
                depth[vecino] = depth[node] + 1
                children[node].append(vecino)
                queue.append(vecino)
        return children, depth, visited

    def _tidy_tree_positions(
        self,
        children: defaultdict[int, list[int]],
        depth: dict[int, int],
        origin: int,
    ) -> dict[int, tuple[float, float]]:
        if origin not in depth:
            return {}
        subtree: dict[int, float] = {}

        def _size(node: int) -> float:
            hijos = children.get(node, [])
            if not hijos:
                subtree[node] = 1.0
                return 1.0
            total = 0.0
            for hijo in hijos:
                total += _size(hijo)
            subtree[node] = max(total, 1.0)
            return subtree[node]

        def _assign(node: int, start: float, positions: dict[int, float]):
            hijos = children.get(node, [])
            if not hijos:
                positions[node] = start + 0.5
                return positions[node]
            cursor = start
            # Reducimos la separacion base para que no esten tan lejos
            separacion = 1.0 + min(0.5, max(0, len(hijos) - 1) * 0.05)
            for hijo in hijos:
                _assign(hijo, cursor, positions)
                cursor += subtree[hijo] * separacion
            primero = hijos[0]
            ultimo = hijos[-1]
            positions[node] = (positions[primero] + positions[ultimo]) / 2.0
            return positions[node]

        _size(origin)
        x_positions: dict[int, float] = {}
        _assign(origin, 0.0, x_positions)

        min_x = min(x_positions.values(), default=0.0)
        max_depth = max(depth.values(), default=0)
        
        # Calculo dinamico de espaciado basado en la densidad del nivel mas ancho
        level_counts: defaultdict[int, int] = defaultdict(int)
        for node, level in depth.items():
            level_counts[level] += 1
        max_level_width = max(level_counts.values(), default=1)
        
        # Formula logaritmica para suavizar el espaciado
        # Si hay pocos nodos, espaciado generoso (80). Si hay muchos, se compacta un poco pero no demasiado.
        base_spacing = 80.0
        dynamic_factor = 150.0 / (1.0 + math.log10(max_level_width))
        spacing_x = base_spacing + dynamic_factor
        
        spacing_y = 140.0 # Altura fija entre niveles, mas compacta que antes
        
        layout: dict[int, tuple[float, float]] = {}
        for node, x_val in x_positions.items():
            centered_x = (x_val - min_x) * spacing_x
            layout[node] = (centered_x, depth[node] * spacing_y)
        min_y = min((pos[1] for pos in layout.values()), default=0.0)
        if min_y != 0.0:
            for node, (x, y) in layout.items():
                layout[node] = (x, y - min_y)
        return layout

    def _layered_force_positions(self, graph: nx.DiGraph, depth: dict[int, int]):
        undirected = graph.to_undirected()
        node_count = max(1, graph.number_of_nodes())
        k = max(0.25, 1.5 / math.sqrt(node_count))
        pos = nx.spring_layout(undirected, seed=32, k=k, iterations=80, scale=1.0)
        xs = [coord[0] for coord in pos.values()]
        ys = [coord[1] for coord in pos.values()]
        min_x, max_x = min(xs, default=0.0), max(xs, default=1.0)
        min_y, max_y = min(ys, default=0.0), max(ys, default=1.0)
        span_x = max(max_x - min_x, 1e-4)
        span_y = max(max_y - min_y, 1e-4)
        horizontal_scale = 1000.0 + (math.log10(node_count + 10) * 420.0)
        max_depth = max(depth.values(), default=0)
        spacing_y = max(150.0, 220.0 - (math.log10(node_count + 3) * 20.0))
        jitter_scale = spacing_y * 0.55
        default_depth = max_depth + 1
        layout: dict[int, tuple[float, float]] = {}
        for node in graph.nodes():
            base = pos.get(node, (0.0, 0.0))
            normalized_x = ((base[0] - min_x) / span_x) - 0.5
            normalized_y = ((base[1] - min_y) / span_y) - 0.5
            depth_level = depth.get(node, default_depth)
            x = normalized_x * horizontal_scale
            y = (depth_level * spacing_y) + (normalized_y * jitter_scale)
            layout[node] = (x, y)
        min_y_layout = min((coord[1] for coord in layout.values()), default=0.0)
        for node, (x, y) in layout.items():
            adjusted_y = y - min_y_layout
            if node in depth and depth[node] == 0:
                adjusted_y = max(0.0, adjusted_y - spacing_y * 0.35)
            layout[node] = (x, adjusted_y)
        return layout

    def _grid_fallback_positions(self, nodes: list[int], depth: dict[int, int]):
        if not nodes:
            return {}
        max_depth = max(depth.values(), default=0)
        start_level = max_depth + 1
        columns = max(3, int(math.sqrt(len(nodes))))
        spacing_x = 170.0
        spacing_y = 190.0
        layout: dict[int, tuple[float, float]] = {}
        for idx, node in enumerate(sorted(nodes)):
            col = idx % columns
            row = idx // columns
            offset_x = (col - (columns / 2)) * spacing_x * 1.1
            y = (start_level + row) * spacing_y
            layout[node] = (offset_x, y)
        return layout

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
            self._log("No hay subgrafo para exportar", type="WARNING")
            return
        try:
            from pyvis.network import Network
        except ImportError:
            self._log("Pyvis no esta instalado en el entorno", type="ERROR")
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
        self._log(f"Subgrafo exportado a {destino}", type="SUCCESS")

    # Generacion de contenido PyVis en hilo (worker)
    def _generate_pyvis_content(self, data: dict, progress_callback=None):
        try:
            from pyvis.network import Network
        except ImportError:
            return
            
        nodos = data["nodos"]
        aristas = data["aristas"]
        origen = data["origen"]
        
        destino = Path.cwd() / "neuronet_subgraph.html"
        net = Network(height="100vh", width="100%", bgcolor="#0f111a", font_color="#f5f5f7", directed=True)
        
        unique_nodes = list(set(nodos))
        total_nodes = len(unique_nodes)
        
        # Agregar nodos con progreso
        for i, node in enumerate(unique_nodes):
            if progress_callback and i % 1000 == 0:
                 porcentaje = int((i / total_nodes) * 50) # Primer 50%
                 progress_callback(f"Generando PyVis: Nodos {porcentaje}%")
                 
            color = "#ffbd39" if node == origen else "#6c63ff"
            net.add_node(int(node), label=str(node), color=color)
            
        # Agregar aristas con progreso
        total_edges = len(aristas)
        unique_nodes_set = set(unique_nodes)
        for i, (u, v) in enumerate(aristas):
            if progress_callback and i % 1000 == 0:
                 porcentaje = 50 + int((i / total_edges) * 50) # Segundo 50%
                 progress_callback(f"Generando PyVis: Aristas {porcentaje}%")

            if u in unique_nodes_set and v in unique_nodes_set:
                net.add_edge(int(u), int(v), color="#8888ff")
                
        net.write_html(str(destino), notebook=False)
        
        # Inyectar CSS
        css = """
<style>
  html, body { margin: 0; height: 100%; background-color: #0f111a; }
  #mynetwork { height: 100vh !important; width: 100% !important; }
</style>
"""
        try:
            contenido = destino.read_text()
            if "</head>" in contenido:
                contenido = contenido.replace("</head>", f"{css}\n</head>", 1)
                destino.write_text(contenido)
        except Exception:
            pass

    # exportacion directa (legacy/wrapper si se necesita desde main thread)
    def _export_pyvis_direct(self, data: dict):
        # Esta funcion ahora solo se usa si se llama desde main thread, 
        # pero la logica principal se movio a _generate_pyvis_content
        self._generate_pyvis_content(data)
        destino = Path.cwd() / "neuronet_subgraph.html"
        webbrowser.open(destino.as_uri(), new=2)
        self._log(f"Subgrafo exportado a {destino}", type="SUCCESS")

    def _log(self, message: str, type: str = "INFO", accent: str = None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Definir colores segun el tipo si no se especifica acento (Paleta Vibrante)
        if accent is None:
            if type == "ERROR":
                accent = "#E74C3C" # Red
            elif type == "WARNING":
                accent = "#F1C40F" # Yellow
            elif type == "SUCCESS":
                accent = "#2ECC71" # Green
            else:
                accent = "#3498DB" # Blue
        
        # Formatear mensaje
        # Timestamp en gris suave, Tipo en negrita con color, Mensaje en blanco
        formatted = (
            f"<span style='color:#888888'>[{timestamp}]</span> "
            f"<strong style='color:{accent}'>[{type}]</strong> "
            f"<span style='color:#e0e0e0'>{message}</span>"
        )
        self.log_view.append(formatted)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    # control de botones y barra
    def _set_busy(self, busy: bool):
        self.load_btn.setDisabled(busy)
        self.bfs_btn.setDisabled(busy)
        self.random_btn.setDisabled(busy)
        self.pyvis_btn.setDisabled(busy) # Ahora pyvis tambien es una accion asincrona


    # estado de progreso visual
    def _set_progress(self, msg: str, busy: bool):
        if busy:
            self.progress_bar.setFormat(msg)
            self.progress_bar.setRange(0, 0) # Indeterminate
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
