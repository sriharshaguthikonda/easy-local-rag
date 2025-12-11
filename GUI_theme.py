def apply_dark_theme(widget):
    widget.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
            QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2D2D2D;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                padding: 4px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #007ACC;
            }
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QPushButton:disabled {
                background-color: #3C3C3C;
                color: #808080;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3C3C3C;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #3C3C3C;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #D4D4D4;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1E1E1E;
                border-bottom: 2px solid #007ACC;
            }
            QScrollBar:vertical {
                background-color: #2D2D2D;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #3C3C3C;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4C4C4C;
            }
            QTableWidget {
                background-color: #2D2D2D;
                gridline-color: #3C3C3C;
            }
            QHeaderView::section {
                background-color: #252526;
                color: #D4D4D4;
                padding: 5px;
                border: 1px solid #3C3C3C;
            }
            QTreeWidget {
                background-color: #2D2D2D;
            }
            QTreeWidget::item:hover {
                background-color: #3C3C3C;
            }
            QTreeWidget::item:selected {
                background-color: #094771;
            }
            QToolBar {
                background-color: #252526;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #007ACC;
                color: white;
            }
            QProgressBar {
                border: none;
                background-color: #3C3C3C;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #0E639C;
                border-radius: 4px;
            }
        """)


def apply_light_theme(widget):
    widget.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #F5F5F5;
                color: #333333;
            }
            QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 4px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #0078D4;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #808080;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #E5E5E5;
                color: #333333;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #F5F5F5;
                border-bottom: 2px solid #0078D4;
            }
            QTableWidget {
                background-color: white;
                gridline-color: #CCCCCC;
            }
            QHeaderView::section {
                background-color: #E5E5E5;
                color: #333333;
                padding: 5px;
                border: 1px solid #CCCCCC;
            }
            QTreeWidget {
                background-color: white;
            }
            QTreeWidget::item:hover {
                background-color: #E5E5E5;
            }
            QTreeWidget::item:selected {
                background-color: #CCE4F7;
            }
            QToolBar {
                background-color: #E5E5E5;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #0078D4;
                color: white;
            }
        """)
