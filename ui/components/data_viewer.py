from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTableView, QMenu, 
                               QInputDialog, QMessageBox, QHeaderView)
from PySide6.QtCore import Qt, QAbstractTableModel, Signal
from PySide6.QtGui import QAction, QCursor
import pandas as pd

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                val = self._data.iloc[index.row(), index.column()]
                return str(val)
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
    
    def set_data(self, data):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

class DataViewer(QWidget):
    data_changed = Signal()

    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableView()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.table)
    
    def set_dataframe(self, df):
        self.df = df
        self.model = PandasModel(self.df)
        self.table.setModel(self.model)

    def get_dataframe(self):
        return self.df

    def show_context_menu(self, pos):
        if self.df.empty:
            return

        index = self.table.indexAt(pos)
        header_height = self.table.horizontalHeader().height()
        
        # Check if right-click is on header or cell
        # Simple hack: if y < header_height, it's near header, but QTableView context menu logic is tricky.
        # Alternatively, we just allow operations on the column of the selected cell.
        
        col_idx = -1
        if index.isValid():
            col_idx = index.column()
        else:
            # Maybe they clicked the blank space?
            return

        menu = QMenu(self)
        
        col_name = self.df.columns[col_idx]
        
        action_rename = QAction(f"Rename '{col_name}'", self)
        action_rename.triggered.connect(lambda: self.rename_column(col_idx))
        menu.addAction(action_rename)
        
        action_drop = QAction(f"Drop Column '{col_name}'", self)
        action_drop.triggered.connect(lambda: self.drop_column(col_idx))
        menu.addAction(action_drop)
        
        menu.addSeparator()
        
        action_info = QAction("Column Info", self)
        action_info.triggered.connect(lambda: self.show_info(col_idx))
        menu.addAction(action_info)

        menu.exec(QCursor.pos())

    def rename_column(self, col_idx):
        old_name = self.df.columns[col_idx]
        new_name, ok = QInputDialog.getText(self, "Rename Column", "New Name:", text=old_name)
        if ok and new_name:
            self.df.rename(columns={old_name: new_name}, inplace=True)
            self.model.set_data(self.df)
            self.data_changed.emit()

    def drop_column(self, col_idx):
        col_name = self.df.columns[col_idx]
        reply = QMessageBox.question(self, "Confirm Drop", f"Are you sure you want to delete '{col_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.df.drop(columns=[col_name], inplace=True)
            self.model.set_data(self.df)
            self.data_changed.emit()

    def show_info(self, col_idx):
        col_name = self.df.columns[col_idx]
        dtype = self.df[col_name].dtype
        missing = self.df[col_name].isna().sum()
        unique = self.df[col_name].nunique()
        msg = f"Column: {col_name}\nType: {dtype}\nMissing: {missing}\nUnique: {unique}"
        QMessageBox.information(self, "Column Info", msg)
