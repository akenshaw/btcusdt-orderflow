import sys, logging, json, gc, time, collections
import numpy as np
from collections import deque

from PyQt6 import QtCore, QtWidgets, QtWebSockets
from PyQt6.QtNetwork import QNetworkRequest, QNetworkAccessManager
import pyqtgraph as pg
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QSizePolicy, QLayout, QGridLayout, QBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QInputDialog, QVBoxLayout, QTableWidgetItem, QTableWidget
import pyqtgraph.opengl as gl
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class ConnectorWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(ConnectorWindow, self).__init__(*args, **kwargs)

        self.symbol_input = QtWidgets.QLineEdit(self)
        self.connect_button = QtWidgets.QPushButton("Connect", self)
        self.disconnect_button = QtWidgets.QPushButton("Disconnect", self)

        # Initially, disconnect button is disabled
        self.disconnect_button.setEnabled(False)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.symbol_input)
        layout.addWidget(self.connect_button)
        layout.addWidget(self.disconnect_button)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.connect_button.clicked.connect(self.connect)
        self.disconnect_button.clicked.connect(self.disconnect)

    def connect(self):
        symbol = self.symbol_input.text()
        if symbol != '':
            if hasattr(self, 'main_window'):
                self.main_window.close()

            self.main_window = MainWindow(symbol.upper())
            self.main_window.show()

            # Enable disconnect button and disable connect button
            self.disconnect_button.setEnabled(True)
            self.connect_button.setEnabled(False)

    def disconnect(self):
        self.main_window.close() 
        self.main_window.close_websocket()

        # Delete the main_window attribute
        del self.main_window

        # Perform manual garbage collection
        gc.collect()

        # Enable connect button and disable disconnect button
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

class Scatter3DWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Scatter3DWindow, self).__init__(*args, **kwargs)
        self.resize(800, 600) 

        self.glview = gl.GLViewWidget()
        self.setCentralWidget(self.glview)

        self.glview.setCameraPosition(distance=10)

        # create a grid item
        grid = gl.GLGridItem()
        grid.scale(0.2, 0.4, 0.1)
        self.glview.addItem(grid)
        
        self.scatterItems = deque(maxlen=180)

    def setup_plot(self, trade_slice, b_quantity, a_quantity, tick_size):
        trade_time = trade_slice['time']
        trade_price = trade_slice['price']
        trade_quantity = trade_slice['quantity']
        trade_is_buyer_maker = trade_slice['is_buyer_maker']

        min_price = np.min(trade_price)
        max_price = np.max(trade_price)

        num_buy_trades = np.count_nonzero(trade_is_buyer_maker)
        num_sell_trades = np.count_nonzero(np.logical_not(trade_is_buyer_maker))
        if num_buy_trades > 0 and num_sell_trades > 0:
            normalized_trade_price = (trade_price - min_price) / (max_price - min_price)
            #logging.info(f"normalized trade price: {normalized_trade_price[-1]}")
            
            book_skew, book_imb = self.calculate_liquidity(b_quantity, a_quantity)
            volume_skew, volume_imb = self.calculate_trade_liquidity(trade_slice)

            sp1_coords = np.array([[normalized_trade_price[-1]], [book_skew], [book_imb]])
            sp1 = gl.GLScatterPlotItem(pos=sp1_coords.T, size=0.05, pxMode=False, color=(0.3, 1, 0.9, 0.6))
            self.glview.addItem(sp1)

            sp2_coords = np.array([[normalized_trade_price[-1]], [volume_skew], [volume_imb]])
            sp2 = gl.GLScatterPlotItem(pos=sp2_coords.T, size=0.05, pxMode=False, color=(1, 0.8, 0.5, 0.6))
            self.glview.addItem(sp2)

            self.scatterItems.append((sp1, sp2))

        if len(self.scatterItems) == self.scatterItems.maxlen:
            oldest_scatter_pair = self.scatterItems.popleft()  # Remove oldest pair from deque
            self.glview.removeItem(oldest_scatter_pair[0])  # Remove oldest sp1 from glview
            self.glview.removeItem(oldest_scatter_pair[1])  # Remove oldest sp2 from glview
        
        logging.info(len(self.scatterItems))

    def calculate_liquidity(self, b_quantity, a_quantity):
        try: 
            bid_volumes = np.sum(b_quantity)
            ask_volumes = np.sum(a_quantity)
            book_skew = np.log(bid_volumes) - np.log(ask_volumes)
            book_imb = (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes)
        except RuntimeWarning as e:
            logging.info(e, bid_volumes, ask_volumes)
            raise e

        return book_skew, book_imb

    def calculate_trade_liquidity(self, trade_slice):
        try:
            buy_trades = [x[2] for x in trade_slice if x[3] == False]
            sell_trades = [x[2] for x in trade_slice if x[3] == True]

            buy_volumes = np.sum(buy_trades)
            sell_volumes = np.sum(sell_trades)
            num_buy_trades = len(buy_trades)
            num_sell_trades = len(sell_trades)
            
            volume_skew = np.log((buy_volumes + 1e-9)/(num_buy_trades + 1e-9)) - np.log((sell_volumes + 1e-9)/(num_sell_trades + 1e-9))
            volume_imb = (buy_volumes - sell_volumes) / (buy_volumes + sell_volumes)
        except RuntimeWarning as e:
            logging.info(e, buy_volumes, sell_volumes)
            raise e

        return volume_skew, volume_imb

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, symbol, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        logging.basicConfig(filename='/Users/berkes/failing_again/app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.symbol = symbol

        self.manager = QNetworkAccessManager()
        
        self.request = QNetworkRequest(QtCore.QUrl("https://fapi.binance.com/fapi/v1/exchangeInfo"))
        self.manager.finished.connect(self.handle_response)
        self.manager.get(self.request)
        
        self.request2 = QNetworkRequest(QtCore.QUrl(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={self.symbol}"))
        self.manager.finished.connect(self.handle_response2)
        self.manager.get(self.request2)
        
        self.unknown_factor = None

        self.label = QLabel(self)
        self.label.setText("Loading...")
        
        self.header_layout = QVBoxLayout()

        #plot init#
        self.graphWidget = pg.PlotWidget() 
        self.lineGraphWidget = pg.PlotWidget()
        self.volumeGraphWidget = pg.PlotWidget() 
        self.scatterGraphWidget = pg.PlotWidget()

        layout = QtWidgets.QGridLayout()

        layout.addLayout(self.header_layout, 0, 0, 1, 2)  # Add the header layout at the top
        layout.addWidget(self.lineGraphWidget, 1, 0)  
        layout.addWidget(self.graphWidget, 1, 1)
        layout.addWidget(self.volumeGraphWidget, 2, 0) 
        layout.addWidget(self.scatterGraphWidget, 2, 1)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.lineGraphWidget.setYLink(self.graphWidget)
        self.volumeGraphWidget.setXLink(self.lineGraphWidget)

        self.plot_item = pg.PlotDataItem(x=[], y=[])

        #array init#
        self.b_price = np.array([])
        self.b_quantity = np.array([])
        self.a_price = np.array([])
        self.a_quantity = np.array([])
        
        #self.book_threshold = 1000000 / self.b_price[0]

        self.spread_history = np.array([])

        self.trade_dtype = np.dtype([('time', 'int64'), ('price', 'float64'), ('quantity', 'float64'), ('is_buyer_maker', 'bool')])
        self.trade_array = np.empty(50000, dtype=self.trade_dtype)

        self.trade_index = 0
        self.trades_buffer = []

        self.buffer_timer = QtCore.QTimer()
        self.buffer_timer.setInterval(450)
        self.buffer_timer.timeout.connect(self.process_buffer)
        self.buffer_timer.start()

        self.websocket = QtWebSockets.QWebSocket()
        logging.info(f"websocket state: {self.websocket.state()}")
        self.websocket.connected.connect(self.on_connected)
        self.websocket.open(QtCore.QUrl(f"wss://fstream.binance.com/stream?streams={self.symbol.lower()}@depth20@100ms/{self.symbol.lower()}@aggTrade"))

        self.resize(1440, 900)
        #self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        #self.scatter3d_window = Scatter3DWindow()
        #self.scatter3d_window.show()

        self.table = Table(50, 5)
        
    def on_connected(self):
        self.websocket.textMessageReceived.connect(self.on_text_message_received)
        logging.info(f"websocket state: {self.websocket.state()}")

    def close_websocket(self):
        self.websocket.close()
        logging.info(f"websocket state: {self.websocket.state()}")

    def on_text_message_received(self, message):
        data = json.loads(message)
        stream_type = data['stream'].split('@')[1]
        
        if stream_type == 'depth20':            
            self.b_price = np.array([float(x[0]) for x in data["data"]["b"]])
            self.b_quantity = np.array([float(x[1]) for x in data["data"]["b"]])
            self.a_price = np.array([float(x[0]) for x in data["data"]["a"]])
            self.a_quantity = np.array([float(x[1]) for x in data["data"]["a"]])

            update_time = data['data']['E']

            self.update_plot_data()

            spread_data = np.column_stack((update_time, self.b_price[0], self.a_price[0]))
            self.spread_history = np.vstack((self.spread_history, spread_data)) if self.spread_history.size else spread_data

            self.update_spread_history()

        elif stream_type == 'aggTrade':
            trade_record = (data['data']['T'], float(data['data']['p']), float(data['data']['q']), data['data']['m'])
            self.trades_buffer.append(trade_record)

    def process_buffer(self):
        if not self.trades_buffer:
            return
        
        trades_array = np.array(self.trades_buffer, dtype=self.trade_dtype)

        n_trades = len(trades_array)

        self.trade_array = np.roll(self.trade_array, -n_trades)

        np.copyto(self.trade_array[-n_trades:], trades_array)

        #logging.info((len(trades_array), len(self.trades_buffer)))
        self.update_line_plot_data()

        self.trades_buffer.clear()
    
    ### PLOTTING ###
    def update_plot_data(self):
       ## updates graphWidget ## 
        if str(self.websocket.state()) != "SocketState.ConnectedState":
            logging.info(f"websocket state: {self.websocket.state()}, skipping update_plot_data")
            return

        self.graphWidget.clear()
        
        max_value = max(np.max(self.b_quantity), np.max(self.a_quantity))
        #min_value = min(np.min(self.b_quantity), np.min(self.a_quantity))

        if self.unknown_factor is not None:
            self.graphWidget.getPlotItem().setXRange(0, self.unknown_factor if max_value < self.unknown_factor else max_value*1.1)
        
        bid_bars = pg.BarGraphItem(x0=0, y=self.b_price, height=self.tick_size, width=self.b_quantity, brush='g')
        self.graphWidget.addItem(bid_bars)
        
        ask_bars = pg.BarGraphItem(x0=0, y=self.a_price, height=self.tick_size, width=self.a_quantity, brush='r')
        self.graphWidget.addItem(ask_bars)

        self.table.setRowCount(len(self.b_price))
        self.table.update(self.b_price, self.b_quantity, self.a_price, self.a_quantity, self.trades_buffer)
        
    def update_spread_history(self):
        ## updates lineGraphWidget ##
        if str(self.websocket.state()) != "SocketState.ConnectedState":
            logging.info(f"websocket state: {self.websocket.state()}, skipping update_spread_history")
            return

        if hasattr(self, 'spread_history_plot_items'):
            for item in self.spread_history_plot_items:
                self.lineGraphWidget.removeItem(item)
        
        one_minute_ago_index = max(0, len(self.spread_history) - 550)
        spread_history_last_minute = self.spread_history[one_minute_ago_index:]

        try:
            plot_item1 = pg.PlotDataItem(x=spread_history_last_minute[:, 0], y=spread_history_last_minute[:, 1], pen=pg.mkPen(color=(0, 255, 0, 125), width=2)) 
            plot_item2 = pg.PlotDataItem(x=spread_history_last_minute[:, 0], y=spread_history_last_minute[:, 2], pen=pg.mkPen(color=(255, 0, 0, 125), width=2)) 
            
            self.lineGraphWidget.addItem(plot_item1)
            self.lineGraphWidget.addItem(plot_item2)
            
            self.spread_history_plot_items = [plot_item1, plot_item2]
        except IndexError as e:
            logging.info(f"spread history shape: {self.spread_history.shape}, {self.spread_history}")
            logging.info(f"last minute spread history shape: {self.spread_history_last_minute.shape}, {self.spread_history_last_minute}")

    def update_line_plot_data(self):
        ## updates lineGraphWidget and volumeGraphWidget ##
        if str(self.websocket.state()) != "SocketState.ConnectedState":
            logging.info(f"websocket state: {self.websocket.state()}, skipping update_line_plot_data")
            return

        if hasattr(self, 'scatter_plot_item'):
            self.lineGraphWidget.removeItem(self.scatter_plot_item)
            self.volumeGraphWidget.removeItem(self.volume_graph_item)
        if hasattr(self, 'scatter_plot_item2'): 
            self.scatterGraphWidget.removeItem(self.scatter_plot_item2) 

        current_time = self.trade_array['time'][-1]        
        
        index = np.searchsorted(self.trade_array['time'], current_time - 70 * 1000)
        
        trade_slice = self.trade_array[index:]
        #logging.info(f"trade slice: {trade_slice}")
        
        trade_time = trade_slice['time']
        trade_price = trade_slice['price']
        trade_quantity = trade_slice['quantity']
        trade_is_buyer_maker = trade_slice['is_buyer_maker']

        max_quantity = np.max(trade_quantity)
        min_quantity = np.min(trade_quantity)
        
        #line plot#
        brush = np.where(trade_is_buyer_maker, 'r', 'g')      
        size = np.array([self.scale_size_fixed(price, quantity) for price, quantity in zip(trade_price, trade_quantity)])
        scatter = pg.ScatterPlotItem(x=trade_time, y=trade_price, brush=brush, size=size)

        self.scatter_plot_item = scatter
        self.lineGraphWidget.addItem(scatter)
        
        volume_height = np.where(trade_is_buyer_maker, -trade_quantity, trade_quantity) 
        volume_bars = pg.BarGraphItem(x=trade_time, height=volume_height, width=0.6)
        
        self.volume_graph_item = volume_bars     
        self.volumeGraphWidget.setYRange(-max_quantity, max_quantity)                       
        self.volumeGraphWidget.addItem(volume_bars)
        
        #scatter plot#
        num_buy_trades = np.count_nonzero(trade_is_buyer_maker)
        num_sell_trades = np.count_nonzero(~trade_is_buyer_maker) # ~ is the logical NOT operator   
        if num_buy_trades > 0:
            bins = np.arange(0, 500, min_quantity) 
            buy_trades_per_bin = np.histogram(trade_quantity[~trade_is_buyer_maker], bins=bins)[0]
            sell_trades_per_bin = -(np.histogram(trade_quantity[trade_is_buyer_maker], bins=bins)[0])

            y_buys_mask = buy_trades_per_bin > 0
            y_sells_mask = sell_trades_per_bin < 0

            scatter2 = pg.ScatterPlotItem() 

            scatter2.addPoints(x=bins[:-1][y_buys_mask], y=buy_trades_per_bin[y_buys_mask], symbol='o', brush='g')
            scatter2.addPoints(x=bins[:-1][y_sells_mask], y=sell_trades_per_bin[y_sells_mask], symbol='o', brush='r') 
            
            self.scatter_plot_item2 = scatter2
            self.scatterGraphWidget.addItem(scatter2)

            #self.scatter3d_window.setup_plot(trade_slice, self.b_quantity, self.a_quantity, self.tick_size)
    
    ### UTILS ###
    def handle_response(self, reply):
        if reply.request() == self.request:
            exchange_info = json.loads(bytes(reply.readAll()))

            print(f"Getting tick size for {self.symbol}...")
            symbol_info = [s for s in exchange_info['symbols'] if s['symbol'] == self.symbol]
            if symbol_info:
                self.tick_size = float(symbol_info[0]['filters'][0]['tickSize'])
            else:
                self.tick_size = None

            self.label.setText(f"Tick size: {self.tick_size}")
            logging.info(f"Tick size: {self.tick_size}")

    def handle_response2(self, reply):
        if reply.request() == self.request2:
            premium_index = json.loads(bytes(reply.readAll()))

            self.unknown_factor = 1000000 / float(premium_index['indexPrice'])
            logging.info(f"Premium index: {premium_index}\nUnknown factor: {self.unknown_factor}")

    def scale_size_fixed(self, price, quantity):
        try:
            quoteQty = price * quantity
            thresholds = [500, 2500, 7500, 20000, 45000, 75000, 105000, 170000, 280000, 420000, 700000, 1150000]
            sizes = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
            
            for i, threshold in enumerate(thresholds):
                if quoteQty <= threshold:
                    return sizes[i]
            return sizes[-1]
        except RuntimeWarning as e:
            logging.info(e, price, quantity)
            raise e

class Table(QTableWidget):
    def __init__(self, row, column):
        super().__init__()
        self.setRowCount(row)
        self.setColumnCount(column)
        self.setHorizontalHeaderLabels(['Bid Quantity', 'Sells', 'Price', 'Buys', 'Ask Quantity'])
        self.show()

        self.buys = {}
        self.sells = {}
        
    def update(self, b_price, b_quantity, a_price, a_quantity, trades_buffer):
        some_value = 10

        b_price = np.flip(b_price)
        b_quantity = np.flip(b_quantity)

        prices = np.concatenate((b_price, a_price))
        bid_quantities = np.concatenate((b_quantity, np.zeros_like(a_quantity)))
        ask_quantities = np.concatenate((np.zeros_like(b_quantity), a_quantity))

        sort_indices = np.argsort(prices)[::-1]
        prices = prices[sort_indices]
        bid_quantities = bid_quantities[sort_indices]
        ask_quantities = ask_quantities[sort_indices]

        self.setRowCount(len(prices))

        #print(trades_buffer)

        for i in range(len(prices)):
            for j in range(5):  # Change this to 5 because we have 5 columns now
                if j == 0:
                    item = QTableWidgetItem(str(bid_quantities[i]))
                    if bid_quantities[i] > some_value:  
                        item.setBackground(QColor(0, 255, 0))  
                    self.setItem(i, j, item)
                elif j == 2:  # Change this to 2 because 'Price' is the third column
                    self.setItem(i, j, QTableWidgetItem(str(prices[i])))
                elif j == 4:  # Change this to 4 because 'Ask Quantity' is the fifth column
                    item = QTableWidgetItem(str(ask_quantities[i]))
                    if ask_quantities[i] > some_value: 
                        item.setBackground(QColor(255, 0, 0))  
                    self.setItem(i, j, item)

        if len(trades_buffer) > 0:
            for trade in trades_buffer:
                trade_time, trade_price, trade_quantity, is_sell = trade
                if not is_sell:
                    if trade_price in self.sells:
                        self.sells[trade_price] += trade_quantity
                    else:
                        self.sells[trade_price] = trade_quantity
                else:
                    if trade_price in self.buys:
                        self.buys[trade_price] += trade_quantity
                    else:
                        self.buys[trade_price] = trade_quantity

        # Update 'Buys' and 'Sells' columns
        for i in range(len(prices)):
            if prices[i] in self.buys:
                self.setItem(i, 1, QTableWidgetItem(str(round(self.buys[prices[i]], 4))))  # 'Buys' is the second column
            if prices[i] in self.sells:
                self.setItem(i, 3, QTableWidgetItem(str(round(self.sells[prices[i]], 4))))  # 'Sells' is the fourth column

def main():
    app = QtWidgets.QApplication(sys.argv)
    connector = ConnectorWindow()
    connector.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
