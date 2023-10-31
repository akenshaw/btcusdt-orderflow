import asyncio, time, os, json, timeit, logging, sys
import websockets, requests, aiohttp, bisect, math
import numpy as np
from decimal import Decimal, getcontext
from collections import deque, defaultdict

from PyQt6 import QtCore, QtWidgets, QtWebSockets
from PyQt6.QtNetwork import QNetworkRequest, QNetworkAccessManager
import pyqtgraph as pg
from PyQt6.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel, QFont
from PyQt6.QtWidgets import QFrame, QTableWidgetItem, QTableWidget, QTableView, QAbstractItemView, QLabel, QVBoxLayout
import pyqtgraph.opengl as gl

logging.basicConfig(filename='/Users/berkes/regress/fa_oct.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
url_depth_ss = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500"

async def calculate_liquidity(order_book):
    bid_volumes = round(np.sum([x[1] for x in order_book['bids']]))
    ask_volumes = round(np.sum([x[1] for x in order_book['asks']]))
    skew = round(np.log(bid_volumes) - np.log(ask_volumes), 3)
    imbalance = round((bid_volumes - ask_volumes) / (bid_volumes + ask_volumes), 3)

    price_range = ((order_book['asks'][-1][0] - order_book['bids'][-1][0]) / order_book['bids'][0][0])*100
    
    print("\nPrice range: ", round(price_range, 3), "%")
    print("Sum of bids: ", round(np.sum([x[1] for x in order_book['bids']])), "Sum of asks: ", round(np.sum([x[1] for x in order_book['asks']])))
    print("Skew: ", skew, "\nImb: ", imbalance)
   
    #print(len(order_book['bids']), len(order_book['asks']))
    
    #print("\n", order_book['bids'][:5], order_book['asks'][:5])

async def get_mark_price():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT") as response:
            data = await response.json()
            mark_price = float(data['markPrice'])
            funding_rate = round(float(data['lastFundingRate'])*100, 4)
            
            return mark_price, funding_rate

async def get_exchange_info():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://fapi.binance.com/fapi/v1/exchangeInfo") as response:
            data = await response.json()
            print(f"Getting tick size for BTCUSDT...")
            symbol_info = [s for s in data['symbols'] if s['symbol'] == "BTCUSDT"]
            if symbol_info:
                tick_size = float(symbol_info[0]['filters'][0]['tickSize'])
            else:
                tick_size = None

            return tick_size

async def get_depth_snapshot(session):
    async with session.get(url_depth_ss) as response:
        data = await response.json()
        return data

async def update_order_book(bids, asks, new_bids, new_asks):
    best_bid_price = bids[0][0]

    new_bids = new_bids[new_bids['price'] >= best_bid_price*0.999]
    new_asks = new_asks[new_asks['price'] <= best_bid_price*1.001]

    # Update bids
    _, idx_order, idx_new = np.intersect1d(bids['price'], new_bids['price'], return_indices=True)
    valid_idx_new = idx_new[idx_new < len(new_bids)]
    valid_idx_order = idx_order[idx_new < len(new_bids)]
    bids['quantity'][valid_idx_order] = new_bids['quantity'][valid_idx_new]

    # Add new bid price levels
    new_price_levels = np.setdiff1d(new_bids['price'], bids['price'])
    bids = np.concatenate((bids, new_bids[np.isin(new_bids['price'], new_price_levels)]))

    # Remove price levels with quantity 0 in bids
    bids = bids[bids['quantity'] != 0]

    # Sort bids by price in descending order
    bids = np.sort(bids, order=['price'])[::-1]

    # Update asks
    _, idx_order, idx_new = np.intersect1d(asks['price'], new_asks['price'], return_indices=True)
    valid_idx_new = idx_new[idx_new < len(new_asks)]
    valid_idx_order = idx_order[idx_new < len(new_asks)]
    asks['quantity'][valid_idx_order] = new_asks['quantity'][valid_idx_new]

    # Add new ask price levels
    new_price_levels = np.setdiff1d(new_asks['price'], asks['price'])
    asks = np.concatenate((asks, new_asks[np.isin(new_asks['price'], new_price_levels)]))

    # Remove price levels with quantity 0 in asks
    asks = asks[asks['quantity'] != 0]

    # Sort asks by price in ascending order
    asks = np.sort(asks, order=['price'])

    return bids, asks

class OrderBook:
    def __init__(self, bids, asks):
        self.order_book = self.initialize_order_book(bids, asks)

    def initialize_order_book(self, bids, asks):
        bids_array = np.array([tuple(map(float, bid)) for bid in bids], dtype=np.dtype([('price', float), ('quantity', float)]))
        asks_array = np.array([tuple(map(float, ask)) for ask in asks], dtype=np.dtype([('price', float), ('quantity', float)]))

        return {'bids': bids_array, 'asks': asks_array}

    async def refresh_order_book(self, session):
        while True:
            data = await get_depth_snapshot(session)
            data_b_filtered = [item for item in data['bids'] if float(item[0]) >= float(data['bids'][0][0])*0.999]
            data_a_filtered = [item for item in data['asks'] if float(item[0]) <= float(data['bids'][0][0])*1.001]
            
            self.order_book = self.initialize_order_book(data_b_filtered, data_a_filtered)       
            await asyncio.sleep(3)  

    async def update_order_book(self, new_bids, new_asks):
        #start_time = timeit.default_timer()
        
        new_bids = np.array([tuple(map(float, bid)) for bid in new_bids], dtype=np.dtype([('price', float), ('quantity', float)]))
        new_asks = np.array([tuple(map(float, ask)) for ask in new_asks], dtype=np.dtype([('price', float), ('quantity', float)]))
        self.order_book['bids'], self.order_book['asks'] = await update_order_book(
            self.order_book['bids'], self.order_book['asks'], new_bids, new_asks)   

        #elapsed = timeit.default_timer() - start_time
        #logging.info(f"Elapsed time for 'update_order_book': {elapsed}, {len(self.order_book['bids'])} bids, {len(self.order_book['asks'])} asks.") 

class Trades:
    def __init__(self, buffer_time):
        self.buffer_time = buffer_time * 60 * 1000
        self.trades_buffer = deque()

    async def remove_old_trades(self):
        while True:
            #start_time = timeit.default_timer()

            while self.trades_buffer and self.trades_buffer[0]['T'] < time.time() * 1000 - self.buffer_time:
                self.trades_buffer.popleft()
                if self.returns:  
                    self.returns.popleft()

            #elapsed = timeit.default_timer() - start_time
            #logging.info(f"Elapsed time for remove_old_trades': {elapsed}, {len(self.trades_buffer)} trades in buffer.")

            await asyncio.sleep(2) 

async def ws_handler(symbol, data_signal):
     async with aiohttp.ClientSession() as session:            
        async with session.get(url_depth_ss) as response:
            uri = f"wss://fstream.binance.com/stream?streams={symbol}@depth@100ms/{symbol}@aggTrade"

            tick_size = await get_exchange_info()
            last_mark_price, funding_rate = await get_mark_price()

            data = await response.json()
            lastUpdateId = data['lastUpdateId']

            order_book = OrderBook(data['bids'], data['asks'])
            aggr_trades = Trades(0.5)

            asyncio.create_task(aggr_trades.remove_old_trades())
            asyncio.create_task(order_book.refresh_order_book(session))
            #asyncio.create_task(aggr_trades.calculate_returns())

            is_first_event = True
            async with websockets.connect(uri) as websocket:
                while True:       
                    event_data = json.loads(await websocket.recv())
                    
                    if event_data['stream'] == "btcusdt@depth@100ms":
                        try:
                            stream = event_data['data']

                            final_id = stream['u']
                            first_id = stream['U']
                            previous_final_id = stream['pu']

                            if final_id < lastUpdateId:
                                continue
                                
                            if is_first_event:
                                if first_id <= lastUpdateId and final_id >= lastUpdateId:
                                    print("\nFirst processed event succeed.") 
                                    is_first_event = False
                                else:
                                    print("\nOut of sync at the first event, reinitializing order book...")
                                    data = await get_depth_snapshot(session)
                                    await order_book.update_order_book(data['bids'], data['asks'])
                                    lastUpdateId = data['lastUpdateId']
                                    continue

                            elif previous_final_id != lastUpdateId:
                                print("\nOut of sync, reinitializing order book...")
                                data = await get_depth_snapshot(session)
                                await order_book.update_order_book(data['bids'], data['asks'])
                                lastUpdateId = data['lastUpdateId']
                                continue

                            asyncio.create_task(order_book.update_order_book(stream['b'], stream['a']))     
                            lastUpdateId = final_id

                            update_time = stream['E']

                            #await calculate_liquidity(order_book.order_book)
                            trades_buffer_emit = aggr_trades.trades_buffer.copy()
                            data_signal.emit(update_time, order_book.order_book['bids'], order_book.order_book['asks'], trades_buffer_emit)
                            aggr_trades.trades_buffer.clear()
            
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            raise e

                    elif event_data['stream'] == "btcusdt@aggTrade":
                        try:
                            trade_record = (event_data['data']['a'] ,event_data['data']['T'], float(event_data['data']['p']), float(event_data['data']['q']), event_data['data']['m'])
                            aggr_trades.trades_buffer.append(trade_record)
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            break

class AsyncThread(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object, object, object, object)

    def __init__(self, loop):
        QtCore.QThread.__init__(self)
        self.loop = loop
    def run(self):
        self.loop.run_until_complete(ws_handler("btcusdt", self.data_signal))

class AggregatorThread(QtCore.QThread):
    aggregated_data_signal = QtCore.pyqtSignal(object, object, object, object, object)

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.bids = None
        self.asks = None
        self.trades_buffer = None

        self.buys = defaultdict(int)
        self.sells = defaultdict(int)

        self.processed_trades = set()

    @QtCore.pyqtSlot(object, object, object, object)
    def update_data(self, update_time, bids, asks, trades_buffer):
        self.bids = bids
        self.asks = asks
        self.trades_buffer = trades_buffer
        self.start()

    def run(self):
        bids, asks, buys, sells, price_bins = self.perform_aggregation(self.bids, self.asks, self.trades_buffer)
        self.aggregated_data_signal.emit(bids, asks, buys, sells, price_bins)

    def perform_aggregation(self, bids, asks, trades_buffer):
        bids_bins = np.arange(round(min(bids['price'])), round(max(bids['price'])) + 2, 1)
        asks_bins = np.arange(round(min(asks['price'])) - 1, round(max(asks['price'])) + 1, 1)

        bids_bin_indices = np.digitize(bids['price'], bids_bins)
        asks_bin_indices = np.digitize(asks['price'], asks_bins)

        bids_binned_quantities = np.bincount(bids_bin_indices, weights=bids['quantity'])
        asks_binned_quantities = np.bincount(asks_bin_indices, weights=asks['quantity'])

        bids_binned = np.array([(price, quantity) for price, quantity in zip(bids_bins, bids_binned_quantities)], dtype=np.dtype([('price', float), ('quantity', float)]))
        asks_binned = np.array([(price, quantity) for price, quantity in zip(asks_bins, asks_binned_quantities)], dtype=np.dtype([('price', float), ('quantity', float)]))

        price_bins = np.concatenate((bids_bins, asks_bins))
        price_bins = np.flip(np.unique(price_bins))

        ### trade_id, trade_time, trade_price, trade_quantity, is_sell = trade ##
        for trade in trades_buffer:          
            trade_id = trade[0]
            if trade_id not in self.processed_trades:
                self.processed_trades.add(trade_id)
                
                rounded_price = round(trade[2] + 0.5) 
                scaled_quantity = int(trade[3] * 1000)          
                if trade[4] == True:
                    self.sells[rounded_price] += scaled_quantity
                else:
                    self.buys[rounded_price] += scaled_quantity
        
        return bids_binned, asks_binned, self.buys, self.sells, price_bins

### DOM Table ###
class ColorfulModel(QStandardItemModel):
    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        value = super().data(index, role)

        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            value = super().data(index, QtCore.Qt.ItemDataRole.EditRole)
            column = index.column()
            if value is not None:
                return self.set_color_gradient(float(value), column)
        
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            value = super().data(index, QtCore.Qt.ItemDataRole.EditRole)
            column = index.column()
            if value is not None:
                if column == 2:
                    return QColor(135, 135, 135)
                if float(value) == 0:
                    return QColor(23, 22, 22, 0)
                
        if role == QtCore.Qt.ItemDataRole.FontRole:
            column = index.column()
            if column == 2:
                font = QFont("Helvetica", 17, QFont.Weight.Bold)
                return font
        
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            column = index.column()
            if column == 2:
                return QtCore.Qt.AlignmentFlag.AlignCenter
        
        return super().data(index, role)

    def set_color_gradient(self, value, column):
        threshold = 30
        color = QColor(255, 255, 255, 0)
        
        if value > 0:
            if column == 0:
                if value > threshold*2:
                    color = QColor(0, 255, 0, 120)
                elif value > threshold:
                    color = QColor(0, 255, 0, 60)
                elif value > threshold/2:
                    color = QColor(0, 255, 0, 20)
                elif value > threshold/5:
                    color = QColor(0, 255, 0, 5)
                
            elif column == 1:
                if value > threshold*2:
                    color = QColor(255, 0, 100, 120)
                elif value > threshold:
                    color = QColor(255, 0, 100, 60)
                elif value > threshold/2:
                    color = QColor(255, 0, 100, 20)
                elif value > threshold/5:
                    color = QColor(255, 0, 100, 5)
                
            elif column == 3:
                if value > threshold*2:
                    color = QColor(0, 255, 200, 120)
                elif value > threshold:
                    color = QColor(0, 255, 200, 60)
                elif value > threshold/2:
                    color = QColor(0, 255, 200, 20)
                elif value > threshold/5:
                    color = QColor(0, 255, 200, 5)
                
            elif column == 4:
                if value > threshold*2:
                    color = QColor(255, 0, 0, 120)
                elif value > threshold:
                    color = QColor(255, 0, 0, 60)
                elif value > threshold/2:
                    color = QColor(255, 0, 0, 20)
                elif value > threshold/5:
                    color = QColor(255, 0, 0, 5)
        return color

class Table(QTableView):
    def __init__(self, row, column):
        super().__init__()

        self.model = ColorfulModel(row, column)
        self.setModel(self.model)
        self.model.setHorizontalHeaderLabels(['Bid Quantity', 'Sells', 'Price', 'Buys', 'Ask Quantity'])

        self.resize(550, 1300)

        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        self.show()

    def update_table(self, bids, asks, buys, sells, price_bins):
        # Add new rows if necessary
        for _ in range(self.model.rowCount(), len(price_bins)):
            for j in range(5):
                self.model.setItem(self.model.rowCount(), j, QStandardItem())

        # Update the data in the table
        for i, price in enumerate(price_bins):
            bid_quantity = next((bid['quantity'] for bid in bids if bid['price'] == price), 0)
            ask_quantity = next((ask['quantity'] for ask in asks if ask['price'] == price), 0)
            buy_quantity = buys.get(price, 0) / 1000
            sell_quantity = sells.get(price, 0) / 1000

            # Update the table items
            for j, quantity in enumerate([bid_quantity, sell_quantity, price, buy_quantity, ask_quantity]):
                item = self.model.item(i, j)
                if item is None:
                    item = QStandardItem()
                    self.model.setItem(i, j, item)
                item.setText(str(round(quantity, 4)))

### Heatmap ###
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, symbol, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
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

        self.unknown_factor = 80
        self.tick_size = 1
        
        self.b_price = np.array([])
        self.b_quantity = np.array([])
        self.a_price = np.array([])
        self.a_quantity = np.array([])

        self.spread_history = np.array([])

        self.trade_dtype = np.dtype([('id', 'int64'), ('time', 'int64'), ('price', 'float64'), ('quantity', 'float64'), ('is_buyer_maker', 'bool')])
        self.trade_array = np.empty(50000, dtype=self.trade_dtype)

        self.resize(1440, 900)
        self.show()
    
    def receive_data(self, update_time, bids, asks, trades_buffer):
        spread_data = np.column_stack((update_time, bids[0][0], asks[0][0]))
        self.spread_history = np.vstack((self.spread_history, spread_data)) if self.spread_history.size else spread_data

        self.b_price = np.array([bid['price'] for bid in bids])
        self.b_quantity = np.array([bid['quantity'] for bid in bids])
        self.a_price = np.array([ask['price'] for ask in asks])
        self.a_quantity = np.array([ask['quantity'] for ask in asks])

        self.update_heatmap()

        if not trades_buffer:
            return
        trades_array = np.array(trades_buffer, dtype=self.trade_dtype)
        n_trades = len(trades_array)
        self.trade_array = np.roll(self.trade_array, -n_trades)
        np.copyto(self.trade_array[-n_trades:], trades_array)

        self.update_scatter_plot()

    def update_heatmap(self):
        self.graphWidget.clear()
        
        max_value = max(np.max(self.b_quantity), np.max(self.a_quantity))

        ### graphWidget
        if self.unknown_factor is not None:
            self.graphWidget.getPlotItem().setXRange(0, self.unknown_factor if max_value < self.unknown_factor else max_value*1.1)

        bid_bars = pg.BarGraphItem(x0=0, y=self.b_price, height=self.tick_size/10, width=self.b_quantity, brush='g')
        self.graphWidget.addItem(bid_bars) 
        ask_bars = pg.BarGraphItem(x0=0, y=self.a_price, height=self.tick_size/10, width=self.a_quantity, brush='r')
        self.graphWidget.addItem(ask_bars)

        ### lineGraphWidget
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
            logging.info(f"last minute spread history shape: {spread_history_last_minute.shape}, {spread_history_last_minute}")

    def update_scatter_plot(self):
        ### volumeGraphWidget
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
        
### Main Loop ###
def main(symbol):
    app = QtWidgets.QApplication(sys.argv)
 
    book_depth = Table(80, 5)
    main_window = MainWindow(symbol)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async_thread = AsyncThread(loop)
    aggregator_thread = AggregatorThread()
    
    async_thread.data_signal.connect(aggregator_thread.update_data)
    async_thread.data_signal.connect(main_window.receive_data)
    aggregator_thread.aggregated_data_signal.connect(book_depth.update_table)

    async_thread.start()
    sys.exit(app.exec())

if __name__ == "__main__":
    main("btcusdt")