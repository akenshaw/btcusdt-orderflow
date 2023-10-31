import asyncio, time, os, json
import websockets, requests, aiohttp, bisect
import numpy as np
from decimal import Decimal, getcontext

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
    async with session.get("https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500") as response:
        data = await response.json()
        return data

class OrderBook:
    def __init__(self, bids, asks):
        self.order_book = self.initialize_order_book(bids, asks)

    def initialize_order_book(self, bids, asks):
        bids_dtype = np.dtype([('price', float), ('quantity', float)])
        asks_dtype = np.dtype([('price', float), ('quantity', float)])

        bids_array = np.array([tuple(map(float, bid)) for bid in bids], dtype=bids_dtype)
        asks_array = np.array([tuple(map(float, ask)) for ask in asks], dtype=asks_dtype)

        return {'bids': bids_array, 'asks': asks_array}

    async def update_order_book(self, new_bids, new_asks):
        order_book = self.order_book

        #best_bid_price = order_book['bids'][0][0]
        #new_bids = [bid for bid in new_bids if float(bid[0]) >= best_bid_price*0.99]
        #new_asks = [ask for ask in new_asks if float(ask[0]) <= best_bid_price*1.01]

        new_bids_array = np.array([tuple(map(float, bid)) for bid in new_bids], dtype=order_book['bids'].dtype)
        new_asks_array = np.array([tuple(map(float, ask)) for ask in new_asks], dtype=order_book['asks'].dtype)

        # Update bids
        _, idx_order, idx_new = np.intersect1d(order_book['bids']['price'], new_bids_array['price'], return_indices=True)
        valid_idx_new = idx_new[idx_new < len(new_bids_array)]
        valid_idx_order = idx_order[idx_new < len(new_bids_array)]
        order_book['bids']['quantity'][valid_idx_order] = new_bids_array['quantity'][valid_idx_new]

        # Add new bid price levels
        new_price_levels = np.setdiff1d(new_bids_array['price'], order_book['bids']['price'])
        quantities = new_bids_array['quantity'][np.isin(new_bids_array['price'], new_price_levels)]
        order_book['bids'] = np.concatenate((order_book['bids'], np.array(list(zip(new_price_levels, quantities)), dtype=order_book['bids'].dtype)))

        # Remove price levels with quantity 0 in bids
        order_book['bids'] = order_book['bids'][order_book['bids']['quantity'] != 0]

        # Sort bids by price in descending order
        order_book['bids'] = np.sort(order_book['bids'], order=['price'])[::-1]

        # Update asks
        _, idx_order, idx_new = np.intersect1d(order_book['asks']['price'], new_asks_array['price'], return_indices=True)
        valid_idx_new = idx_new[idx_new < len(new_asks_array)]
        valid_idx_order = idx_order[idx_new < len(new_asks_array)]
        order_book['asks']['quantity'][valid_idx_order] = new_asks_array['quantity'][valid_idx_new]

        # Add new ask price levels
        new_price_levels = np.setdiff1d(new_asks_array['price'], order_book['asks']['price'])
        quantities = new_asks_array['quantity'][np.isin(new_asks_array['price'], new_price_levels)]
        order_book['asks'] = np.concatenate((order_book['asks'], np.array(list(zip(new_price_levels, quantities)), dtype=order_book['asks'].dtype)))

        # Remove price levels with quantity 0 in asks
        order_book['asks'] = order_book['asks'][order_book['asks']['quantity'] != 0]

        # Sort asks by price in ascending order
        order_book['asks'] = np.sort(order_book['asks'], order=['price'])

        return order_book

    async def refresh_order_book(self, session):
        while True:
            data = await get_depth_snapshot(session)
            data_b_filtered = [item for item in data['bids'] if float(item[0]) >= float(data['bids'][0][0])*0.999]
            data_a_filtered = [item for item in data['asks'] if float(item[0]) <= float(data['bids'][0][0])*1.001]
            self.order_book = self.initialize_order_book(data_b_filtered, data_a_filtered)
            await asyncio.sleep(3)  

async def manage_order_book():
    uri = "wss://fstream.binance.com/stream?streams=btcusdt@depth@100ms"
    queue = asyncio.Queue()

    async def producer():
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    event = await websocket.recv()
                    await queue.put(event)
                except asyncio.CancelledError:
                    print("Producer task cancelled")
                    raise 

    async def consumer():
        async with aiohttp.ClientSession() as session:            
            async with session.get("https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500") as response:
                data = await response.json()
                lastUpdateId = data['lastUpdateId']

                tick_size = await get_exchange_info()
                last_mark_price, funding_rate = await get_mark_price() 

                order_book = OrderBook(data['bids'], data['asks'])
                is_first_event = True

                asyncio.create_task(order_book.refresh_order_book(session))
                while True:
                    try:
                        event = await queue.get()
                        event_data = json.loads(event)

                        if event_data['stream'] == "btcusdt@depth@100ms":
                            stream = event_data['data']

                            final_id = stream['u']
                            first_id = stream['U']
                            previous_final_id = stream['pu']

                            if final_id < lastUpdateId:
                                continue

                            best_bid_price = data['bids'][0][0]

                            stream_b_filtered = [item for item in stream['b'] if float(item[0]) >= float(best_bid_price)*0.999]
                            stream_a_filtered = [item for item in stream['a'] if float(item[0]) <= float(best_bid_price)*1.001]
                             
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

                            await order_book.update_order_book(stream_b_filtered, stream_a_filtered)     
                            lastUpdateId = final_id
                     
                        #await calculate_liquidity(order_book.order_book)
                        
                        #print("\n", order_book.order_book['bids'][-1], order_book.order_book['asks'][-1])
                        #print(len(order_book.order_book['bids']), len(order_book.order_book['asks']))

                    except Exception as e:
                        print(f"An error occurred: {e}")
                        break
      
    await asyncio.gather(producer(), consumer())

asyncio.run(manage_order_book())
