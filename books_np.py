import asyncio, time, os, json
import websockets, requests, bisect
import numpy as np

order_book = {'bids': [], 'asks': []}

async def update_order_book(side, changes):
    for change in changes:
        price, quantity = map(float, change)
        # Find the index of any existing bid/ask with the same price
        index = next((i for i, item in enumerate(order_book[side]) if item[0] == price), None)
        if quantity == 0:
            if index is not None:
                # Remove the bid/ask if it exists
                order_book[side].pop(index)
        else:
            if index is not None:
                # Update the quantity of the existing bid/ask
                order_book[side][index] = (price, quantity)
            else:
                # Add a new bid/ask
                order_book[side].append((price, quantity))
    order_book[side].sort(key=lambda x: x[0], reverse=(side == 'bids'))
    order_book[side] = order_book[side][:500]

async def manage_order_book():
    uri = "wss://fstream.binance.com/stream?streams=btcusdt@depth@100ms"

    async with websockets.connect(uri) as websocket:
        response = requests.get("https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500")
        data = response.json()
        lastUpdateId = data['lastUpdateId']

        await asyncio.gather(update_order_book('bids', data['bids']), update_order_book('asks', data['asks']))

        while True:
            event = await websocket.recv()
            event_data = json.loads(event)

            stream = event_data['data']

            final_id = stream['u']
            first_id = stream['U']
            previous_final_id = stream['pu']

            if final_id < lastUpdateId:
                continue

            if first_id <= lastUpdateId and final_id >= lastUpdateId:
                await asyncio.gather(update_order_book('bids', stream['b']), update_order_book('asks', stream['a']))
                lastUpdateId = final_id
                continue

            if previous_final_id != lastUpdateId:
                response = requests.get("https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=500")
                data = response.json()

                await asyncio.gather(update_order_book('bids', data['bids']), update_order_book('asks', data['asks']))
                lastUpdateId = data['lastUpdateId']

            await asyncio.gather(update_order_book('bids', stream['b']), update_order_book('asks', stream['a']))                
            lastUpdateId = final_id

            asyncio.create_task(calculate_liquidity(order_book))

async def calculate_liquidity(order_book):
    #print("\n", order_book['asks'][25], order_book['bids'][25])
    print("Sum of asks: ", round(np.sum([x[1] for x in order_book['asks']])), "Sum of bids: ", round(np.sum([x[1] for x in order_book['bids']])))

asyncio.run(manage_order_book())
