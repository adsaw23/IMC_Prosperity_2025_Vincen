import jsonpickle
import numpy as np
from typing import List,Dict, Any
import json

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.KELP_prices = []
        self.KELP_vwap = []
        self.squid_prices = []
        

    

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        two_third_limit = (2/3) * limit  
        
        #Calulate Best ask above fair and Best bid below fair (Market making)
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])


        if len(order_depth.sell_orders) != 0:
            # print("Checking Sell Orders for buying")
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                # print("Ask Value found")
                quantity = min(best_ask_amount, limit - position) # max amt to buy 
                # logger.print(quantity)
                if quantity > 0:
                    logger.print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    logger.print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        if position > two_third_limit:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= fair_value - 1:
                    available_amount = order_depth.buy_orders[bid_price]
                    sell_quantity = min(available_amount, position - (position // 2))
                    
                    if sell_quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", bid_price, -sell_quantity))
                        sell_order_volume += sell_quantity
                    break
        if position < -two_third_limit:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= fair_value + 1:
                    available_amount = -1 * order_depth.sell_orders[ask_price]
                    buy_quantity = min(available_amount, abs(position - (position // 2)))
                    
                    if buy_quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", ask_price, buy_quantity))
                        buy_order_volume += buy_quantity
                    break

        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            # logger.print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order("RAINFOREST_RESIN", int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            # logger.print("Sell_Order", str(sell_quantity) + "x", baaf-1)
            orders.append(Order("RAINFOREST_RESIN", int(baaf - 1), -sell_quantity))  # Sell order

        return orders
    def KELP_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        two_third_limit = (2/3) * limit  
       

        fair_value= 2035
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0: 

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            # Filter for decent volume levels to compute more reliable mid-price
            mmmid_price = (best_ask + best_bid) / 2  
            self.KELP_prices.append(mmmid_price)

            if len(self.KELP_prices) > timespan:
                self.KELP_prices.pop(0)
        
            fair_value = np.mean(self.KELP_prices)
        
          #Calulate Best ask above fair and Best bid below fair (Market making)
        
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

        if len(order_depth.sell_orders) != 0:
            # logger.print("Checking Sell Orders for buying")
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                # logger.print("Ask Value found")
                quantity = min(best_ask_amount, limit - position) # max amt to buy 
                # logger.logger.print(quantity)
                if quantity > 0:
                    logger.print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order("KELP", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    logger.print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order("KELP", best_bid, -1 * quantity))
                    sell_order_volume += quantity
     

        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            # logger.print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order("KELP", int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            # logger.print("Sell_Order", str(sell_quantity) + "x", baaf-1)
            orders.append(Order("KELP", int(baaf - 1), -sell_quantity))  # Sell order

        return orders
    def squid_orders(self, order_depth: OrderDepth, timespan: int, width: float, squid_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())   
        # Filter for decent volume levels to compute more reliable mid-price
        mmmid_price = (best_ask + best_bid) / 2  
        self.KELP_prices.append(mmmid_price)

        if len(self.KELP_prices) > timespan:
            self.KELP_prices.pop(0)
    
        fair_value = np.mean(self.KELP_prices)

        buy_vol = sum(order_depth.buy_orders.values())  # buy side volume
        sell_vol = sum(abs(v) for v in order_depth.sell_orders.values())  # sell side volume

        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)  # +1e-6 to avoid division by zero
        # Positive imbalance => more buyers => price likely to go up

        # Adjust fair value based on imbalance
        offset = imbalance * width
        fair_value += offset
        if len(order_depth.sell_orders) != 0:
            # print("Checking Sell Orders for buying")
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value and best_ask> fair_value-squid_take_width:
                # print("Ask Value found")
                quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                # logger.print(quantity)
                if quantity > 0:
                    logger.print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order("SQUID_INK", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value and best_bid <fair_value + squid_take_width:
                quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                if quantity > 0:
                    logger.print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        

        return orders

    
    def run(self, state: TradingState) -> Dict[str,List[Order]]:
        result = {}

        resin_acceptance_value = 10000  # Participant should calculate this value
        resin_limit = 50
        
        KELP_position_limit = 50
        KELP_timemspan = 10

        squid_make_width = 3.5
        squid_take_width = 5
        squid_position_limit = 50
        squid_timemspan = 5

          # Ensure traderData is correctly initialized
        if not state.traderData or state.traderData.strip() == "":
            traderData = {"KELP_prices": [], "KELP_vwap": [], "SQUID_inks_prices": []}
        else:
            try:
                traderData = jsonpickle.decode(state.traderData)
                if not isinstance(traderData, dict):  
                    raise ValueError("Decoded traderData is not a dictionary")
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}")
                traderData = {"KELP_prices": [], "KELP_vwap": [], "SQUID_inks_prices" : []}  # Fallback
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"],resin_acceptance_value, resin_position, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders
        
        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            KELP_orders = self.KELP_orders(state.order_depths["KELP"], KELP_timemspan, KELP_position, KELP_position_limit)
            result["KELP"] = KELP_orders
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            squid_orders = self.squid_orders(state.order_depths["SQUID_INK"], squid_timemspan, squid_make_width, squid_take_width, squid_position, squid_position_limit)
            result["SQUID_INK"] = squid_orders


        # traderData = jsonpickle.encode( { "KELP_prices": self.KELP_prices, "KELP_vwap": self.KELP_vwap })
        traderData = jsonpickle.encode({
        "KELP_prices": self.KELP_prices, 
        "KELP_vwap": self.KELP_vwap,
        "SQUID_inks_prices": self.squid_prices

    }, unpicklable=False)  # Ensures it's JSON serializable
        #traderData = "SAMPLE"

        conversions = 1

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    