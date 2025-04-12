from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List,Dict
import string
import jsonpickle
import numpy as np
import math
import pandas as pd


class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.jam_prices = []
        self.djembe_prices = None
        self.crois_prices = []
        self.squid_prices = []
       
        
        
    def market_take(self, order_depth: OrderDepth, orders: List[Order], product:string, fair_value: int, position :int, limit:int, buy_order_volume: int, sell_order_volume:int) -> tuple[List[Order],int,int]:

        if len(order_depth.sell_orders) != 0:
            # print("Checking Sell Orders for buying")
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                # print("Ask Value found")
                quantity = min(best_ask_amount, limit - position) # max amt to buy 
                print(quantity)
                if quantity > 0:
                    print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order(product, best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
        return orders, buy_order_volume, sell_order_volume
    def manage_position(self, order_depth: OrderDepth, orders: List[Order], product:string, fair_value: int, position :int, limit:int,exceed_limit:int, buy_order_volume: int, sell_order_volume:int) -> tuple[List[Order],int,int]:
         
        if position > exceed_limit:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= fair_value - 1:
                    available_amount = order_depth.buy_orders[bid_price]
                    sell_quantity = min(available_amount, position - (position // 2))
                    
                    if sell_quantity > 0:
                        orders.append(Order(product, bid_price, -sell_quantity))
                        sell_order_volume += sell_quantity
                    break
        if position < -exceed_limit:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= fair_value + 1:
                    available_amount = -1 * order_depth.sell_orders[ask_price]
                    buy_quantity = min(available_amount, abs(position - (position // 2)))
                    
                    if buy_quantity > 0:
                        orders.append(Order(product, ask_price, buy_quantity))
                        buy_order_volume += buy_quantity
                    break
        return orders,buy_order_volume, sell_order_volume
    
    def market_make(self, order_depth: OrderDepth, orders: List[Order], product:string, fair_value: int,position :int, limit:int, buy_order_volume: int, sell_order_volume:int) -> tuple[List[Order],int,int]:
         #Calulate Best ask above fair and Best bid below fair (Market making)
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order(product, int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            print("Sell_Order", str(sell_quantity) + "x", baaf-1)
            orders.append(Order(product, int(baaf - 1), -sell_quantity))  # Sell order

        return orders, buy_order_volume, sell_order_volume

    


    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit  
        
        orders, buy_order_volume, sell_order_volume = self.market_take(order_depth,orders,"RAINFOREST_RESIN",fair_value,position, limit ,buy_order_volume, sell_order_volume)
        
        orders, buy_order_volume ,sell_order_volume = self.manage_position(order_depth, orders, "RAINFOREST_RESIN",fair_value,position,limit, exceed_limit,buy_order_volume, sell_order_volume)

        orders, buy_order_volume ,sell_order_volume = self.market_make(order_depth, orders, "RAINFOREST_RESIN",fair_value,position,limit,buy_order_volume, sell_order_volume)

        return orders
    def KELP_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit  
       
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0: 

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            # Filter for decent volume levels to compute more reliable mid-price
            mmmid_price = (best_ask + best_bid) / 2  
            self.kelp_prices.append(mmmid_price)

            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
        if not self.kelp_prices:
            return orders
        
        fair_value = (max(self.kelp_prices)+min(self.kelp_prices))/2 if self.kelp_prices else 0

        orders, buy_order_volume, sell_order_volume = self.market_take(order_depth,orders,"KELP",fair_value,position, limit ,buy_order_volume, sell_order_volume)
        
        # orders, buy_order_volume ,sell_order_volume = self.manage_position(order_depth, orders, "KELP",fair_value,position,limit, exceed_limit,buy_order_volume, sell_order_volume)

        orders, buy_order_volume ,sell_order_volume = self.market_make(order_depth, orders, "KELP",fair_value,position,limit,buy_order_volume, sell_order_volume)

        return orders
    def jam_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit 
        slope = 2.190229824422925
        intercept = -910.9710425043777 

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            mid_price = (best_ask + best_bid) / 2  
            self.jam_prices.append(mid_price)

            if len(self.jam_prices) > timespan:
                self.jam_prices.pop(0)

        if len(self.jam_prices) < 26:
            return orders  # Not enough data for MACD calculation

        # Compute MACD and Signal Line
        prices = pd.Series(self.jam_prices)
        exp1 = prices.ewm(span=12, adjust=False).mean().iloc[-1]
        exp2 = prices.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = exp1 - exp2
        signal = prices.ewm(span=9, adjust=False).mean().iloc[-1]
        
        # Set fair value depending on MACD direction
        fair_value = None
        if macd > signal:
            fair_value = mid_price + 1.5  # We expect price to rise
        elif macd < signal:
            fair_value = mid_price - 1.5  # We expect price to fall

        if fair_value:
            orders, buy_order_volume, sell_order_volume = self.market_take(
                order_depth, orders, "JAMS", fair_value, position, limit, buy_order_volume, sell_order_volume
            )

        return orders
    def djembe_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit 
        slope = 2.190229824422925
        intercept = -910.9710425043777 
       
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0: 

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            # Filter for decent volume levels to compute more reliable mid-price
            mmmid_price = (best_ask + best_bid) / 2  
            self.djembe_prices = mmmid_price

        if self.jam_prices is None:
            return orders
        
        fair_value = np.mean(self.jam_prices)*slope + intercept

        # orders, buy_order_volume, sell_order_volume = self.market_take(order_depth,orders,"DJEMBES",fair_value,position, limit ,buy_order_volume, sell_order_volume)
        
        # orders, buy_order_volume ,sell_order_volume = self.manage_position(order_depth, orders, "KELP",fair_value,position,limit, exceed_limit,buy_order_volume, sell_order_volume)

        orders, buy_order_volume ,sell_order_volume = self.market_make(order_depth, orders, "DJEMBES",fair_value,position,limit,buy_order_volume, sell_order_volume)

        return orders
    def crois_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit  

        mid_price = None
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            mid_price = (best_ask + best_bid) / 2  
            self.crois_prices.append(mid_price)

            if len(self.crois_prices) > timespan:
                self.crois_prices.pop(0)

        if len(self.crois_prices) < 26:
            return orders  # Not enough data for MACD calculation

        # Compute MACD and Signal Line
        prices = pd.Series(self.crois_prices)
        exp1 = prices.ewm(span=12, adjust=False).mean().iloc[-1]
        exp2 = prices.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = exp1 - exp2
        signal = prices.ewm(span=9, adjust=False).mean().iloc[-1]
        
        # Set fair value depending on MACD direction
        fair_value = None
        if macd > signal:
            fair_value = mid_price + 1.5  # We expect price to rise
        elif macd < signal:
            fair_value = mid_price - 1.5  # We expect price to fall

        if fair_value:
            orders, buy_order_volume, sell_order_volume = self.market_take(
                order_depth, orders, "CROISSANTS", fair_value, position, limit, buy_order_volume, sell_order_volume
            )

        return orders
    
    def squid_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit  

        mid_price = None
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            mid_price = (best_ask + best_bid) / 2  
            self.squid_prices.append(mid_price)

            if len(self.squid_prices) > timespan:
                self.squid_prices.pop(0)

        if len(self.squid_prices) < 26:
            return orders  # Not enough data for MACD calculation

        # Compute MACD and Signal Line
        prices = pd.Series(self.crois_prices)
        exp1 = prices.ewm(span=12, adjust=False).mean().iloc[-1]
        exp2 = prices.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = exp1 - exp2
        signal = prices.ewm(span=9, adjust=False).mean().iloc[-1]
        
        # Set fair value depending on MACD direction
        fair_value = None
        if macd > signal:
            fair_value = mid_price + 1.5  # We expect price to rise
        elif macd < signal:
            fair_value = mid_price - 1.5  # We expect price to fall

        if fair_value:
            orders, buy_order_volume, sell_order_volume = self.market_take(
                order_depth, orders, "SQUID_INK", fair_value, position, limit, buy_order_volume, sell_order_volume
            )

        return orders
        
    def run(self, state: TradingState) -> Dict[str,List[Order]]:
        result = {}

        resin_acceptance_value = 10000  # Participant should calculate this value
        resin_limit = 50
        
        KELP_position_limit = 50
        KELP_timemspan = 15

        jam_position_limit = 350
        jam_timespan = 30

        djembe_position_limit = 60

        crois_timespan = 30
        crois_position_limit = 250

        squid_timespan = 30
        squid_position_limit = 50

          # Ensure traderData is correctly initialized
           # Ensure traderData is correctly initialized
        if not state.traderData or state.traderData.strip() == "":
            traderData = {"kelp_prices" :[], "jam_prices" :[], "djembe_prices" : None}
        else:
            try:
                traderData = jsonpickle.decode(state.traderData)
                if not isinstance(traderData, dict):  
                    raise ValueError("Decoded traderData is not a dictionary")
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderData = {"kelp_prices" : [], "jam_prices":[], "djembe_prices": None}  # Fallback
       
        self.kelp_prices =traderData.get("kelp_prices", [])
        self.jam_prices =traderData.get("jam_prices", [])
        self.djembe_prices =traderData.get("djembe_prices", None)

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"],resin_acceptance_value, resin_position, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders
        
        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            KELP_orders = self.KELP_orders(state.order_depths["KELP"], KELP_timemspan, KELP_position, KELP_position_limit)
            result["KELP"] = KELP_orders
        if "JAMS" in state.order_depths:
            jam_position = state.position["JAMS"] if "JAMS" in state.position else 0
            jam_orders = self.jam_orders(state.order_depths["JAMS"], jam_timespan, jam_position, jam_position_limit)
            result["JAMS"] = jam_orders
        if "DJEMBES" in state.order_depths:
            djembe_position = state.position["DJEMBES"] if "DJEMBES" in state.position else 0
            djembe_orders = self.djembe_orders(state.order_depths["DJEMBES"], 5, djembe_position, djembe_position_limit)
            result["DJEMBES"] = djembe_orders
        if "CROISSANTS" in state.order_depths:
            crois_position = state.position["CROISSANTS"] if "CROISSANTS" in state.position else 0
            crois_orders = self.crois_orders(state.order_depths["CROISSANTS"], crois_timespan, crois_position, crois_position_limit)
            result["CROISSANTS"] = crois_orders
        
        # if "SQUID_INK" in state.order_depths:
        #     squid_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
        #     squid_orders = self.squid_orders(state.order_depths["SQUID_INK"], squid_timespan, squid_position, squid_position_limit)
        #     result["SQUID_INK"] = squid_orders
        
        traderData = jsonpickle.encode({
        "kelp_prices": self.kelp_prices,
        "jam_prices" : self.jam_prices,
        "djembe_prices" : self.djembe_prices
    }, unpicklable=False)  # Ensures it's JSON serializable
        #traderData = "SAMPLE"

        conversions = 1

        return result, conversions, traderData

    