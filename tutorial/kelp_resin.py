from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List,Dict
import string
import jsonpickle
import numpy as np
import math


class Trader:
    def __init__(self):
        self.KELP_prices = []
        self.KELP_vwap = []

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        
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
                if quantity > 0:
                    # print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    # print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            # print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            # print("Sell_Order", str(sell_quantity) + "x", baaf-1)
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order

        return orders
    def KELP_orders(self, order_depth: OrderDepth, timespan:int, width: float, KELP_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.KELP_prices.append(mmmid_price)
            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.KELP_vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.KELP_vwap) > timespan:
                self.KELP_vwap.pop(0)
            
            if len(self.KELP_prices) > timespan:
                self.KELP_prices.pop(0)
        
            fair_value = sum([x["vwap"]*x['vol'] for x in self.KELP_vwap]) / sum([x['vol'] for x in self.KELP_vwap])
            
            #fair_value = mmmid_price

            # take all orders we can
            # for ask in order_depth.sell_orders.keys():
            #     if ask <= fair_value - KELP_take_width:
            #         ask_amount = -1 * order_depth.sell_orders[ask]
            #         if ask_amount <= 20:
            #             quantity = min(ask_amount, position_limit - position)
            #             if quantity > 0:
            #                 orders.append(Order("KELP", ask, quantity))
            #                 buy_order_volume += quantity
            
            # for bid in order_depth.buy_orders.keys():
            #     if bid >= fair_value + KELP_take_width:
            #         bid_amount = order_depth.buy_orders[bid]
            #         if bid_amount <= 20:
            #             quantity = min(bid_amount, position_limit + position)
            #             if quantity > 0:
            #                 orders.append(Order("KELP", bid, -1 * quantity))
            #                 sell_order_volume += quantity

            # only taking best bid/ask
        
            if best_ask <= fair_value - KELP_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + KELP_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity
    
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP",int( bbbf + 1), buy_quantity))  # Buy order

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", int(baaf - 1), -sell_quantity))  # Sell order

        return orders

    
    def run(self, state: TradingState) -> Dict[str,List[Order]]:
        result = {}

        resin_acceptance_value = 10000  # Participant should calculate this value
        resin_limit = 50
        
        KELP_make_width = 3.5
        KELP_take_width = 5
        KELP_position_limit = 50
        KELP_timemspan = 5

          # Ensure traderData is correctly initialized
        if not state.traderData or state.traderData.strip() == "":
            traderData = {"KELP_prices": [], "KELP_vwap": []}
        else:
            try:
                traderData = jsonpickle.decode(state.traderData)
                if not isinstance(traderData, dict):  
                    raise ValueError("Decoded traderData is not a dictionary")
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderData = {"KELP_prices": [], "KELP_vwap": []}  # Fallback
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"],resin_acceptance_value, resin_position, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders
        
        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            KELP_orders = self.KELP_orders(state.order_depths["KELP"], KELP_timemspan, KELP_make_width, KELP_take_width, KELP_position, KELP_position_limit)
            result["KELP"] = KELP_orders


        # traderData = jsonpickle.encode( { "KELP_prices": self.KELP_prices, "KELP_vwap": self.KELP_vwap })
        traderData = jsonpickle.encode({
        "KELP_prices": self.KELP_prices, 
        "KELP_vwap": self.KELP_vwap
    }, unpicklable=False)  # Ensures it's JSON serializable
        #traderData = "SAMPLE"

        conversions = 1

        return result, conversions, traderData

    