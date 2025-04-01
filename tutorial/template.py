from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List,Dict
import string
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        self.resin_prices = []


    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        
        #Calulate Best ask above fair and Best bid below fair (Market making)
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, limit - position) # max amt to buy 
                if quantity > 0:
                    print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.manage_position(orders, order_depth, position, limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value)

        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order

        return orders
    
    def manage_position(self, orders: List[Order], order_depth: OrderDepth, position: int, limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: int) -> List[Order]:
        
        new_position = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
       

        buy_quantity = limit - (position + buy_order_volume)
        sell_quantity = limit + (position - sell_order_volume)

        if new_position > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], new_position)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if new_position < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(new_position))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
   

    def run(self, state: TradingState) -> Dict[str,List[Order]]:
        result = {}

        resin_acceptance_value = 10000  # Participant should calculate this value
        resin_limit = 50
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"],resin_acceptance_value, resin_position, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders

       
        traderData = "SAMPLE"

        conversions = 1

        return result, conversions, traderData

    