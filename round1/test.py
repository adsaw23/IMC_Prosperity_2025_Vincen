from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List,Dict
import string
import jsonpickle
import numpy as np
import math


class Trader:
    def __init__(self):
        self.kelp_prices = []
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
                print(quantity)
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
            print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order("RAINFOREST_RESIN", int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            print("Sell_Order", str(sell_quantity) + "x", baaf-1)
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
            self.kelp_prices.append(mmmid_price)

            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            fair_value = np.mean(self.kelp_prices)
        
          #Calulate Best ask above fair and Best bid below fair (Market making)
        
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

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
                    orders.append(Order("KELP", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    print("SELL", str(quantity) + "x", best_bid)
                    orders.append(Order("KELP", best_bid, -1 * quantity))
                    sell_order_volume += quantity
     

        buy_quantity = limit - (position + buy_order_volume)
        if buy_quantity > 0:
            print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order("KELP", int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            print("Sell_Order", str(sell_quantity) + "x", baaf-1)
            orders.append(Order("KELP", int(baaf - 1), -sell_quantity))  # Sell order

        return orders
    def squid_orders(self, order_depth: OrderDepth, timespan: int, width: float, squid_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            # Filter for decent volume levels to compute more reliable mid-price
            filtered_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= 15]

            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

           

            # Basic mid_price from market-making quotes
            mid_price = (mm_ask + mm_bid) / 2
            self.squid_prices.append(mid_price)

            # Maintain a fixed timespan window
            if len(self.squid_prices) > timespan:
                self.squid_prices.pop(0)


            # Base fair value as rolling average of mid_price
            fair_value = sum(self.squid_prices) / len(self.squid_prices)

            # --- Order Book Imbalance Calculation ---
            buy_vol = sum(order_depth.buy_orders.values())  # buy side volume
            sell_vol = sum(abs(v) for v in order_depth.sell_orders.values())  # sell side volume

            imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)  # +1e-6 to avoid division by zero
            # Positive imbalance => more buyers => price likely to go up

            # Adjust fair value based on imbalance
            offset = imbalance * width
            fair_value += offset

            if fair_value +squid_take_width <= best_bid <= fair_value + squid_take_width:
            
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        buy_order_volume += quantity
            if  fair_value -squid_take_width <= best_ask <= fair_value :
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
                        sell_order_volume += quantity
    
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            buy_quantity = position_limit - (position + buy_order_volume)
            # if buy_quantity > 0:
            #     orders.append(Order("SQUID_INK",int( bbbf + 1), buy_quantity))  # Buy order

            # sell_quantity = position_limit + (position - sell_order_volume)
            # if sell_quantity > 0:
            #     orders.append(Order("SQUID_INK", int(baaf - 1), -sell_quantity))  # Sell order

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
           # Ensure traderData is correctly initialized
        if not state.traderData or state.traderData.strip() == "":
            traderData = { "SQUID_inks_prices": [],"kelp_prices" :[]}
        else:
            try:
                traderData = jsonpickle.decode(state.traderData)
                if not isinstance(traderData, dict):  
                    raise ValueError("Decoded traderData is not a dictionary")
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderData = {"SQUID_inks_prices" : [],"kelp_prices" : []}  # Fallback
        self.squid_prices = traderData.get("SQUID_inks_prices", [])
        self.kelp_prices =traderData.get("kelp_prices", [])
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            resin_orders = self.resin_orders(state.order_depths["RAINFOREST_RESIN"],resin_acceptance_value, resin_position, resin_limit)
            result["RAINFOREST_RESIN"] = resin_orders
        
        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            KELP_orders = self.KELP_orders(state.order_depths["KELP"], KELP_timemspan, KELP_position, KELP_position_limit)
            result["KELP"] = KELP_orders
        # if "SQUID_INK" in state.order_depths:
        #     squid_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
        #     squid_orders = self.squid_orders(state.order_depths["SQUID_INK"], squid_timemspan, squid_make_width, squid_take_width, squid_position, squid_position_limit)
        #     result["SQUID_INK"] = squid_orders


        # traderData = jsonpickle.encode( { "KELP_prices": self.KELP_prices, "KELP_vwap": self.KELP_vwap })
        traderData = jsonpickle.encode({
        "SQUID_inks_prices": self.squid_prices,
        "kelp_prices": self.kelp_prices
    }, unpicklable=False)  # Ensures it's JSON serializable
        #traderData = "SAMPLE"

        conversions = 1

        return result, conversions, traderData

    