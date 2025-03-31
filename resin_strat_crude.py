from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        target_position = 50  # We want to go long +50 below 9998 and short -50 above 10002
        
        for product in state.order_depths:
            if product != "RAINFOREST_RESIN":
                continue
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
            print(f"Mid price for {product}: {mid_price}")
            
            current_position = state.position.get(product, 0)
            print(f"Current position: {current_position}")
            
            if mid_price < 9998 and current_position < target_position:
                # Buy up to +50 position
                buy_amount = target_position - current_position
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                buy_amount = min(buy_amount, -best_ask_amount)  # Can't buy more than available liquidity
                
                if buy_amount > 0:
                    print(f"BUY {buy_amount} at {best_ask}")
                    orders.append(Order(product, best_ask, buy_amount))
            
            elif mid_price > 10002 and current_position > -target_position:
                # Sell down to -50 position
                sell_amount = abs(-target_position - current_position)
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                sell_amount = min(sell_amount, best_bid_amount)  # Can't sell more than available liquidity
                
                if sell_amount > 0:
                    print(f"SELL {sell_amount} at {best_bid}")
                    orders.append(Order(product, best_bid, -sell_amount))
            
            result[product] = orders
        
        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData