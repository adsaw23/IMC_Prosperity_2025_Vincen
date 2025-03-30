from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    def run(self, state: TradingState):
        print("Trader Data: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        traderData = state.traderData if state.traderData else ""
        past_prices = {p: float(v) for p, v in (eval(traderData) if traderData else {}).items()}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2

                past_mid = past_prices.get(product, mid_price)
                acceptable_price = (mid_price + past_mid) / 2  # Smooth fair price

                spread = best_ask - best_bid
                threshold = spread * 0.2  # Adjusted threshold for more responsiveness

                print(f"{product} | Fair Price: {acceptable_price:.2f}, Spread: {spread}")

                # Buy if the best ask is significantly lower than fair value
                if best_ask < acceptable_price - threshold:
                    buy_qty = min(abs(order_depth.sell_orders[best_ask]), 10)  # Controlled execution
                    print(f"BUY {buy_qty}x @ {best_ask}")
                    orders.append(Order(product, best_ask, buy_qty))

                # Sell if the best bid is significantly higher than fair value
                if best_bid > acceptable_price + threshold:
                    sell_qty = min(abs(order_depth.buy_orders[best_bid]), 10)
                    print(f"SELL {sell_qty}x @ {best_bid}")
                    orders.append(Order(product, best_bid, -sell_qty))

                past_prices[product] = mid_price  # Update past prices

            result[product] = orders

        return result, 0, str(past_prices)
