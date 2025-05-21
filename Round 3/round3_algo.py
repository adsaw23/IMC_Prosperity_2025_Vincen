from sqlite3.dbapi2 import Timestamp
import stat
import jsonpickle
import numpy as np
from typing import List,Dict, Any, Tuple
import json
import pandas as pd
import string
from statistics import NormalDist
from math import sqrt, exp, log
from collections import deque

from data_model import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState



class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    VOLCANIC_ROCK= "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500= "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750= "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000= "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250= "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500= "VOLCANIC_ROCK_VOUCHER_10500"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    JAM_DE_SPREAD = "JAM_DE_SPREAD"
    JAM_SQUID_SPREAD = "JAM_SQUID_SPREAD"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "acceptance_value": 10000,
        "limit": 50,
    },
    Product.KELP: {
        "timespan": 15,
        "limit": 50,
    },
    Product.SQUID_INK: {
        "limit": 50,
        "timespan": 15,  
    },
    Product.JAMS: {
        "limit": 350,
        "timespan": 15,
        "buy_order_volume": 0,
        "sell_order_volume":0
    },
    Product.DJEMBES: {
        "limit": 60,
        "timespan": 15,
    },
    Product.VOLCANIC_ROCK: {
        "limit": 400,
        "timespan": 15,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500:{
        "limit": 200,
        "strike":9500,
        "vol_window": 10,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750:{
        "limit": 200,
        "strike":9750,
        "vol_window": 10,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000:{
        "limit": 200,
        "strike":10000,
        "vol_window": 10,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250:{
        "limit": 200,
        "strike":10250,
        "vol_window": 10,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500:{
        "limit": 200,
        "strike":10500,
        "vol_window": 10,
        "zscore_threshold": 2,
    },
    Product.PICNIC_BASKET1: {
        "limit": 60,
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1,
    },
    Product.PICNIC_BASKET2 :{
        "limit": 100,
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
        Product.DJEMBES: 0,
    },
    Product.SPREAD: {
        "default_spread_mean": 191.26,
        "default_spread_std": 24.92,
        "spread_std_window": 40,
        "zscore_threshold": 3,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": -35.37,
        "default_spread_std": 16.91,
        "spread_std_window": 45,
        "zscore_threshold": 3,
        "target_position": 100,
    },
    Product.JAM_DE_SPREAD: {
        "default_spread_mean": 0,
        "default_spread_std": 6.73,
        "spread_std_window": 40,
        "zscore_threshold": 1.5,
    },
   
}



class BlackScholes:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        """Calculate Black-Scholes call price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S *NormalDist().cdf(d1) - K * np.exp(-r*T) * NormalDist().cdf(d2)

    @staticmethod
    def implied_volatility(
        call_price, 
        spot, 
        strike, 
        time_to_expiry, 
        risk_free_rate=0.0,
        max_iterations=200, 
        tolerance=1e-10
    ):
        """
        Calculate implied volatility using bisection method
        
        Parameters:
        call_price (float): Observed call option price
        spot (float): Current underlying price
        strike (float): Strike price
        time_to_expiry (float): Time to expiration in years
        risk_free_rate (float): Annual risk-free rate (default 0)
        max_iterations (int): Max iterations for bisection
        tolerance (float): Acceptable price difference
        
        Returns:
        float: Implied volatility (NaN if not found)
        """
        # Check for arbitrage bounds first
        intrinsic = max(0, spot - strike)
        if call_price < intrinsic:
            return np.nan  # Arbitrage violation
        
        low_vol = 0.001  # 0.1%
        high_vol = 5.0    # 500%
        
        # Initial checks at boundaries
        low_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, low_vol)
        if abs(low_price - call_price) < tolerance:
            return low_vol
            
        high_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, high_vol)
        if abs(high_price - call_price) < tolerance:
            return high_vol
        
        # Bisection search
        for _ in range(max_iterations):
            mid_vol = (low_vol + high_vol) / 2
            mid_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, mid_vol)
            
            if abs(mid_price - call_price) < tolerance:
                return mid_vol
                
            if mid_price < call_price:
                low_vol = mid_vol
            else:
                high_vol = mid_vol
                
        # If not converged, return best estimate
        return (low_vol + high_vol) / 2
    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)
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
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.PICNIC_BASKET1: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
        }
    def market_take(self, order_depth: OrderDepth, orders: List[Order], product:string, fair_value: int, position :int, limit:int, buy_order_volume: int, sell_order_volume:int) -> tuple[List[Order],int,int]:

        if len(order_depth.sell_orders) != 0:
            # logger.print("Checking Sell Orders for buying")
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                # logger.print("Ask Value found")
                quantity = min(best_ask_amount, limit - position) # max amt to buy 
                logger.print(quantity)
                if quantity > 0:
                    logger.print("BUY", str(quantity) + "x", best_ask)
                    orders.append(Order(product, best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, limit + position) # should be the max we can sell 
                if quantity > 0:
                    logger.print("SELL", str(quantity) + "x", best_bid)
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
            logger.print("BUY_Order", str(buy_quantity) + "x", bbbf+1)
            orders.append(Order(product, int(bbbf + 1), buy_quantity))  # Buy order

        sell_quantity = limit + (position - sell_order_volume)
        if sell_quantity > 0:
            logger.print("Sell_Order", str(sell_quantity) + "x", baaf-1)
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
    def kelp_orders(self, order_depth: OrderDepth, timespan: int, position: int, limit: int, kelp_data: Dict[str, Any]) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        exceed_limit = (2/3) * limit  
       
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0: 

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())   
            # Filter for decent volume levels to compute more reliable mid-price
            mmmid_price = (best_ask + best_bid) / 2  
            kelp_data["kelp_prices"].append(mmmid_price)

            if len(kelp_data["kelp_prices"]) > timespan:
                kelp_data["kelp_prices"].pop(0)
        
        if not kelp_data["kelp_prices"]:
            return orders
        
        fair_value = (max(kelp_data["kelp_prices"])+min(kelp_data["kelp_prices"]))/2 if kelp_data["kelp_prices"] else 0

        orders, buy_order_volume, sell_order_volume = self.market_take(order_depth,orders,"KELP",fair_value,position, limit ,buy_order_volume, sell_order_volume)
        
        # orders, buy_order_volume ,sell_order_volume = self.manage_position(order_depth, orders, "KELP",fair_value,position,limit, exceed_limit,buy_order_volume, sell_order_volume)

        orders, buy_order_volume ,sell_order_volume = self.market_make(order_depth, orders, "KELP",fair_value,position,limit,buy_order_volume, sell_order_volume)

        return orders
    def get_swmid(self, order_depth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth], product: Product
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = PARAMS[product][Product.CROISSANTS]
        JAMS_PER_BASKET = PARAMS[product][Product.JAMS]
        DJEMBES_PER_BASKET = PARAMS[product][Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        DJEMBES_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBES_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
            + DJEMBES_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
            + DJEMBES_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            DJEMBES_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                // DJEMBES_PER_BASKET
            ) if DJEMBES_PER_BASKET > 0 else 1e9
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume, DJEMBES_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            DJEMBES_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                // DJEMBES_PER_BASKET
            ) if DJEMBES_PER_BASKET > 0 else 1e9
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume, DJEMBES_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth], product: Product
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths,product
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * PARAMS[product][Product.CROISSANTS],
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * PARAMS[product][Product.JAMS],
            )
            DJEMBES_order = Order(
                Product.DJEMBES, DJEMBES_price, quantity * PARAMS[product][Product.DJEMBES]
            )

            # # Add the component orders to the respective lists
            # component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            # component_orders[Product.JAMS].append(JAMS_order)
            # component_orders[Product.DJEMBES].append(DJEMBES_order)
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.DJEMBES].append(DJEMBES_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        synth_product: Product
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, product)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(product, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(synth_product, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths,product
            )
            aggregate_orders[product] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(product, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(synth_product, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, product
            )
            aggregate_orders[product] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
        spread_product: Product,
        synth_product: Product,
    ):
        if product not in order_depths.keys():
            return None

        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths,product)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[spread_product]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[spread_product]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[spread_product]["default_spread_mean"]
        ) / spread_std

        true_condition= spread>= 50 if product is Product.PICNIC_BASKET1 else spread>=25
        false_condition= spread<= -50 if product is Product.PICNIC_BASKET1 else spread<=-25

        if true_condition:
            if basket_position != -self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    synth_product
                    
                )

        if false_condition:
            if basket_position != self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,synth_product
                )

        spread_data["prev_zscore"] = zscore
        return None
   

    def voucher_strategy(self, state, current_position, vol_data:Dict[str,Any] ):
        """
        Volatility smile-based options trading strategy.
        Fits a quadratic smile at each timestamp and trades mispriced vouchers.
        """
        # Initialize orders dictionary
        orders = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: [],
            Product.VOLCANIC_ROCK_VOUCHER_9750: [],
            Product.VOLCANIC_ROCK_VOUCHER_10000: [],
            Product.VOLCANIC_ROCK_VOUCHER_10250: [],
            Product.VOLCANIC_ROCK_VOUCHER_10500: [],
        }

        # Get current underlying price (mid-market)
        hedge_orders= []
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return orders,hedge_orders
        order_depth = state.order_depths[Product.VOLCANIC_ROCK]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        S = (best_bid + best_ask) / 2  # Mid-market price
   
    

        # Collect moneyness (m = S/K - 1) and implied volatilities for all strikes
        strikes = [9500, 9750, 10000, 10250, 10500]
        m_list, iv_list = [], []
        total_delta=0

        for strike in strikes:
            product = getattr(Product, f"VOLCANIC_ROCK_VOUCHER_{strike}")
            if product in state.order_depths:
                # Get voucher mid-market price (premium)
                voucher_depth = state.order_depths[product]
                if not voucher_depth.buy_orders or not voucher_depth.sell_orders:
                    return orders, hedge_orders
                voucher_bid = max(voucher_depth.buy_orders.keys())
                voucher_ask = min(voucher_depth.sell_orders.keys())
                premium = (voucher_bid + voucher_ask) / 2

                # Calculate implied volatility (using Black-Scholes)
                T =  (3 - state.timestamp/ 1000000) / 365  # Time to expiry (annualized)
                iv = BlackScholes.implied_volatility(premium,S, strike, T)  # Assume calls for now
                delta = BlackScholes.delta(S, strike, T, iv)
                total_delta += state.position.get(product,0) * delta



                m = np.log(strike/S)/np.sqrt(T)
                m_list.append(m)
                iv_list.append(iv)

        hedge_delta= total_delta

        # Fit volatility smile (quadratic: iv = a + b*m + c*m²)
        if len(m_list) >= 3:  # Need at least 3 points to fit
            coeffs = np.polyfit(m_list, iv_list, 2)
            a, b, c = coeffs

            # Calculate fitted IV for each strike and identify mispricing
            for strike in strikes:
                product = getattr(Product, f"VOLCANIC_ROCK_VOUCHER_{strike}")
                if product in state.order_depths:
                    m = np.log(strike/S)/np.sqrt(T)
                    fitted_iv = a + b*m + c*(m**2)
                    actual_iv = iv_list[strikes.index(strike)] if strikes.index(strike) < len(iv_list) else None
                    if actual_iv is None:
                        continue
                    iv_diff = actual_iv - fitted_iv

                    # Update trader object with IV difference history
        
                    vol_data[product]["iv_diff_history"].append(iv_diff)
                    if len(vol_data[product]["iv_diff_history"]) > PARAMS[product]["vol_window"]:
                        vol_data[product]["iv_diff_history"].pop(0)

                    # Calculate z-score of IV difference (mean reversion)
                    window = PARAMS[product]["vol_window"]
                    history = vol_data[product]["iv_diff_history"][-window:]
                    if len(history) >= window:
            
                        std_diff = np.std(history)
                        zscore = iv_diff  / std_diff if std_diff != 0 else 0

                        # Trading logic
                        position_limit = PARAMS[product]["limit"]
                        current_pos = state.position.get(product, 0)
                        threshold = PARAMS[product]["zscore_threshold"]

                        if iv_diff > 0:
                            # Overpriced: SELL voucher (hit the bid)
                            if state.order_depths[product].buy_orders:  # Check if there are buyers
                                best_bid = max(state.order_depths[product].buy_orders.keys())
                                bid_volume = state.order_depths[product].buy_orders[best_bid]
                                
                                # Calculate maximum we can sell (current_pos can be negative)
                                max_possible_sell = PARAMS[product]["limit"] + current_pos
                                quantity = min(bid_volume, max_possible_sell)
                                
                                if quantity > 0:
                                    orders[product].append(Order(product, best_bid, -quantity))  # Negative for sell

                        elif iv_diff < 0:
                            # Underpriced: BUY voucher (lift the ask)
                            if state.order_depths[product].sell_orders:  # Check if there are sellers
                                best_ask = min(state.order_depths[product].sell_orders.keys())
                                ask_volume = -state.order_depths[product].sell_orders[best_ask]  # Convert to positive
                                
                                # Calculate maximum we can buy
                                max_possible_buy = PARAMS[product]["limit"] - current_pos
                                quantity = min(ask_volume, max_possible_buy)
                                
                                if quantity > 0:
                                    orders[product].append(Order(product, best_ask, quantity))  # Positive for buy
            
           
            rock_depth = state.order_depths[Product.VOLCANIC_ROCK]
            rock_position= state.position.get(Product.VOLCANIC_ROCK,0)
            if hedge_delta > rock_position:  # Need to buy underlying
                quantity= int(round(hedge_delta - rock_position))
                if quantity > 0 and rock_depth.sell_orders:
                    best_ask = min(rock_depth.sell_orders.keys())
                    best_ask_volume = -rock_depth.sell_orders[best_ask]  # Convert to positive
                    quantity = min(quantity, best_ask_volume)
                    if quantity > 0:
                        hedge_orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
            elif hedge_delta <rock_position:
                quantity= int(round(rock_position - hedge_delta)) 
                if quantity > 0 and rock_depth.buy_orders:
                    best_bid = max(rock_depth.buy_orders.keys())
                    best_bid_volume = rock_depth.buy_orders[best_bid]  # Convert to positive
                    quantity = min(quantity, best_bid_volume)
                    if quantity > 0:
                        hedge_orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders,hedge_orders
    def jam_djembe_orders(
             self,
        order_depths: Dict[str, OrderDepth],
        spread_data: Dict[str, Any],
        jam_position:int,
        djembe_position:int,
    ) -> Tuple[List[Order], List[Order]]:
        jam_orders: List[Order] = []
        djembe_orders: List[Order] = []
        if Product.JAMS not in order_depths.keys() or Product.DJEMBES not in order_depths.keys():
            return jam_orders, djembe_orders

        jams_order_depth = order_depths[Product.JAMS]
        jams_swmid = self.get_swmid(jams_order_depth)
        djembe_order_depth = order_depths[Product.DJEMBES]
        djembe_swmid = self.get_swmid(djembe_order_depth)
        slope= 2.799110039757307
        intercept=  -4833.996099346226
        pred_djembe =  slope * jams_swmid + intercept
        spread = djembe_swmid - pred_djembe
        spread_data["spread_history"].append(spread)
        if (
            len(spread_data["spread_history"])
            < self.params[Product.JAM_DE_SPREAD]["spread_std_window"]
        ):
            return jam_orders, djembe_orders
        elif len(spread_data["spread_history"]) > self.params[Product.JAM_DE_SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        

        zscore = (
            spread - self.params[Product.JAM_DE_SPREAD]["default_spread_mean"]
        ) / spread_std
        spread_product = Product.JAM_DE_SPREAD

        jam_limit= PARAMS[Product.JAMS]['limit']
        djembe_limit= PARAMS[Product.DJEMBES]['limit']
        
        if zscore >= self.params[spread_product]["zscore_threshold"]:
            best_bid = max(djembe_order_depth.buy_orders.keys())
            best_ask = min(jams_order_depth.sell_orders.keys())
            best_bid_volume = abs(djembe_order_depth.buy_orders[best_bid])
            best_ask_volume = abs(jams_order_depth.sell_orders[best_ask])

            jams_quantity = min(best_ask_volume, jam_limit - jam_position)
            jam_orders.append(Order(Product.JAMS, best_ask, jams_quantity))

            djembe_quantity = min(best_bid_volume, djembe_limit + djembe_position)
            djembe_orders.append(Order(Product.DJEMBES, best_bid, -djembe_quantity))
           
        elif zscore <= -self.params[spread_product]["zscore_threshold"]:
            best_bid = max(jams_order_depth.buy_orders.keys())
            best_ask = min(djembe_order_depth.sell_orders.keys())
            best_bid_volume = abs(jams_order_depth.buy_orders[best_bid])
            best_ask_volume = abs(djembe_order_depth.sell_orders[best_ask])

            jams_quantity = min(best_bid_volume, jam_limit + jam_position)
            jam_orders.append(Order(Product.JAMS, best_bid, -jams_quantity))

            djembe_quantity = min(best_ask_volume, djembe_limit - djembe_position)
            djembe_orders.append(Order(Product.DJEMBES, best_ask, djembe_quantity))
        
           

        spread_data["prev_zscore"] = zscore
        return jam_orders, djembe_orders
    
    
    def squid_strategy(self, order_depth: OrderDepth, position: int, squid_data: Dict) -> List[Order]:
        orders = []
        limit = squid_data["limit"]
        
        # 1. Calculate true value
        true_value = self.calculate_squid_true_value(order_depth, squid_data)
        
        # 2. Position management
        to_buy = limit - position
        to_sell = limit + position
        
        # Update position window (for liquidation logic)
        squid_data["position_window"].append(abs(position) == limit)
        
        # 3. Generate orders
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        # Liquidation flags
        window_size = len(squid_data["position_window"])
        soft_liquidate = (window_size == 10 and 
                         sum(squid_data["position_window"]) >= 5 and 
                         squid_data["position_window"][-1])
        hard_liquidate = (window_size == 10 and 
                         all(squid_data["position_window"]))
        if len(squid_data["position_window"])>20:
            squid_data["position_window"].pop(0)
        
        # Dynamic pricing based on position
        max_buy_price = true_value - 1 if position > limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < limit * -0.5 else true_value
        
        # Market take - buy side
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                orders.append(Order("SQUID_INK", price, quantity))
                to_buy -= quantity
        
        # Liquidation orders - buy
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            orders.append(Order("SQUID_INK", true_value, quantity))
            to_buy -= quantity
        elif to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            orders.append(Order("SQUID_INK", true_value - 2, quantity))
            to_buy -= quantity
        
        # Market make - buy side
        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda x: x[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order("SQUID_INK", price, to_buy))
        
        # Market take - sell side
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                orders.append(Order("SQUID_INK", price, -quantity))
                to_sell -= quantity
        
        # Liquidation orders - sell
        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            orders.append(Order("SQUID_INK", true_value, -quantity))
            to_sell -= quantity
        elif to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            orders.append(Order("SQUID_INK", true_value + 2, -quantity))
            to_sell -= quantity
        
        # Market make - sell side
        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda x: x[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            orders.append(Order("SQUID_INK", price, -to_sell))
        
        return orders

    def calculate_squid_true_value(self, order_depth: OrderDepth, squid_data: Dict) -> int:
        buys = order_depth.buy_orders
        sells = order_depth.sell_orders
        
        # DWAP Calculation
        buy_dwap_num = sum(p*v for p,v in buys.items())
        buy_dwap_den = sum(buys.values())
        
        sell_dwap_num = sum(p*-v for p,v in sells.items())
        sell_dwap_den = sum(-v for v in sells.values())
        
        if buy_dwap_den + sell_dwap_den == 0:
            dwap = 2000
        else:
            dwap = (
                (buy_dwap_num + sell_dwap_num) / (buy_dwap_den + sell_dwap_den)
                if buy_dwap_den > 0 and sell_dwap_den > 0
                else (
                    buy_dwap_num / buy_dwap_den if buy_dwap_den > 0
                    else sell_dwap_num / sell_dwap_den
                )
            )
        
        # Update DWAP history
        squid_data["dwap_history"].append(dwap)
        if len(squid_data["dwap_history"]) > 7:
            squid_data["dwap_history"].pop(0)
        
        # Trend detection
        trend_up = (len(squid_data["dwap_history"]) >= 3 and
                   squid_data["dwap_history"][-1] > squid_data["dwap_history"][-3])
        trend_down = (len(squid_data["dwap_history"]) >= 3 and
                     squid_data["dwap_history"][-1] < squid_data["dwap_history"][-3])
        
        # Order book imbalance
        total_bid = sum(buys.values())
        total_ask = sum(-v for v in sells.values())
        imbalance = ((total_bid - total_ask) / 
                    (total_bid + total_ask + 1e-6))  # Avoid division by zero
        
        # Skew adjustment
        if trend_up:
            skew = imbalance * 5.5
        elif trend_down:
            skew = imbalance * 0.9
        else:
            skew = imbalance * 3.2
        
        true_value = dwap + skew
        return int(true_value)
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0
       
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.KELP not in traderObject:
            traderObject[Product.KELP] = {
                "kelp_prices": [],
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.SQUID_INK not in traderObject:
            traderObject[Product.SQUID_INK]={
                "position": 0,
                "dwap_history": [],
                "position_window": [],
                "limit": 50,
            }
        
        if Product.JAM_DE_SPREAD not in traderObject:
            traderObject[Product.JAM_DE_SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.JAM_SQUID_SPREAD not in traderObject:
            traderObject[Product.JAM_SQUID_SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.VOLCANIC_ROCK_VOUCHER_9500 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_9500] = {
                "iv_diff_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.VOLCANIC_ROCK_VOUCHER_9750 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_9750] = {
                "iv_diff_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }  
        if Product.VOLCANIC_ROCK_VOUCHER_10000 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_10000] = {
                "iv_diff_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            } 
        if Product.VOLCANIC_ROCK_VOUCHER_10250 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_10250] = {
                "iv_diff_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        if Product.VOLCANIC_ROCK_VOUCHER_10500 not in traderObject:
            traderObject[Product.VOLCANIC_ROCK_VOUCHER_10500] = {
                "iv_diff_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        # if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
        #     resin_position = ( state.position[Product.RAINFOREST_RESIN]if Product.RAINFOREST_RESIN in state.position else 0)
        #     resin_orders = self.resin_orders(state.order_depths[Product.RAINFOREST_RESIN],
        #                                      PARAMS[Product.RAINFOREST_RESIN]['acceptance_value'], resin_position,
        #                                      PARAMS[Product.RAINFOREST_RESIN]['limit'])
        #     result[Product.RAINFOREST_RESIN] = resin_orders

        # if Product.KELP in self.params and Product.KELP in state.order_depths:
        #     kelp_position = ( state.position[Product.KELP]if Product.KELP in state.position else 0)
        #     kelp_orders = self.kelp_orders(state.order_depths[Product.KELP],PARAMS[Product.KELP]['timespan'] ,
        #                                     kelp_position, PARAMS[Product.KELP]['limit'],
        #                                     traderObject[Product.KELP])
        #     result[Product.KELP] = kelp_orders

        # basket_position = (
        #     state.position[Product.PICNIC_BASKET1]
        #     if Product.PICNIC_BASKET1 in state.position
        #     else 0
        # )
        # spread_orders = self.spread_orders(
        #     state.order_depths,
        #     Product.PICNIC_BASKET1,
        #     basket_position,
        #     traderObject[Product.SPREAD],
        #     Product.SPREAD,
        #     Product.SYNTHETIC
        # )
       
        # if spread_orders != None:
        #     # result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
        #     # result[Product.JAMS] = spread_orders[Product.JAMS]
        #     # result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
        #     result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]
        
        # basket_position_2 = (
        #     state.position[Product.PICNIC_BASKET2]
        #     if Product.PICNIC_BASKET2 in state.position
        #     else 0
        # )
        # spread_orders2 = self.spread_orders(
        #     state.order_depths,
        #     Product.PICNIC_BASKET2,
        #     basket_position_2,
        #     traderObject[Product.SPREAD2],
        #     Product.SPREAD2,
        #     Product.SYNTHETIC2
        # )
        # if spread_orders2 != None:
        #     result[Product.CROISSANTS] = spread_orders2[Product.CROISSANTS]
        #     result[Product.JAMS] = spread_orders2[Product.JAMS]
        #     result[Product.DJEMBES] = spread_orders2[Product.DJEMBES]
        #     result[Product.PICNIC_BASKET2] = spread_orders2[Product.PICNIC_BASKET2]
        
        # if Product.JAMS in state.order_depths and Product.DJEMBES in state.order_depths:
        #     jam_position = ( state.position[Product.JAMS]if Product.JAMS in state.position else 0)
        #     djembe_position = ( state.position[Product.DJEMBES]if Product.DJEMBES in state.position else 0)
        #     spread_orders_3 = self.jam_djembe_orders(
        #         state.order_depths,
        #         traderObject[Product.JAM_DE_SPREAD],
        #         jam_position,
        #         djembe_position,
        #     )
        #     if spread_orders_3 != None:
        #         result[Product.JAMS] = spread_orders_3[0]
        #         result[Product.DJEMBES] = spread_orders_3[1]
        if Product.VOLCANIC_ROCK in state.order_depths:
            volcanic_orders, hedge_orders = self.voucher_strategy(
                state, 
                state.position.get(Product.VOLCANIC_ROCK, 0),
                traderObject
            )
            # Add to result dict
            for product, product_orders in volcanic_orders.items():
                result[product] = product_orders
            result[Product.VOLCANIC_ROCK]=hedge_orders
                    
               

        # if Product.SQUID_INK in state.order_depths:
        #     squid_orders = self.squid_strategy(
        #         state.order_depths[Product.SQUID_INK],
        #         state.position.get(Product.SQUID_INK, 0),
        #         traderObject[Product.SQUID_INK]
        #     )
        #     result[Product.SQUID_INK] = squid_orders


        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    