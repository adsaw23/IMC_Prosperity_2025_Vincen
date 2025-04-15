import stat
import jsonpickle
import numpy as np
from typing import List,Dict, Any, Tuple
import json
import pandas as pd
import string
from statistics import NormalDist

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        "vol_window": 20,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750:{
        "limit": 200,
        "strike":9750,
        "vol_window": 20,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000:{
        "limit": 200,
        "strike":10000,
        "vol_window": 20,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250:{
        "limit": 200,
        "strike":10250,
        "vol_window": 20,
        "zscore_threshold": 2,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500:{
        "limit": 200,
        "strike":10500,
        "vol_window": 20,
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
        "zscore_threshold": 5,
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
    Product.JAM_SQUID_SPREAD: { 
        "default_spread_mean": 13061.36,
        "default_spread_std": 12.14,
        "spread_std_window": 25,
        "zscore_threshold": 1.5,
    }
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
            if product == Product.PICNIC_BASKET2:
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

        if zscore >= self.params[spread_product]["zscore_threshold"]:
            if basket_position != -self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,
                    synth_product
                    
                )

        if zscore <= -self.params[spread_product]["zscore_threshold"]:
            if basket_position != self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product,synth_product
                )

        spread_data["prev_zscore"] = zscore
        return None
   

    def voucher_strategy(
        self,
        order_depths: Dict[str, OrderDepth],
        position: int,
        vol_data: Dict[str, Any],  # Stores volatility history
        product: Product

    ) -> List[Order]:
        orders: List[Order] = []
        
        if Product.VOLCANIC_ROCK_VOUCHER_9500 not in order_depths.keys():
            return orders

        # 1. Get current market data
        order_depth = order_depths[product]
        current_swmid = self.get_swmid(order_depth)
        current_underlying = self.get_swmid(order_depths[Product.VOLCANIC_ROCK])
        if current_swmid is None or current_underlying is None:
            return orders
        tte = 5/365  # Days to expiry
        
        # 2. Calculate moneyness and implied vol (updated parameters)
        m_t = np.log(PARAMS[product]["strike"]/current_underlying)/np.sqrt(tte)
        current_iv = BlackScholes.implied_volatility(
            call_price=current_swmid,
            spot=current_underlying,
            strike=PARAMS[product]["strike"],
            time_to_expiry=tte,
            risk_free_rate=0.0
        )
        
        # Handle case where IV calculation failed
        if np.isnan(current_iv):
            return orders
        
        # 3. Get fitted IV from parabola coefficients
        fitted_iv = 0.1961 + (-0.0579)*m_t + (-0.5406)*(m_t**2)
        iv_diff = current_iv - fitted_iv
        
        # 4. Update volatility history
        vol_data["iv_diff_history"].append(iv_diff)
        if len(vol_data["iv_diff_history"]) > self.params[product]["vol_window"]:
            vol_data["iv_diff_history"].pop(0)
        
        # 5. Generate signals when we have enough data
        if len(vol_data["iv_diff_history"]) < self.params[product]["vol_window"]:
            return orders
            
        iv_std = np.std(vol_data["iv_diff_history"])
        zscore = iv_diff / iv_std if iv_std != 0 else 0
        
        # 6. Trading logic
        position_limit = PARAMS[product]['limit']
        
        if zscore >= self.params[product]["zscore_threshold"]:
            # Overpriced - SELL
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = abs(order_depth.buy_orders[best_bid])
            sell_volume = min(best_bid_volume, position_limit + position)
            orders.append(Order(product, best_bid, -sell_volume))
            
        elif zscore <= -self.params[product]["zscore_threshold"]:
            # Underpriced - BUY
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            buy_volume = min(best_ask_volume, position_limit - position)
            orders.append(Order(product, best_ask, buy_volume))
        
        vol_data["prev_zscore"] = zscore
        return orders
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
    
    
    def squid_jams_orders(
    self,
    order_depths: Dict[str, OrderDepth],
    spread_data: Dict[str, Any],
    jam_position: int,
    squid_position: int,
) -> Tuple[List[Order], List[Order]]:
        jam_orders: List[Order] = []
        squid_orders: List[Order] = []
        jam_position= jam_position+ PARAMS[Product.JAMS]["buy_order_volume"] - PARAMS[Product.JAMS]["sell_order_volume"]
        
        # Check if both products exist in order depths
        if Product.JAMS not in order_depths or Product.SQUID_INK not in order_depths:
            return jam_orders, squid_orders

        # Get SWMID prices
        jams_order_depth = order_depths[Product.JAMS]
        jams_swmid = self.get_swmid(jams_order_depth)
        
        squid_order_depth = order_depths[Product.SQUID_INK]
        squid_swmid = self.get_swmid(squid_order_depth)

        # Quadratic regression coefficients
        intercept = -2703.695826
        linear_coef = 10.213396
        quadratic_coef = -0.002819

        # Calculate predicted Jams price using quadratic model
        predicted_jams = (quadratic_coef * squid_swmid**2 + 
                        linear_coef * squid_swmid + 
                        intercept)

        # Spread calculation (actual vs predicted)
        spread = jams_swmid + predicted_jams  # Residual from regression
        spread_data["spread_history"].append(spread)

        # Maintain rolling window
        if len(spread_data["spread_history"]) > self.params[Product.JAM_SQUID_SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        # Skip if not enough data
        if len(spread_data["spread_history"]) < self.params[Product.JAM_SQUID_SPREAD]["spread_std_window"]:
            return jam_orders, squid_orders

        # Calculate z-score
        spread_mean = np.mean(spread_data["spread_history"])
        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - spread_mean) / self.params[Product.JAM_SQUID_SPREAD]['default_spread_std']

        # Get position limits
        jam_limit = PARAMS[Product.JAMS]['limit']
        squid_limit = PARAMS[Product.SQUID_INK]['limit']

        # Trading logic for negative correlation
        if zscore >= self.params[Product.JAM_SQUID_SPREAD]["zscore_threshold"]:
            # Spread too HIGH: Sell Jams (overpriced), Buy Squid (underpriced)
            best_bid_jam = max(jams_order_depth.buy_orders.keys())
            best_ask_squid = min(squid_order_depth.sell_orders.keys())

            jam_qty = min(
                jams_order_depth.buy_orders[best_bid_jam],
                jam_limit + jam_position  # Can sell up to (limit + position)
            )
            squid_qty = min(
                -squid_order_depth.sell_orders[best_ask_squid],
                squid_limit - squid_position  # Can buy up to (limit - position)
            )

            jam_orders.append(Order(Product.JAMS, best_bid_jam, -jam_qty))
            squid_orders.append(Order(Product.SQUID_INK, best_ask_squid, squid_qty))

        elif zscore <= -self.params[Product.JAM_SQUID_SPREAD]["zscore_threshold"]:
            # Spread too LOW: Buy Jams (underpriced), Sell Squid (overpriced)
            best_ask_jam = min(jams_order_depth.sell_orders.keys())
            best_bid_squid = max(squid_order_depth.buy_orders.keys())

            jam_qty = min(
                -jams_order_depth.sell_orders[best_ask_jam],
                jam_limit - jam_position  # Can buy up to (limit - position)
            )
            squid_qty = min(
                squid_order_depth.buy_orders[best_bid_squid],
                squid_limit + squid_position  # Can sell up to (limit + position)
            )

            # if jam_qty > 0 and squid_qty > 0:
            jam_orders.append(Order(Product.JAMS, best_ask_jam, jam_qty))
            squid_orders.append(Order(Product.SQUID_INK, best_bid_squid, -squid_qty))

        # Store current z-score for reference
        spread_data["prev_zscore"] = zscore
        
        return jam_orders, squid_orders

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
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = ( state.position[Product.RAINFOREST_RESIN]if Product.RAINFOREST_RESIN in state.position else 0)
            resin_orders = self.resin_orders(state.order_depths[Product.RAINFOREST_RESIN],
                                             PARAMS[Product.RAINFOREST_RESIN]['acceptance_value'], resin_position,
                                             PARAMS[Product.RAINFOREST_RESIN]['limit'])
            result[Product.RAINFOREST_RESIN] = resin_orders

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = ( state.position[Product.KELP]if Product.KELP in state.position else 0)
            kelp_orders = self.kelp_orders(state.order_depths[Product.KELP],PARAMS[Product.KELP]['timespan'] ,
                                            kelp_position, PARAMS[Product.KELP]['limit'],
                                            traderObject[Product.KELP])
            result[Product.KELP] = kelp_orders

        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD],
            Product.SPREAD,
            Product.SYNTHETIC
        )
       
        if spread_orders != None:
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]
        
        basket_position_2 = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )
        spread_orders2 = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket_position_2,
            traderObject[Product.SPREAD2],
            Product.SPREAD2,
            Product.SYNTHETIC2
        )
        if spread_orders2 != None:
            result[Product.CROISSANTS] = spread_orders2[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders2[Product.JAMS]
            result[Product.DJEMBES] = spread_orders2[Product.DJEMBES]
            result[Product.PICNIC_BASKET2] = spread_orders2[Product.PICNIC_BASKET2]
        
        if Product.JAMS in state.order_depths and Product.DJEMBES in state.order_depths:
            jam_position = ( state.position[Product.JAMS]if Product.JAMS in state.position else 0)
            djembe_position = ( state.position[Product.DJEMBES]if Product.DJEMBES in state.position else 0)
            spread_orders_3 = self.jam_djembe_orders(
                state.order_depths,
                traderObject[Product.JAM_DE_SPREAD],
                jam_position,
                djembe_position,
            )
            if spread_orders_3 != None:
                result[Product.JAMS] = spread_orders_3[0]
                result[Product.DJEMBES] = spread_orders_3[1]
        vouchers: Product= [Product.VOLCANIC_ROCK_VOUCHER_9500, Product.VOLCANIC_ROCK_VOUCHER_9750]
        for voucher in vouchers:
            if voucher in state.order_depths:
                volcanic_orders = self.voucher_strategy(
                state.order_depths,
                state.position[voucher] if voucher in state.position else 0,
                traderObject[voucher],
                voucher
            )
            result[voucher] = volcanic_orders
            
               


        # if Product.JAMS in state.order_depths and Product.SQUID_INK in state.order_depths:
        #     jam_position = ( state.position[Product.JAMS]if Product.JAMS in state.position else 0)
        #     squid_position = ( state.position[Product.SQUID_INK]if Product.SQUID_INK in state.position else 0)
        #     spread_orders_4 = self.squid_jams_orders(
        #         state.order_depths,
        #         traderObject[Product.JAM_SQUID_SPREAD],
        #         jam_position,
        #         squid_position,
        #     )
        #     if spread_orders_4 != None:
        #         # result[Product.JAMS] += spread_orders_4[0]
        #         result[Product.SQUID_INK] = spread_orders_4[1]


        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    