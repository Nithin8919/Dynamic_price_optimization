import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
import logging
from src.components.utils import save_object, load_object


@dataclass
class OptimizationConfig:
    optimized_price_file_path = "artifacts/optimized_price.pkl"
    accuracy_evaluation_file_path = "artifacts/optimization_accuracy.pkl"


class PriceOptimizer:
    def __init__(self, config=OptimizationConfig()):
        self.config = config

    def load_model_and_data(self):
        """Load the trained model and test data for optimization."""
        try:
            self.model = load_object("artifacts/model.pkl")
            self.data = load_object("artifacts/test_data.pkl")
            logging.info("Model and data loaded successfully for optimization.")
        except Exception as e:
            logging.error(f"Error loading model or data: {e}")
            raise e

    def objective_function(self, price, features):
        """Objective function for optimization (e.g., profit maximization)."""
        demand = self.model.predict(features)
        revenue = price * demand
        return -revenue  # Negative for minimization (maximize revenue)

    def scipy_optimization(self, features):
        """Run scipy optimization to find optimal price."""
        initial_guess = 50
        bounds = [(10, 200)]
        result = minimize(self.objective_function, x0=initial_guess, args=(features,), bounds=bounds)
        return result.x[0] if result.success else None

    def hjb_optimization(self, features):
        """Placeholder for custom HJB algorithm implementation."""
        # Custom logic for HJB optimization
        optimal_price = 100  # Placeholder logic
        return optimal_price

    def gils_optimization(self, features):
        """Placeholder for custom GILS algorithm implementation."""
        optimal_price = 100  # Placeholder logic
        return optimal_price

    def evaluate_accuracy(self, optimized_prices):
        """Calculate and log estimated revenue to evaluate optimization accuracy."""
        estimated_revenue = []
        for idx, price in enumerate(optimized_prices):
            if price is not None:
                features = self.data.iloc[idx].drop("Historical_Cost_of_Ride").values.reshape(1, -1)
                demand = self.model.predict(features)
                revenue = price * demand
                estimated_revenue.append(revenue)
        average_revenue = np.mean(estimated_revenue) if estimated_revenue else 0
        logging.info(f"Average estimated revenue from optimized prices: {average_revenue}")
        save_object(self.config.accuracy_evaluation_file_path, estimated_revenue)

    def run_optimization(self):
        self.load_model_and_data()
        algorithms = ["scipy", "hjb", "gils"]
        best_algorithm, best_results = None, []

        for algorithm in algorithms:
            optimized_prices = []
            logging.info(f"Running optimization using the {algorithm} algorithm.")

            for idx, row in self.data.iterrows():
                features = row.drop("Historical_Cost_of_Ride").values.reshape(1, -1)

                # Select the optimization method
                if algorithm == "scipy":
                    optimal_price = self.scipy_optimization(features)
                elif algorithm == "hjb":
                    optimal_price = self.hjb_optimization(features)
                elif algorithm == "gils":
                    optimal_price = self.gils_optimization(features)

                if optimal_price is not None and optimal_price > 0:
                    optimized_prices.append(optimal_price)
                else:
                    logging.warning(f"No valid optimized price for row {idx} using {algorithm}.")

            # Save optimization results and check if they are better
            if optimized_prices:
                save_object(self.config.optimized_price_file_path, optimized_prices)
                logging.info(f"{algorithm} optimization completed and results saved.")
                if not best_results or np.mean(optimized_prices) > np.mean(best_results):
                    best_algorithm, best_results = algorithm, optimized_prices

            else:
                logging.warning(f"No optimized prices found for {algorithm}.")

        if best_results:
            logging.info(f"Best optimization algorithm: {best_algorithm}")
            self.evaluate_accuracy(best_results)
        else:
            logging.warning("No successful optimization results were generated.")
