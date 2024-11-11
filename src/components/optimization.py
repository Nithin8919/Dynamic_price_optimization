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
        self.optimized_result = None  # To store the final output for post-processing

    def load_model_and_data(self):
        """Load predicted prices and test data for optimization."""
        try:
            self.predicted_prices = load_object("artifacts/predicted_prices.pkl")  # Load predicted prices
            self.data = load_object("artifacts/test_data.pkl")  # Load additional data if needed
            logging.info("Predicted prices and data loaded successfully for optimization.")
        except Exception as e:
            logging.error(f"Error loading predicted prices or data: {e}")
            raise e

    def objective_function(self, price, predicted_price):
        """Objective function for optimization (e.g., fine-tuning around predicted price)."""
        # Here you could define an objective to adjust around the predicted price
        # For example, minimizing the difference from predicted price within a revenue-based constraint
        return abs(price - predicted_price)  # Simple example: minimize deviation from initial prediction

    def scipy_optimization(self, predicted_price):
        """Run scipy optimization to fine-tune around the predicted price."""
        initial_guess = predicted_price
        bounds = [(predicted_price * 0.8, predicted_price * 1.2)]  # Allow a 20% variation around prediction
        result = minimize(self.objective_function, x0=initial_guess, args=(predicted_price,), bounds=bounds)
        return result.x[0] if result.success else None

    def evaluate_accuracy(self, optimized_prices):
        """Calculate and log estimated revenue to evaluate optimization accuracy."""
        estimated_revenue = [price for price in optimized_prices if price is not None]
        average_revenue = np.mean(estimated_revenue) if estimated_revenue else 0
        logging.info(f"Average estimated revenue from optimized prices: {average_revenue}")
        save_object(self.config.accuracy_evaluation_file_path, estimated_revenue)
        return average_revenue

    def run_optimization(self):
        self.load_model_and_data()
        algorithms = ["scipy"]  # Only using scipy here for fine-tuning
        best_algorithm, best_results = None, []
        best_average_revenue = 0

        for algorithm in algorithms:
            optimized_prices = []
            logging.info(f"Running optimization using the {algorithm} algorithm.")

            for idx, predicted_price in enumerate(self.predicted_prices):
                # Fine-tune around the predicted price
                if algorithm == "scipy":
                    optimal_price = self.scipy_optimization(predicted_price)

                if optimal_price is not None and optimal_price > 0:
                    optimized_prices.append(optimal_price)
                else:
                    logging.warning(f"No valid optimized price for index {idx} using {algorithm}.")

            # Evaluate and save the best optimization results
            if optimized_prices:
                average_revenue = np.mean(optimized_prices)
                if average_revenue > best_average_revenue:
                    best_algorithm, best_results, best_average_revenue = algorithm, optimized_prices, average_revenue
                    save_object(self.config.optimized_price_file_path, best_results)
                    logging.info(f"{algorithm} optimization completed with best results saved.")

        # Evaluate accuracy of the best results
        if best_results:
            final_revenue = self.evaluate_accuracy(best_results)
            self.optimized_result = {
                "best_algorithm": best_algorithm,
                "optimized_prices": best_results,
                "average_revenue": final_revenue
            }
            logging.info(f"Best optimization algorithm: {best_algorithm} with average revenue: {final_revenue}")
        else:
            logging.warning("No successful optimization results were generated.")
            self.optimized_result = None

        return self.optimized_result


# Code to run the optimizer and display the results
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Instantiate the optimizer
    optimizer = PriceOptimizer()

    # Run optimization and get the result
    optimized_result = optimizer.run_optimization()

    # Display the optimized result
    if optimized_result:
        print("Optimization Results:")
        print(f"Best Algorithm: {optimized_result['best_algorithm']}")
        print(f"Optimized Prices: {optimized_result['optimized_prices']}")
        print(f"Average Revenue: {optimized_result['average_revenue']}")
    else:
        print("No optimization results generated.")
