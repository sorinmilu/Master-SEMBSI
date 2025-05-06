import json

class DietOntology:
    def __init__(self, json_data):
        # Load data from the provided JSON data
        self.food_items = json_data["alimente"]
        self.preferinte_dietetice = json_data["preferințe_dietetice"]
        self.meals = json_data["mese"]
        self.health_conditions = json_data["condiții_medicale"]
        self.alergii = json_data["alergii"]

    # Query 1: Get foods suitable for a specific diet
    def get_suitable_foods_for_diet(self, diet_name):
        if diet_name in self.preferinte_dietetice:
            return self.preferinte_dietetice[diet_name]
        return []

    # Query 2: Get foods to avoid for a specific allergy
    def get_foods_to_avoid_for_allergy(self, allergy_name):
        if allergy_name in self.alergii:
            return self.alergii[allergy_name]
        return []

    # Query 3: Get meals that contain a specific food
    def get_meals_with_food(self, food_name):
        meals_with_food = []
        for meal, details in self.meals.items():
            if isinstance(details, dict) and "contine" in details:
                if isinstance(details["contine"], list) and food_name in details["contine"]:
                    meals_with_food.append(meal)
        return meals_with_food

    # Query 4: Get nutrients in a specific food item
    def get_nutrients_for_food(self, food_name):
        if food_name in self.food_items:
            return self.food_items[food_name]["nutrienti_la_100g"]
        return {}

    def calculate_meal_nutrition(self, meal):
            total_calories = 0
            total_protein = 0
            total_carbs = 0
            total_fats = 0
            
            for food_item, weight in meal.items():
                # Get food item information from the food_items dictionary
                if food_item in self.food_items:
                    food_info = self.food_items[food_item]['nutrienti_la_100g']
                    calories_per_100g = food_info["kcal"]
                    protein_per_100g = food_info["proteine"]
                    carbs_per_100g = food_info["carbohidrati"]
                    fats_per_100g = food_info["grasimi"]
                    
                    # Calculate the nutrients for the specific weight of the food item
                    calories = (calories_per_100g * weight) / 100
                    protein = (protein_per_100g * weight) / 100
                    carbs = (carbs_per_100g * weight) / 100
                    fats = (fats_per_100g * weight) / 100
                    
                    # Add to the totals
                    total_calories += calories
                    total_protein += protein
                    total_carbs += carbs
                    total_fats += fats
            
            # Return the totals as a dictionary
            return {
                "calorii_totale": total_calories,
                "proteine_totale": total_protein,
                "carbohidrati_totali": total_carbs,
                "grăsimi_totale": total_fats
            }

# Load JSON data from a file
def load_json_data(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Example usage
# Load the data from a JSON file
json_data = load_json_data("alimente.json")

# Initialize the DietOntology with the loaded data
ontology = DietOntology(json_data)

# Query 1: Get suitable foods for a Vegan diet
print("Mancaruri pentru dieta vegana:", ontology.get_suitable_foods_for_diet("vegan"))

# Query 2: Get foods to avoid for a Gluten allergy
print("Mancaruri pentru alergii la gluten:", ontology.get_foods_to_avoid_for_allergy("gluten"))

# Query 3: Get meals containing Apple
print("Mancaruri cu mar:", ontology.get_meals_with_food("Mar"))

# Query 4: Get nutrients in Chicken (Piept de pui)
print("Nutrienti in carnea de pui:", ontology.get_nutrients_for_food("Piept de pui"))

meal = {
    "Piept de pui": 250,   # 250 grams of chicken
    "Cartof": 200,        # 200 grams of potatoes
    "Pâine integrală": 130 # 130 grams of whole wheat bread
}

# Calculate the total nutrition for the meal
meal_nutrition = ontology.calculate_meal_nutrition(meal)

# Print the total macronutrients and calories for the meal
print(f"    Informații nutritive:")
print(f"    Calorii: {meal_nutrition['calorii_totale']:.3f} kcal")
print(f"    Proteine: {meal_nutrition['proteine_totale']:.3f} g")
print(f"    Carbohidrati: {meal_nutrition['carbohidrati_totali']:.3f} g")
print(f"    Grasimi: {meal_nutrition['grăsimi_totale']:.3f} g")