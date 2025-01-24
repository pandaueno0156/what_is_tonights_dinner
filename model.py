import torch
import torch.nn as nn
import torch.optim as optim
import random

# Sample restaurant data (name, type, rating)
restaurants = [
    {"name": "R1", "features": [1, 0, 4.5]},  # Example features: [type1, type2, rating]
    {"name": "R2", "features": [0, 1, 4.0]},
    {"name": "R3", "features": [1, 1, 3.8]},
    {"name": "R4", "features": [0, 0, 5.0]},
    {"name": "R5", "features": [1, 0, 4.2]},
    {"name": "R6", "features": [0, 1, 3.5]},
    {"name": "R7", "features": [1, 0, 4.7]},
    {"name": "R8", "features": [0, 1, 3.9]},
    {"name": "R9", "features": [1, 1, 4.1]},
    {"name": "R10", "features": [0, 0, 4.8]},
    {"name": "R11", "features": [1, 0, 4.9]},
    {"name": "R12", "features": [0, 1, 4.6]},
]
# Convert features to tensors
for r in restaurants:
    r["features"] = torch.tensor(r["features"], dtype=torch.float32)


restaurants_name_feature_dict = {r["name"]: r for r in restaurants}
print(restaurants_name_feature_dict)
# Define the RankNet model
class RankNet(nn.Module):
    def __init__(self, input_size):
        super(RankNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),  # Hidden layer
            nn.ReLU(),
            nn.Linear(16, 1)  # Single score output
        )
    
    def forward(self, x):
        return self.fc(x)

# Initialize the RankNet model
input_size = len(restaurants[0]["features"])  # Number of features per restaurant
model = RankNet(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Helper function to predict and train
def train_ranknet(model, optimizer, criterion, rounds=10):
    remaining_restaurants = restaurants.copy()
    
    # keep track of the current winner and contender with names
    current_winner = None
    current_contender = None
    
    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")

        # To ensure that there are at least two restaurants left for comparison
        if len(remaining_restaurants) < 2:
            print("Not enough restaurants left for comparison.")
            break
        
        if round_num == 0:
            # First round: randomly select two restaurants
            selected_restaurants = random.sample(remaining_restaurants, 2)
            rest_a, rest_b = selected_restaurants[0], selected_restaurants[1]

            # remove the selected restaurants from the remaining_restaurants list
            remaining_restaurants.remove(rest_a)
            remaining_restaurants.remove(rest_b)
        else:
            # Subsequent rounds: pick the remaining restaurant (highest-scoring) and pair it with the 
            # current_winner to compete
            remaining_restaurants_scores = [{f"name": r["name"], "score": model(r["features"])} for r in remaining_restaurants]
            remaining_restaurants_scores.sort(key=lambda x: x["score"], reverse=True)
            current_contender = remaining_restaurants_scores[0]["name"]
            rest_a = restaurants_name_feature_dict.get(current_winner)
            rest_b = restaurants_name_feature_dict.get(current_contender)

            # remove the current_contender from the remaining_restaurants list
            remaining_restaurants.remove(rest_b)

        features_a = rest_a["features"]
        features_b = rest_b["features"]
        
        # Compute scores for both restaurants
        score_a = model(features_a)
        score_b = model(features_b)
        
        # Compute the probability that the user will prefer A over B (the model prediction)
        prob_a = torch.sigmoid(score_a - score_b).item()
        # print(f"Restaurant A: {rest_a['name']}, Score: {score_a.item():.3f}")
        # print(f"Restaurant B: {rest_b['name']}, Score: {score_b.item():.3f}")
        # print(f"Predicted P(A > B): {prob_a:.3f}")
        
        # Simulate user input (replace this with actual user feedback)
        print(f"Option A: Restaurant: {rest_a['name']}")
        print(f"Option B: Restaurant: {rest_b['name']}")
        print()

        correct_answer_flag = False
        while not correct_answer_flag:
            user_choice = input("Enter your choice of restaurant (A or B): ").strip().upper()
            
            if user_choice == "A" or user_choice == "B":
                print(f"User chose: {user_choice}")
                correct_answer_flag = True
            else:
                print("Invalid input. Please enter A or B.")
                print()

        # This round's winner is the one that the user chose
        if user_choice == "A":
            current_winner = rest_a["name"]
        else:
            current_winner = rest_b["name"]
        
        # Determine the ground truth label
        y = torch.tensor(1.0 if user_choice == "A" else 0.0).unsqueeze(0)  # Reshaped to match prob_a_tensor
        
        # Compute loss and backpropagate
        optimizer.zero_grad()
        prob_a_tensor = torch.sigmoid(score_a - score_b)
        loss = criterion(prob_a_tensor, y)
        loss.backward()
        optimizer.step()
        
        print(f"Training Loss: {loss.item():.4f}")

    # Final restaurant recommendation
    print(f"\nFinal Recommendation: {current_winner}")

# Run the training loop
train_ranknet(model, optimizer, criterion)
