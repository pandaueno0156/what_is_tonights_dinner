import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from datetime import datetime
from scrapping import scrape_restaurant_data

def main():
    try:
        # this function will scrape the restaurant data and save it to the database
        # This will ensure that the database is updated with the latest restaurant data
        scrape_restaurant_data()
    except Exception as e:
        print(f"Error: {e}")

    # read the most recent database
    df = pd.read_csv('restaurant_database.csv')
    # print(df)

    def transform_to_restaurants(df):
        new_df = df.copy()

        # filter out the rows where is_open is 0
        # To ensure that we only consider the restaurants that are currently open
        new_df = new_df[new_df['is_open'] == 1]
        new_df.drop(columns=['last_time_scraped','is_open', 'restaurant_address', 'restaurant_img'], inplace=True)
        
        # normalize the ratings in new_df
        new_df['restaurant_rating'] = (new_df['restaurant_rating'] - 0.0) / (5.0 - 0.0)
        # print(new_df['restaurant_rating'])

        features_columns = [f for f in new_df.columns if f not in ['restaurant_name']]

        output_list = [
            {
                "name": name,
                "features": features.tolist()
            }
            for name, features in zip(new_df['restaurant_name'], new_df[features_columns].values)
        ]
        return output_list

    restaurants = transform_to_restaurants(df)

    # Sample restaurant data (name, type, rating)
    # restaurants = [
    #     {"name": "R1", "features": [1, 0, 4.5]},  # Example features: [type1, type2, rating]
    #     {"name": "R2", "features": [0, 1, 4.0]},
    #     {"name": "R3", "features": [1, 1, 3.8]},
    #     {"name": "R4", "features": [0, 0, 5.0]},
    #     {"name": "R5", "features": [1, 0, 4.2]},
    #     {"name": "R6", "features": [0, 1, 3.5]},
    #     {"name": "R7", "features": [1, 0, 4.7]},
    #     {"name": "R8", "features": [0, 1, 3.9]},
    #     {"name": "R9", "features": [1, 1, 4.1]},
    #     {"name": "R10", "features": [0, 0, 4.8]},
    #     {"name": "R11", "features": [1, 0, 4.9]},
    #     {"name": "R12", "features": [0, 1, 4.6]},
    # ]
    # Convert features to tensors
    for r in restaurants:
        r["features"] = torch.tensor(r["features"], dtype=torch.float32)


    restaurants_name_feature_dict = {r["name"]: r for r in restaurants}
    # print(restaurants_name_feature_dict)

    class TimeAwarePreference:
        def __init__(self, decay_rate=0.9, memory_window=10, time_unit="days"):
            self.history = []
            self.decay_rate = decay_rate
            self.memory_window = memory_window
            self.time_unit = time_unit

        def add_comparison(self,rest_a, rest_b, prob_a, preference, timestamp):
            self.history.append({
                "rest_a": rest_a,
                "rest_b": rest_b,
                "prob": prob_a,
                "preference": preference, # Ground truth label (1.0 if rest_a is preferred, 0.0 if rest_b is preferred)
                "timestamp": timestamp
            })
            if len(self.history) > self.memory_window:
                self.history.pop(0)
        def get_time_difference_calculation(self, current_time, past_time):
            time_difference = (current_time - past_time)

            if self.time_unit == "days":
                # we calculate days based on fraction of days to be more time-sensitive
                return (time_difference).total_seconds() / (24 * 60 * 60)
            elif self.time_unit == "hours":
                return (time_difference).total_seconds() / 3600
            else:
                raise ValueError(f"Invalid time unit: {self.time_unit}")

        def train_with_history(self, model, optimizer, current_comparison):
            total_loss = 0.0
            weight_sum = 0.0
            current_time = current_comparison["timestamp"]

            # Firstly, compute the loss for the current comparison
            # Then, compute the loss for the historical comparisons
            # Then, update the model parameters
            for past_comparison in self.history:
                past_time = past_comparison["timestamp"]
                value_time_difference = self.get_time_difference_calculation(current_time, past_time)
                time_weight = self.decay_rate ** (value_time_difference)

                # get model prediction for the past comparison
                score_a = model(past_comparison["rest_a"]["features"])
                score_b = model(past_comparison["rest_b"]["features"])
                past_prob_a = torch.sigmoid(score_a - score_b)

                # convert preference to a tensor
                past_preference = past_comparison["preference"].clone().detach()

                loss = criterion(past_prob_a, past_preference)

                # total_loss Example calculation:
                # total_loss = (w1 * historical_loss1 + w2 * historical_loss2 + ... + 1.0 * current_loss) / (w1 + w2 + ... + 1.0)
                total_loss += time_weight * loss
                weight_sum += time_weight

            # Add current comparison loss
            current_loss = criterion(current_comparison['prob'], current_comparison['preference'])
            total_loss += current_loss
            weight_sum += 1.0

            # Final weighted average loss
            final_loss = total_loss / weight_sum

            # Example calculation:
            # final_loss = (
            #     (0.729 * historical_loss_oldest) +  # 0.9^2 = 0.729
            #     (0.9 * historical_loss_middle) +    # 0.9^1 = 0.9
            #     (1.0 * current_loss)                # Current weight = 1.0
            # ) / (0.729 + 0.9 + 1.0)                # Normalize by sum of weights

            return final_loss


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

    # Initialize the TimeAwarePreference object
    time_aware_preference_trainer = TimeAwarePreference(decay_rate=0.9, memory_window=10)

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
            prob_a = torch.sigmoid(score_a - score_b)
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
            ground_truth_label = torch.tensor(1.0 if user_choice == "A" else 0.0).unsqueeze(0)  # Reshaped to match prob_a_tensor
            # print(f"Ground truth label: {ground_truth_label}")
            # print(f'prob_a: {prob_a}')
            # current_comparison
            current_comparison = {
                "rest_a": rest_a,
                "rest_b": rest_b,
                'prob': prob_a, # model prediction of the preference that the user will choose A over B
                "preference": ground_truth_label,
                "timestamp": datetime.now()
            }

            # add the current comparison to the time_aware_preference_trainer
            time_aware_preference_trainer.add_comparison(
                rest_a=current_comparison["rest_a"],
                rest_b=current_comparison["rest_b"],
                prob_a=current_comparison['prob'],
                preference=current_comparison["preference"],
                timestamp=current_comparison["timestamp"]
            )

            # Train the model with the current comparison and historical comparisons
            optimizer.zero_grad()
            loss = time_aware_preference_trainer.train_with_history(
                model, 
                optimizer, 
                current_comparison
                )

            loss.backward()
            optimizer.step()
            
            print(f"Training Loss: {loss.item():.4f}")

        # Final restaurant recommendation
        print(f"\nFinal Recommendation: {current_winner}")

    # Run the training loop
    train_ranknet(model, optimizer, criterion)

    # modify the restaurant_database.csv to make is_open = 0 for all restaurants after the training is complete
    df['is_open'] = 0
    # print("\nVerifying is_open values:")
    # print(df['is_open'].value_counts())  # Should show count of 0s
    df.to_csv('restaurant_database.csv', index=False)

if __name__ == "__main__":
    main()