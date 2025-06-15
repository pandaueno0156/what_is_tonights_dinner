import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from datetime import datetime
from scrapping import scrape_restaurant_data
import tkinter as tk
from tkinter import ttk
from tkinter import N, S, E, W, Tk, StringVar
from PIL import Image, ImageTk
import os
import customtkinter as ctk
import json

### ===== Prepare the restaurant data from the database ===== ###
def scrape_and_prepare_restaurant_data():

    try:
        # This function will scrape the restaurant data and save it to the database for most recent data
        scrape_restaurant_data()
    except Exception as e:
        print(f"Error: {e}")

    # read the most recent database
    df = pd.read_csv('restaurant_database.csv')

    def transform_to_restaurants(df):
        new_df = df.copy()

        # Filter out the rows where is_open is 0
        # To ensure that we only consider the restaurants that are currently open
        new_df = new_df[new_df['is_open'] == 1]
        new_df.drop(columns=['last_time_scraped','is_open', 'restaurant_address'], inplace=True)
        
        # Normalize the ratings in new_df
        new_df['normalized_restaurant_rating'] = (new_df['restaurant_rating'] - 0.0) / (5.0 - 0.0)

        features_columns = [f for f in new_df.columns if f not in ['restaurant_name', 'restaurant_rating', 'restaurant_img', 'restaurant_types']]

        output_list = [
            {
                "name": name,
                "rating": rating,
                "image_path": image_path,
                "types": types,
                "features": features.tolist()
            }
            for name, rating, image_path, types, features in zip(new_df['restaurant_name'], new_df['restaurant_rating'], new_df['restaurant_img'], new_df['restaurant_types'], new_df[features_columns].values)
        ]
        return output_list

    restaurants = transform_to_restaurants(df)

    # Convert features to tensors
    for r in restaurants:
        r["features"] = torch.tensor(r["features"], dtype=torch.float32)

    restaurants_name_feature_dict = {r["name"]: r for r in restaurants}

    return restaurants, restaurants_name_feature_dict, df

# print(restaurants_name_feature_dict)

### ===== Prepare the restaurant data from the database ===== ###


### ===== GUI for the user to select the restaurant options ===== ###
class RestaurantComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("What is tonight's dinner?")

        # Window size
        self.root.geometry("900x680")

        # Window background color
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        # self.root.configure(bg="white")

        # Create mainframe
        self.mainframe = ctk.CTkFrame(self.root)
        self.mainframe.grid(column=0, row=0, sticky="nsew", padx=20)

        # Configure the grid's column and row weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create labels for the restaurant options
        ctk.CTkLabel(self.mainframe, text="Option: Restaurant A", font=("Helvetica", 17, 'bold')).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkLabel(self.mainframe, text="Option: Restaurant B", font=("Helvetica", 17, 'bold')).grid(row=1, column=2, padx=5, pady=5)

        # Create frames for each restaurant inside the mainframe
        self.rest_a_frame = ctk.CTkFrame(self.mainframe)
        self.rest_a_frame.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="nsew")

        self.rest_b_frame = ctk.CTkFrame(self.mainframe)
        self.rest_b_frame.grid(row=2, column=2, padx=(10, 20), pady=10, sticky="nsew")

        # Create round number label
        self.round_number_label = ctk.CTkLabel(self.mainframe, font=("Helvetica", 20, 'bold'))
        self.round_number_label.grid(row=0, column=0, pady=(20, 10),columnspan=3, sticky="n")  # pady = (top, bottom)

        # Choice variable - to store the user's choice from clicking the buttons
        self.user_choice_var = ctk.StringVar()

        # Create UI elements for the restaurant display
        self.create_restaurant_display()

    def create_restaurant_display(self):
        # Restaurant A UI
        # Title
        # Image placeholder
        self.rest_a_image_placeholder_label = ctk.CTkLabel(self.rest_a_frame, text="") # empty text to hold the image
        self.rest_a_image_placeholder_label.grid(row=2, column=0, pady=10)

        # Restaurant name display
        self.rest_a_name_label = ctk.CTkLabel(self.rest_a_frame, text="Restaurant Name", font=("Helvetica", 13), wraplength=400)
        self.rest_a_name_label.grid(row=3, column=0, pady=5)

        # Restaurant rating display
        self.rest_a_rating_label = ctk.CTkLabel(self.rest_a_frame, text="Rating: ", font=("Helvetica", 13))
        self.rest_a_rating_label.grid(row=4, column=0, pady=5)

        # Restaurant types display
        self.rest_a_types_label = ctk.CTkLabel(self.rest_a_frame, text="Types: ", font=("Helvetica", 13))
        self.rest_a_types_label.grid(row=5, column=0, pady=5)

        # Restaurant B UI
        # Title

        # Image placeholder
        self.rest_b_image_placeholder_label = ctk.CTkLabel(self.rest_b_frame, text="")
        self.rest_b_image_placeholder_label.grid(row=2, column=2, pady=10)

        # Restaurant name display
        self.rest_b_name_label = ctk.CTkLabel(self.rest_b_frame, text="Restaurant Name", font=("Helvetica", 13), wraplength=400)
        self.rest_b_name_label.grid(row=3, column=2, pady=5)

        # Restaurant rating display
        self.rest_b_rating_label = ctk.CTkLabel(self.rest_b_frame, text="Rating: ", font=("Helvetica", 13))
        self.rest_b_rating_label.grid(row=4, column=2, pady=5)

        # Restaurant types display
        self.rest_b_types_label = ctk.CTkLabel(self.rest_b_frame, text="Types: ", font=("Helvetica", 13))
        self.rest_b_types_label.grid(row=5, column=2, pady=5)

        # User choice buttons display
        ctk.CTkButton(self.mainframe, text="Choose A", command=lambda: self.make_choice("A"), 
                        corner_radius=32, fg_color="transparent", 
                        hover_color="#4158D0", border_color="#FFCC70",
                        border_width=2).grid(row=3, column=0, pady=20)
        ctk.CTkButton(self.mainframe, text="Choose B", command=lambda: self.make_choice("B"), 
                        corner_radius=32, fg_color="transparent", 
                        hover_color="#4158D0", border_color="#FFCC70",
                        border_width=2).grid(row=3, column=2, pady=20)

    def make_choice(self, choice):
        self.user_choice_var.set(choice)
    
    def update_restaurant_info(self, rest_a, rest_b, round_num):

        # example of rest_a:
        # rest_a: {'name': '極上担々麺 香家 エソラ池袋店 KO-YA', 'rating': 4.4, 'image_path': 'restaurant_images/極上担々麺 香家 エソラ池袋店 KO-YA.webp', 'features': tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8800])}

        # Update round number label
        showing_round_num = round_num + 1
        self.round_number_label.configure(text=f'Round: {showing_round_num} / 10')  

        # Update Restaurant A info
        self.rest_a_name_label.configure(text=f'Name: {rest_a["name"]}')
        self.rest_a_rating_label.configure(text=f'Rating: {rest_a["rating"]}')
        self.rest_a_types_label.configure(text=f'Types: {rest_a["types"]}')

        # Update Restaurant B info
        self.rest_b_name_label.configure(text=f'Name: {rest_b["name"]}')
        self.rest_b_rating_label.configure(text=f'Rating: {rest_b["rating"]}')
        self.rest_b_types_label.configure(text=f'Types: {rest_b["types"]}')

        # Update images
        self.update_image(rest_a['image_path'], self.rest_a_image_placeholder_label)
        self.update_image(rest_b['image_path'], self.rest_b_image_placeholder_label)

    def update_image(self, image_path, image_label):
        if os.path.exists(image_path):
            image = Image.open(image_path)
            # Resize the image to 400x300 using LANCZOS resampling
            image = image.resize((400, 300), Image.Resampling.LANCZOS)

            # Convert the image to a CTkImage object
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=(400, 300))
            image_label.configure(image=photo)
            image_label.image = photo  # Keep a reference to prevent garbage collection.

### ===== GUI for the user to select the restaurant options ===== ###


### ===== TimeAwarePreference class to add the TAP function to the RankNet model ===== ###

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
            # Calculate days based on fraction of days to be more time-sensitive
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

            # Get model prediction for the past comparison
            score_a = model(past_comparison["rest_a"]["features"])
            score_b = model(past_comparison["rest_b"]["features"])
            past_prob_a = torch.sigmoid(score_a - score_b)

            # Convert preference to a tensor
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

    def save_history(self, filename):
        # Save the history when the user quits the program
        torch.save(self.history, filename)
        print()
        print(f'User preference history saved to {filename}')
    
    def load_history(self, user_preference_history):
        # Load the history when the user starts the program
        self.history = user_preference_history
        # print(f'old_loaded_history: {self.history}')

### ===== TimeAwarePreference class to add the TAP function to the RankNet model ===== ###


### ===== Define the RankNet model ===== ###

# class RankNet(nn.Module):
#     def __init__(self, input_size):
#         super(RankNet, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 16),  # Hidden layer (Original paper 50 input, 10 hidden - 1 output, 2 layers)
#             nn.ReLU(),
#             nn.Linear(16, 1)  # Single score output
#         )
    
#     def forward(self, x):
#         return self.fc(x)

class RankNet(nn.Module):
    def __init__(self, input_size):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Hidden layer (Original paper 50 input, 10 hidden - 1 output, 2 layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # Single score output

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output
### ===== Define the RankNet model ===== ###


# Helper function to predict and train
def train_ranknet(model, optimizer, criterion, rounds=10):
    remaining_restaurants = restaurants.copy()

    # Initialize the GUI
    root = ctk.CTk()
    app = RestaurantComparisonGUI(root)
    
    # Keep track of the current winner and contender with names
    current_winner = None
    current_contender = None
    
    def run_training_model():

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

                # print(f'rest_a: {rest_a}')
                # print(f'rest_b: {rest_b}')

                # Remove the selected restaurants from the remaining_restaurants list
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

                # To make sure that the restaurants exist for competing
                assert rest_a is not None and rest_b is not None, "Restaurants must exist"

                # Remove the current_contender from the remaining_restaurants list
                remaining_restaurants.remove(rest_b)

            # Update the GUI with the current restaurants
            app.update_restaurant_info(rest_a, rest_b, round_num)
            root.wait_variable(app.user_choice_var)

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

            # Interaction with the user in GUI
            user_choice = app.user_choice_var.get()

            # This round's winner is the one that the user chose
            if user_choice == "A":
                current_winner = rest_a["name"]
            else:
                current_winner = rest_b["name"]
            
            # Determine the ground truth label
            ground_truth_label = torch.tensor(1.0 if user_choice == "A" else 0.0).unsqueeze(0)  # Reshaped to match prob_a_tensor
            # print(f"Ground truth label: {ground_truth_label}")
            # print(f'before adding to current_comparison and add_comparison')
            # print(f'prob_a: {prob_a}')
            # print(f'ground_truth_label: {ground_truth_label}')
            # print(f'rest_a: {rest_a}')

            # Current comparison
            current_comparison = {
                "rest_a": rest_a,
                "rest_b": rest_b,
                'prob': prob_a, # Model prediction of the preference that the user will choose A over B
                "preference": ground_truth_label,
                "timestamp": datetime.now()
            }

            # Add the current comparison to the time_aware_preference_trainer
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

            if round_num == (rounds - 1):
                # To close the GUI after 10 rounds of decisions are made
                root.quit()
                root.destroy()
        
        # Save the history when the user quits the program
        time_aware_preference_trainer.save_history('./JP-model/user_preference_history.pt')
            
        # Final restaurant recommendation
        print(f"\nFinal Recommendation: {current_winner}")

    # Wait 100ms for GUI to initialize
    root.after(100, run_training_model)
    root.mainloop()


### ===== Initialize the RankNet model and training process===== ###

### ===== Update the restaurant_database.csv to make is_open = 0 ===== ###

def update_restaurant_database(df):
    # Modify the restaurant_database.csv to make is_open = 0 
    # for all restaurants after the training is complete

    df['is_open'] = 0
    df.to_csv('restaurant_database.csv', index=False)

### ===== Update the restaurant_database.csv to make is_open = 0 ===== ###

if __name__ == "__main__":

    restaurants, restaurants_name_feature_dict, df = scrape_and_prepare_restaurant_data()

    input_size = len(restaurants[0]["features"])  # Number of features per restaurant
    model = RankNet(input_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create the JP-model folder if it doesn't exist
    if not os.path.exists('./JP-model'):
        os.makedirs('./JP-model')

    # Load the model and optimizer from the checkpoint
    try:
        checkpoint = torch.load('./JP-model/ranknet_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Ranknet model loaded successfully')
    except:
        print('Ranknet model not found')


    # Initialize the TimeAwarePreference object
    # save 30 rounds of user preference history
    time_aware_preference_trainer = TimeAwarePreference(decay_rate=0.9, memory_window=30, time_unit="days")
    
    # Load the user preference history from the checkpoint
    try:
        user_preference_history = torch.load('./JP-model/user_preference_history.pt')
        time_aware_preference_trainer.load_history(user_preference_history)
        print('User preference history loaded successfully')
    except:
        print('User preference history not found')
        print('Starting a new user preference history')


    # train the ranknet model
    # 10 rounds of training for each session
    train_ranknet(model, optimizer, criterion, rounds=10)

    # Save the model and history after training
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                './JP-model/ranknet_model.pth')

    update_restaurant_database(df)