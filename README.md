# What is tonight's dinner


## Project Purpose
This project aims to help people like myself who spend a lot of time in deciding what to eat on online-deliverying app such as UberEasts to make a quick decision in what to eat. 

## How this program works

1. **Data Collection via Web Scraping**

      The program starts by scraping real-time restaurant data from UberEats and saving it into a database.csv file.

2. **Model Loading / Training**

      If a saved AI model exists, it will be loaded.
  
      If not, a new model will be trained using your past choices.

3. **Recommendation Flow**

      At each round, two restaurant options are shown.
      
      You choose the one you prefer at the moment.
      
      The unchosen option is replaced, and the process repeats.
      
      After 10 iterations, the final option is the one most aligned with your current craving.

## How the program learns the preference of the user's restaurant choice
This system uses **RankNet**, a neural ranking model, to predict restaurant preferences based on pairwise comparisons.

**Key Features**:
Interactive Learning: By choosing between two restaurants repeatedly, the model learns your pairwise preferences.

**Temporal Awareness**:
To adapt to changing tastes, the model incorporates the last 30 user choices into its loss function. 
This helps the system prioritize recent preferences over older ones, ensuring recommendations stay relevant over time.

## User Interface
The app uses a CustomTkinter-based GUI to allow easy interaction, showing restaurant options with relevant details and images (when available). 

Users make selections directly through the interface by clicking the option bottom A or B.

## Quick Start
1. Clone the repository or download the files

2. Install dependencies:
  In terminal
    ```bash
    pip install torch pandas customtkinter selenium pillow
    ```

3. Run the main script:
  In terminal
    ```bash
    python model.py
    ```

## Data Flow
<pre> UberEats data ├──► scrappingg.py ├──► restaurant_database.csv ├──► model.py 
                                                                     └──► ranknet_model.pth 
                                                                     └──► user_preference_history.pt 
</pre>


