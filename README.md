# what_is_tonights_dinner


## Project Purpose
This project aims to help people like myself who spend a lot of time in deciding what to eat on online-deliverying app such as UberEasts to make a quick decision in what to eat. 

## How this program works
Firstly, this program will scrape the UberEats data from the website and store them into the database.csv file.

Secondly, the program will read the exisiting AI model if it exists.

Thirdly, the program will provide 2 recommendations each round based on the user past restaurant choice preference for the user to choose which restaurant he or she prefer to eat at the moment. The restaurant chosen will remain as the potential option and then the model will pick the current highest potential option for the user to compare again. The process will repeat 10 times, and the last restaurant stayed in the option will be the restaurant that the user wants to eat the most.

## How the program learns the preference of the user's restaurant choice
The model uses Ranknet to predict user's preference in restaurants by providing the two highest probabilities of the restaurants that the user may want to eat. Then, the Ranknet model will take the feedback from the user to adjust the model's weight.

Also, as the user's preference in food categories may change over time, the model has implemented the time awareness function, adding the history of last 30 choices into the calculation of model's loss. This will ensure that model is focusing more on learning the user's recent preference than the old preference in food options. 

