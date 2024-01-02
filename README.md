# ai_project2024
Final project for Artificial Intelligence and Machine Learning 2023/2024

by Gianni Wu,275641

# Introduction


In this project we aim to understand the satisfaction of the customers of a train company without the need of a direct evaluation. To accomplish this task, we will study the provided dataset, “trains_dataset.csv”.

The dataset contains many customer ratings and we hope to predict whether they are satisfied or not, as such this assignment is a regression problem. As such for this assignment we will use logistic regression,KNN-Neighbors and XG-Boost. To do this we shall follow the OSEMN method.

Understanding the customers' satisfaction will help the marketing team to effectively target users with promotions and making the retention higher.

# Planning

The dataset has the following features:

• *Satisfied* : whether the customer is satisfied.
• *Onboard General Rating* : rating from 0 to 5 about the service on board.
• *Work or Leisure* : was the travelling for work or leisure.
• *Baggage Handling Rating* : rating from 0 to 5 about the handling of the baggage.
• *Age* : the age of the customer.
• *Cleanliness Rating* : rating from 0 to 5 about the cleanliness of the train.
• *Ticket Class* : the class of the ticket.
• *Loyalty* : is part of a loyalty program?
• *Food'n'Drink Rating* : rating from 0 to 5 about the food and bevarages on board.
• *Gender* : whether male or female.
• *Online Booking Rating* : rating from 0 to 5 about the online booking experience.
• *Ticket ID* : unique ID assigned to the travel ticket.
• *Onboard Service Rating* : rating from 0 to 5 about the service onboard.
• *Legroom Service Rating* : rating from 0 to 5 about the space for the legs.
• *Arrival Delay in Minutes* : the delay on the arrival of the train.
• *Departure Delay in Minutes* : the delay on the departure of the train.
• *Checkin Rating* : rating from 0 to 5 about the checkin experience.
• *Onboard Entertainment Rating* : rating from 0 to 5 about the onboard entertainment
experience.
• *Distance* : the distance of the specific travel.
• *Boarding Rating* : rating from 0 to 5 about the boarding.
• *Onboard WiFi Rating* : rating from 0 to 5 about the WiFi service.
• *Date and Time* : of the travel.
• *Seat Comfort Rating* : rating from 0 to 5 about the comfort of the seating.
• *Track Location Rating* : rating from 0 to 5 about the track where the train has been boarded.
• *Departure Arrival Time Rating* : rating from 0 to 5 about the timing of the travel.

In total there are 25 features, of which 14 are ratings from 1 to 5. The non-rating values are Satisfied, Work or Leisure, Age, Ticket Class, Loyalty, Gender, Ticket ID, Arrival Delay in Minutes , Departure Delay in Minutes, Distance and Date and Time.

The target variable is a binary class, so our task is a binary classification problem and therefore, we can use classification algorithms like logistic regression,KNN-Neighbors and XG-Boost. But in order to use these algorithms we need to prepare our data by cleaning it.

As such I looked for duplicates, but since the only unique variable was Ticker ID I simply set is as index, since if it wasn't unique Python would have warned me; I was able to set it as index and therefore every instance in the dataset is unique. Next I looked at missing values and found out that only 0.3% of my dataset had one 1 variable missing. Since the missing values were so small I chose to simply delete instances with them instead of imputing them. Next I looked for outliers, but because every variable beside "Age","Arrival Delay in Minutes","Distance","Departure Delay in Minutes" were finite classes, I chose to only study these 4 features using boxplot.
![histplot](https://github.com/uhuybubb/ai_project2024/blob/main/hist.png?raw=true)

![boxplot](https://github.com/uhuybubb/ai_project2024/blob/main/output.png?raw=true)

From the boxplots we can see that "Arrival Delay in Minutes","Distance","Departure Delay in Minutes" have a lot of outliers, moreover with the histograms we can also see that they are heavily skewed to the right, meaning that there are many upper outliers. But looking at the skewness of "Departure" and "Arrival" I am afraid that if I drop their Outliers a big part of my dataset will be lost, so instead I hot-encoded them into a binary class which shows which customers had a delayed train or departure.





