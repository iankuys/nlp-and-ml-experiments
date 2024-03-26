import pymysql
from transformers import BertTokenizer, BertModel
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, tree
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz # Import Decision Tree Classifier
import graphviz

def relational_queries():
    print("1. Find the cheapest non-stop flight given airports and a date.")
    print("2. Find the flight and seat information for a customer.")
    print("3. Find all non-stop flights for an airline.")
    print("4. Find plane type that can land on the desired airport.")
    print("5. Find price for flight by name.")
    print("6. Train Bert Model with COVID Tweet dataset.")
    print("7. Find the linear regression of bmi vs blood pressure dataset.")
    print("8. Find Decision Tree of email spam dataset.")
    print("9. Find linear regression on car length vs price.")
    print("10. Find linear regression on car width vs price.")
    print("P. Quit")

def get_user_choice():
    return input("Choose a query (1-10): ")

def execute_query(choice):
    if choice == '1':
        find_cheapest_flight()
    elif choice == '2':
        find_flight_and_seat_info()
    elif choice == '3':
        find_non_stop_flights_for_airline()
    elif choice == '4':
        find_planetype_can_land()
    elif choice == '5':
        find_price_from_name()
    elif choice == '6':
        bert()
    elif choice == '7':
        data = pd.read_csv('./diabetes.csv')
        bmi_vs_bloodpressure(data)
    elif choice == '8':
        data = pd.read_csv('./emails.csv')
        get_decision_tree_model(data)
    elif choice == "9":
        car_price_prediction_carlength()
    elif choice == "10":
        car_price_prediction_carwidth()
    elif choice == 'P':
        exit()
    else:
        print("Please enter a number from 1 to 10!")

def find_cheapest_flight():

    # Get user input
    departure_airport = input("Please enter the airport code for the departure airport: ")
    destination_airport = input("Please enter the airport code for the destination airport: ")
    date = input("What is the date of the flight in yyyy-mm-dd? ")

    # Execute SQL query
    sql = """
            SELECT
                f.Flight_number AS FlightNumber,
                fa.Amount AS FareAmount
            FROM
                flights.Flight f
            JOIN
                flights.fare fa ON f.Flight_number = fa.Flight_number
            JOIN
                flights.flight_leg fl ON f.Flight_number = fl.Flight_number
            JOIN 
                flights.leg_instance leg ON f.Flight_number = fl.Flight_number
            WHERE
                fl.Leg_number = 1 -- Non-stop flight
                AND fl.Departure_airport_code = ?
                AND fl.Arrival_airport_code = ?
                AND leg.Leg_date = ?
            ORDER BY
                fa.Amount
            LIMIT 1;
            """
    cur.execute(sql, (departure_airport, destination_airport, date, ))

    results = cur.fetchall()

    if not results:
        print("No Results Found.")
        return

    # Display results
    for row in results:
        print(f'The cheapest flight is {row[0]}, and the cost is {row[1]}.')

def find_flight_and_seat_info():

    # Get user input
    customer_name = input("Please enter the customer’s name: ")

    # Execute SQL query
    sql = f"""
            SELECT 
                f.Flight_number as FlightNumber, 
                f.Seat_number as SeatNumber
            FROM
                seat_reservation f 
            WHERE
                f.Customer_name = '{customer_name}'
            """

    cur.execute(sql)

    results = cur.fetchall()

    if not results:
        print("No Results Found.")
        return

    # Display results
    for row in results:
        print(f'The flight number is {row[0]}, and the seat number is {row[1]}.')

def find_non_stop_flights_for_airline():

    airline_name = input("What is the name of the airline: ")

    # Execute SQL query
    sql = f"""
            SELECT 
                f.Flight_number 
            FROM 
                flights.flight f 
            JOIN 
                flights.flight_leg fl 
            ON 
                f.Flight_number = fl.Flight_number 
            WHERE 
                fl.Leg_number = 1 
                AND Airline = '{airline_name}' 
            """
    cur.execute(sql)

    results = cur.fetchall()

    if not results:
        print("No Results Found.")
        return
    
    print('The non-stop flights are:', end=" ")
    # Display results
    for i in range(len(results)):
        if i != len(results) - 1:
            print(f'{results[i][0]},', end=" ")
        else:
            print(f'{results[i][0]}.')

def find_planetype_can_land():

    plane_company = input("What is the name of the plane company: ")
    dest_code = input("What is the airport code of the destination: ")

    # Execute SQL query
    sql = f"""
            SELECT 
	            f.Airplane_type_name
            FROM 
                flights.airplane_type f 
            JOIN 
                flights.can_land l
            ON 
                f.Airplane_type_name = l.Airplane_type_name 
            WHERE 
                f.Company = '{plane_company}'
                AND l.Airport_code = '{dest_code}'
            """
    cur.execute(sql)

    results = cur.fetchall()

    if not results:
        print("No Results Found.")
        return

    print("Plane that can land in your desired destination is/are:", end=" ")
    # Display results
    for i in range(len(results)):
        if i != len(results) - 1:
            print(f'{results[i][0]},', end=" ")
        else:
            print(f'{results[i][0]}.')

def find_price_from_name():

    customer_name = input("Please enter customer name: ")

    # Execute SQL query
    sql = f"""
            SELECT 
                fa.Fare_code, fa.Amount 
            FROM 
                flights.seat_reservation sr 
            JOIN 
                flights.fare fa 
            ON 
                sr.Flight_number = fa.Flight_number 
            WHERE 
                sr.Customer_name = '{customer_name}'
            """
    cur.execute(sql)

    results = cur.fetchall()

    if not results:
        print("No Results Found.")
        return
    
    for row in results:
        print(f'The Fare code is {row[0]}, and the seat price is ${row[1]}.')

def bert():
    # Fix the random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Access the GPU of current machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define the tokenizer of the BERT-Base model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 100

    # Load the training dataset
    train_df = pd.read_csv("Corona_NLP_train.csv", encoding="ISO-8859-1")
    # Drop rows with missing sentiment labels
    train_df = train_df.dropna(subset=["Sentiment"])
    #only use 20 % of the dataset
    train_df, _ = train_test_split(train_df, test_size=0.2, random_state=42)


    # Load the test dataset
    test_df = pd.read_csv("Corona_NLP_test.csv", encoding="ISO-8859-1")
    # Drop rows with missing sentiment labels
    test_df = test_df.dropna(subset=["Sentiment"])

    # Tokenize training data
    encoded_train_data = tokenizer.batch_encode_plus(
        train_df["OriginalTweet"].tolist(),
        add_special_tokens=True,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_attention_mask=True,
        return_tensors="pt"
    )

    train_input_ids = encoded_train_data["input_ids"]
    train_attention_mask = encoded_train_data["attention_mask"]

    # Tokenize test data
    encoded_test_data = tokenizer.batch_encode_plus(
        test_df["OriginalTweet"].tolist(),
        add_special_tokens=True,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_attention_mask=True,
        return_tensors="pt"
    )

    test_input_ids = encoded_test_data["input_ids"]
    test_attention_mask = encoded_test_data["attention_mask"]

    label_mapping = {"Extremely Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Extremely Positive": 4}

    # Encode training labels
    train_df["Label"] = train_df["Sentiment"].map(label_mapping)

    # Encode test labels
    test_df["Label"] = test_df["Sentiment"].map(label_mapping)

    # Create DataLoader for training
    batch_size = 32
    train_data = TensorDataset(train_input_ids, train_attention_mask, torch.tensor(train_df["Label"].values))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Create DataLoader for testing
    test_data = TensorDataset(test_input_ids, test_attention_mask, torch.tensor(test_df["Label"].values))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                outputs = model(**inputs)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Get predicted values
                logits = outputs.logits
                predictions.extend(logits.argmax(dim=1).cpu().numpy())  # Assuming a classification task
        
        average_train_loss = total_loss / len(train_dataloader)
        # Calculate validation MSE
        mse = mean_squared_error(test_df["Label"].values, predictions)

        print(f'Epoch {epoch + 1} / {num_epochs}')
        print(f'Training Loss: {average_train_loss}')
        #print(f'Validation Loss: {average_val_loss}')
        print(f'Validation MSE: {mse}')

    # Save the trained linear regression model
    torch.save(model.state_dict(), 'linear_regression_model.pth')

def bmi_vs_bloodpressure(data):

    plt.scatter(data['BMI'], data['BloodPressure'])
    plt.xlabel('BMI')
    plt.ylabel('BloodPressure')
    plt.title('BMI vs BloodPressure Scatter Plot')
 
    plt.savefig("BMI_vs_BloodPressure.png")
    plt.clf()

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = data['BMI'].values.reshape(-1,1)
    Y = data['BloodPressure'].values.reshape(-1,1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    model = None
    model = LinearRegression().fit(X_train, Y_train)
    print(f"\nR^2 score: {model.score(X_train, Y_train)}")
    print(f"Slope: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    equation = f"y = {model.coef_[0]}x + {model.intercept_}"
    print(f"The final equation that the linear regression model has learned is {equation}")
    # y = mx + b

    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    print(f"\nMean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")

    plt.scatter(data['BMI'], data['BloodPressure'])
    plt.plot(X_test, Y_pred, color='purple')
    plt.xlabel('BMI')
    plt.ylabel('BloodPressure')
    plt.title('BMI vs BloodPressure Scatter Plot with Regression Line')

    ######## Your code. Blank #5. END ########
    plt.savefig("bmi_vs_bp_regresion_line.png")
    plt.clf()


def get_decision_tree_model(data):
    # One-hot encode categorical columns
    data_encoded = pd.get_dummies(data, columns=['Email No.'])

    X = data_encoded.drop(columns=['Prediction'])
    y = data_encoded['Prediction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=y.unique().astype(str), filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

    for i in range(1, 11):
        clf_depth = DecisionTreeClassifier(max_depth=i)
        clf_depth.fit(X_train, y_train)
        y_pred_depth = clf_depth.predict(X_test)
        accuracy_depth = metrics.accuracy_score(y_test, y_pred_depth)
        print(f"Max Depth: {i}, Accuracy: {accuracy_depth}")

def car_price_prediction_carlength():
    data = pd.read_csv('./carprice.csv')

    plt.scatter(data['carlength'], data['price'])
    plt.xlabel('carlength')
    plt.ylabel('price')
    plt.title('carlength vs price Scatter Plot')
 
    plt.savefig("carlength_vs_price.png")
    plt.clf()

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = data['carlength'].values.reshape(-1,1)
    Y = data['price'].values.reshape(-1,1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    model = None
    model = LinearRegression().fit(X_train, Y_train)
    print(f"\nR^2 score: {model.score(X_train, Y_train)}")
    print(f"Slope: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    equation = f"y = {model.coef_[0]}x + {model.intercept_}"
    print(f"The final equation that the linear regression model has learned is {equation}")
    # y = mx + b

    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    print(f"\nMean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")

    plt.scatter(data['carlength'], data['price'])
    plt.plot(X_test, Y_pred, color='purple')
    plt.xlabel('carlength')
    plt.ylabel('price')
    plt.title('carlength vs price Scatter Plot with Regression Line')

    plt.savefig("carlength_vs_price_regresion_line.png")
    plt.clf()
    
def car_price_prediction_carwidth():
    data = pd.read_csv('./carprice.csv')

    plt.scatter(data['carwidth'], data['price'])
    plt.xlabel('carwidth')
    plt.ylabel('price')
    plt.title('carwidth vs price Scatter Plot')
 
    plt.savefig("carwidth_vs_price.png")
    plt.clf()

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = data['carwidth'].values.reshape(-1,1)
    Y = data['price'].values.reshape(-1,1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    model = None
    model = LinearRegression().fit(X_train, Y_train)
    print(f"\nR^2 score: {model.score(X_train, Y_train)}")
    print(f"Slope: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    equation = f"y = {model.coef_[0]}x + {model.intercept_}"
    print(f"The final equation that the linear regression model has learned is {equation}")
    # y = mx + b

    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    print(f"\nMean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")

    plt.scatter(data['carwidth'], data['price'])
    plt.plot(X_test, Y_pred, color='purple')
    plt.xlabel('carwidth')
    plt.ylabel('price')
    plt.title('carwidth vs price Scatter Plot with Regression Line')

    plt.savefig("carwidth_vs_price_regresion_line.png")
    plt.clf()

def whitespace():
    print("\n")

if __name__ == "__main__":

    db = pymysql.connect(host='localhost',
                    user='mp',
                    passwd= 'eecs118',
                    db= 'flights')
    condition = True
    cur = db.cursor()

    while condition:
        relational_queries()
        execute_query(get_user_choice())
        whitespace()

    db.close()
