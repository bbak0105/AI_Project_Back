from pulp import *

def getLSTMData(inputValues, uploadedFile):
    ##############################################################
    #################### Data Preparation
    ##############################################################
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    data = pd.read_csv("./uploadFiles/{}".format(uploadedFile))

    # rename the columns
    data.rename(columns={'Product_Code': 'ProductCode',
                         'Product_Category': 'ProductCategory',
                         'Order_Demand': 'OrderDemand'}, inplace=True)
    data.head()

    # check the null data
    data.isnull().sum()  # Date 11239

    # drop the missing values, we can not fill the date so best way drop missing samples
    data.dropna(inplace=True)

    # check the null data again
    data.isnull().sum()

    # sort the data according to date column
    data.sort_values('Date', ignore_index=True, inplace=True)

    # str를 위해 str으로 일단 변경
    data['OrderDemand'] = data['OrderDemand'].astype('str')

    # there are () int the OrderDemand column and we need to remove them
    data['OrderDemand'] = data['OrderDemand'].str.replace('(', "")
    data['OrderDemand'] = data['OrderDemand'].str.replace(')', "")

    # change the dtype as int64
    data['OrderDemand'] = data['OrderDemand'].astype('int64')

    # convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # create Year, Month, Day columns
    data['Year'] = data["Date"].dt.year
    data['Month'] = data["Date"].dt.month
    data['Day'] = data["Date"].dt.day


    # ##############################################################
    # #################### Forecast the Order Demand with LSTM Model
    # ##############################################################

    # for LSTM model
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    from collections import Counter
    import math

    temp_data = data.copy()
    df = temp_data[(temp_data['ProductCode'] == 'Product_1359')]
    df['OrderDemand'] = df['OrderDemand'] / 1000  # 단위를 /1000 적용해 줄여줌

    # convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # create Year, Month, Day columns
    df['Month'] = df["Date"].dt.strftime('%Y-%m')

    df = df.groupby("Month")['OrderDemand'].sum().reset_index()
    df = df.iloc[0:60]

    # [모델 훈련]
    # Create new data with only the "OrderDemand" column
    orderD = df.filter(["OrderDemand"])

    # Convert the dataframe to a np array
    orderD_array = orderD.values

    # See the train data len
    train_close_len = math.ceil(len(orderD_array) - 6)  # 마지막 6개월은 test data로 사용하기 위해 -6 적용

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(orderD_array)

    # Create the training dataset
    train_data = scaled_data[0: train_close_len, :]

    # Create X_train and y_train
    X_train = []
    y_train = []

    for i in range(12, len(train_data) - 6):  # 과거 6~18개월 전 12개월간의 데이터를 기반으로 추정하기 위해 12 적용
        X_train.append(train_data[i - 12: i, 0])
        y_train.append(train_data[i + 6, 0])

    # make X_train and y_train np array
    X_train, y_train = np.array(X_train), np.array(y_train)

    # reshape the data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # create the testing dataset
    test_data = scaled_data[train_close_len - 18:, :]
    # 과거 6~18개월 전 12개월간의 데이터 6세트 만들기 위해 24개월치 테스트 데이터 필요(60-(54-18)=24) -18 적용)

    # create X_test
    X_test = []
    for i in range(12, len(test_data) - 6):  # 12개월치씩 6개 데이터 세트 만들기
        X_test.append(test_data[i - 12: i, 0])

    # convert the test data to a np array and reshape the test data
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # ##############################################################
    # #################### Build a Optimized LSTM Model
    # ##############################################################

    # change the parameters of first LSTM model and build the Optimized LSTM Model
    optimized_model = Sequential()

    optimized_model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))

    optimized_model.add(LSTM(256, activation='relu', return_sequences=False))

    optimized_model.add(Dense(128))

    optimized_model.add(Dense(64))

    optimized_model.add(Dense(32))

    optimized_model.add(Dense(1))

    # compile the model
    optimized_model.compile(optimizer="Adam", loss="mean_squared_error", metrics=['mae'])

    # train the optimized model
    optimized_model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=20,
                        verbose=1)

    # Predict with optimized LSTM model
    o_predictions = optimized_model.predict(X_test)
    o_predictions = scaler.inverse_transform(o_predictions)

    # plot the data
    train = orderD[:train_close_len]
    valid = orderD[train_close_len:]
    valid["Predictions"] = o_predictions

    # 수요예측 결과데이터
    Demand_M1 = o_predictions[0]
    Demand_M2 = o_predictions[1]
    Demand_M3 = o_predictions[2]
    Demand_M4 = o_predictions[3]
    Demand_M5 = o_predictions[4]
    Demand_M6 = o_predictions[5]

    D = o_predictions

    # 변수 입력
    beg_inv = inputValues['begInv'] # 기초 재고
    min_inv = inputValues['minInv']  # 최소 유지해야하는 재고
    max_inv = inputValues['maxInv'] # 저장 가능한 최대 재고
    costs = inputValues['costs'] # 향후 6개월간 예상되는 가격

    # sense: LpMaximize or LpMinimize(default)
    LP = LpProblem(
        name="LP",
        sense=LpMinimize
    )

    # DEFINE decision variable
    # cat: category, "Continuous"(default), "Integer", "Binary"
    X1 = LpVariable(name='M1', lowBound=0, upBound=None, cat='Continuous')
    X2 = LpVariable(name='M2', lowBound=0, upBound=None)
    X3 = LpVariable(name='M3', lowBound=0, upBound=None)
    X4 = LpVariable(name='M4', lowBound=0, upBound=None)
    X5 = LpVariable(name='M5', lowBound=0, upBound=None)
    X6 = LpVariable(name='M6', lowBound=0, upBound=None)

    # OBJECTIVE function
    LP.objective = costs[0] * X1 + costs[1] * X2 + costs[2] * X3 + costs[3] * X4 + costs[4] * X5 + costs[5] * X6

    # CONSTRAINTS
    constraints = [
        beg_inv + X1 - D[0] <= max_inv,
        beg_inv + X1 + X2 - (D[0] + D[1]) <= max_inv,
        beg_inv + X1 + X2 + X3 - (D[0] + D[1] + D[2]) <= max_inv,
        beg_inv + X1 + X2 + X3 + X4 - (D[0] + D[1] + D[2] + D[3]) <= max_inv,
        beg_inv + X1 + X2 + X3 + X4 + X5 - (D[0] + D[1] + D[2] + D[3] + D[4]) <= max_inv,
        beg_inv + X1 + X2 + X3 + X4 + X5 + X6 - (D[0] + D[1] + D[2] + D[3] + D[4] + D[5]) <= max_inv,
        beg_inv + X1 - D[0] >= min_inv,
        beg_inv + X1 + X2 - (D[0] + D[1]) >= min_inv,
        beg_inv + X1 + X2 + X3 - (D[0] + D[1] + D[2]) >= min_inv,
        beg_inv + X1 + X2 + X3 + X4 - (D[0] + D[1] + D[2] + D[3]) >= min_inv,
        beg_inv + X1 + X2 + X3 + X4 + X5 - (D[0] + D[1] + D[2] + D[3] + D[4]) >= min_inv,
        beg_inv + X1 + X2 + X3 + X4 + X5 + X6 - (D[0] + D[1] + D[2] + D[3] + D[4] + D[5]) >= min_inv
    ]

    for i, c in enumerate(constraints):
        constraint_name = f"const_{i}"
        LP.constraints[constraint_name] = c

    # SOLVE model
    res = LP.solve()

    # 최소비용 도출 및 필요 주문량 결과
    Order_M1 = X1.varValue
    Order_M2 = X2.varValue
    Order_M3 = X3.varValue
    Order_M4 = X4.varValue
    Order_M5 = X5.varValue
    Order_M6 = X6.varValue
    Min_TotalCost = value(LP.objective)

    targetVariables = []
    for v in LP.variables():
        targetVariables.append({str(v): v.varValue})
    targetVariables.append({"target": str(Min_TotalCost)})

    return targetVariables
