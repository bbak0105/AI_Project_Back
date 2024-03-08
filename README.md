# LSTM을 활용한 재고 관리 웹 사이트 (백엔드)
![DataNomads](https://github.com/bbak0105/AI_Project_Front/assets/66405572/1d1423ee-e0a9-4a93-8072-aee69b1b261b)

<br/>

## 📌 Backend Skills
### Language
<a><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>

### IDE
<a><img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"/></a>

### Skills
<a><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/></a>
<a><img src="https://img.shields.io/badge/ScikitLearn-FF9900?style=for-the-badge"/></a>

<br/>

## 📌 Backend Descriptions
### `Route`
> ✏️ 플라스크에서 라우트를 설정하는 부분입니다.
> 1. get_uploaded_data : 프론트에서 엑셀을 업로드하면 해당 라우터로 보내집니다. 받은 데이터를 토대로 전역변수에 담아 저장합니다.
> 2. get_analysis_list : 업로드 된 파일을 토대로 데이터 분석을 진행한 후, 분석 데이터를 리턴합니다.
> 3. get_lstm_dat : 프론트에서 예상 재고 수량을 입력하여 받아온 데이터를 토대로 LSTM을 진행하여 최적의 재고를 예측합니다. 예측한 데이터를 리턴합니다.

```python
import flask
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PredictInventory
import TotalAnalyize

app = Flask(__name__)
CORS(app)
global uploadedFile

@app.route('/uploadData', methods=['POST'])
def get_uploaded_data():
    f = request.files['file']
    f.save("./uploadFiles/" + secure_filename(f.filename))
    global uploadedFile
    uploadedFile = f.filename
    return 'Uplaod Success!'

@app.route('/getAnalysisList', methods=["POST", "GET"])
def get_analysis_List():
    data = TotalAnalyize.getTotalAnalyizeList(uploadedFile)
    return data

@app.route('/getLSTMData', methods=["POST", "GET"])
def get_LSTM_data():
    inputValues = request.json
    data = PredictInventory.getLSTMData(inputValues, uploadedFile)
    return data

if __name__ == "__main__":
    app.run()
```

---

### `Data Analysis`
> ✏️ groupby, 빈도분석 등 기초적인 데이터를 분석하는 곳입니다. <br/>
> 기본적인 전처리 작업 이후에 데이터를 분석하여 분석된 데이터를 리턴합니다.

```python
def getTotalAnalyizeList(uploadedFile):
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
  
  # [Monthly] Analysis
  temp_data = data.copy()
  temp_data.Month.replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                          ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                          inplace=True)
  ...

  # [ProductCategory] statistical information about OrderDemand
  productCategoryJSON = {
      "index": data["ProductCategory"].value_counts().index.tolist(),
      "count": pd.Series.tolist(data["ProductCategory"].value_counts())
  }
  ...

  # Warehouse Based Analysis
  warehouseBasedDF = data[["OrderDemand", 'Year', 'Warehouse']] \
      .groupby(["Year", "Warehouse"]) \
      .sum().reset_index().sort_values(by=['Warehouse', 'Year'], ascending=False)
  
  warehouseBasedJSON = {
      "Year": warehouseBasedDF['Year'].tolist(),
      "Warehouse": warehouseBasedDF['Warehouse'].tolist(),
      "OrderDemand": warehouseBasedDF['OrderDemand'].tolist()
  }
  ...

  # Total List Return
  totalList = {
      "productCategoryJSON": productCategoryJSON,
      "warehouseJSON": warehouseJSON,
      "productCodeJSON" : productCodeJSON,
      "yearlyJSON": yearlyJSON,
      "monthlyJSON": monthlyJSON,
      "warehouseBasedJSON": warehouseBasedJSON,
      "productCategoryBasedJSON": productCategoryBasedJSON,
      "productCodeBaseJSON": productCodeBaseJSON
  }
  return totalList
```
[↑ 전체코드보기](https://github.com/bbak0105/AI_Project_Back/blob/main/TotalAnalyize.py)

---

### `Stock Prediction`
> ✏️ 프론트에서 사용자가 보내준 데이터를 바탕으로 LSTM으로 적정 재고량을 예측합니다. <br/>
> 테스트 데이터는 과거 6개월의 데이터로, 학습 데이터는 과거 6~18개월 전 12개월의 데이터로 진행합니다. <br/>
> 데이터전처리, MinMaxScaler 정규화, LSTM 모델, Adam 옵티마이저 및 mean_squared_error 손실함수를 사용하여 모델 컴파일을 진행하였습니다.

```python
# for LSTM model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from collections import Counter
import math

...
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

# change the parameters of first LSTM model and build the Optimized LSTM Model
optimized_model = Sequential()

optimized_model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
...

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
...

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
...

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
...
Min_TotalCost = value(LP.objective)

targetVariables = []
for v in LP.variables():
    targetVariables.append({str(v): v.varValue})
targetVariables.append({"target": str(Min_TotalCost)})

return targetVariables
```

[↑ 전체코드보기](https://github.com/bbak0105/AI_Project_Back/blob/main/PredictInventory.py)



