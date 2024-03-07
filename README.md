# [서강대학원] 프랙티컴 AI 재고관리 수요예측 프로젝트 백엔드
![DataNomads](https://github.com/bbak0105/AI_Project_Front/assets/66405572/1d1423ee-e0a9-4a93-8072-aee69b1b261b)

<br/>

## 📌 Frontend Skills
### Language
<a><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>

### IDE
<a><img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"/></a>

### Skills
<a><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/></a>

<br/>

## 📌 Backend Flow
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
> ✏️ groupby, 빈도분석 등 기초적인 데이터를 분석하는 곳입니다.
> 1. get_uploaded_data : 프론트에서 엑셀을 업로드하면 해당 라우터로 보내집니다. 받은 데이터를 토대로 전역변수에 담아 저장합니다.
> 2. get_analysis_list : 업로드 된 파일을 토대로 데이터 분석을 진행한 후, 분석 데이터를 리턴합니다.

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
  
  monthlyDF = temp_data[['OrderDemand', 'Month', 'Year', ]] \
      .groupby(["Year", "Month"]) \
      .sum().reset_index().sort_values(by=['Year', 'Month'], ascending=False)
  
  monthlyJSON = {
      "Year": monthlyDF['Year'].tolist(),
      "Month": monthlyDF['Month'].tolist(),
      "OrderDemand": monthlyDF['OrderDemand'].tolist()
  }
  
  # [ProductCategory] statistical information about OrderDemand
  productCategoryJSON = {
      "index": data["ProductCategory"].value_counts().index.tolist(),
      "count": pd.Series.tolist(data["ProductCategory"].value_counts())
  }
  
  # [Warehouse] Number of samples according to Warehouse
  warehouseJSON = {
      "index": data['Warehouse'].value_counts().index.tolist(),
      "count": pd.Series.tolist(data["Warehouse"].value_counts())
  }
  
  # [ProductCode] Based Analysis
  productCodeJSON = {
      "index": data['ProductCode'].value_counts().index.tolist(),
      "count": pd.Series.tolist(data["ProductCode"].value_counts())
  }
  
  # [Yearly] Analysis
  analysisDf = data[['OrderDemand', 'Year']].groupby(["Year"]).sum().reset_index().sort_values(by='Year',
                                                                                               ascending=False)
  yearlyJSON = {
      "index": analysisDf['Year'].tolist(),
      "count": analysisDf['OrderDemand'].tolist()
  }
  
  # Warehouse Based Analysis
  warehouseBasedDF = data[["OrderDemand", 'Year', 'Warehouse']] \
      .groupby(["Year", "Warehouse"]) \
      .sum().reset_index().sort_values(by=['Warehouse', 'Year'], ascending=False)
  
  warehouseBasedJSON = {
      "Year": warehouseBasedDF['Year'].tolist(),
      "Warehouse": warehouseBasedDF['Warehouse'].tolist(),
      "OrderDemand": warehouseBasedDF['OrderDemand'].tolist()
  }
  
  # Product Category Based Analysis
  productCategoryBasedDF = data[["OrderDemand", 'ProductCategory', 'Warehouse']] \
      .groupby(["ProductCategory", "Warehouse"]) \
      .sum().reset_index().sort_values(by=['OrderDemand'], ascending=False)
  
  productCategoryBasedJSON = {
      "ProductCategory": productCategoryBasedDF['ProductCategory'].tolist(),
      "Warehouse": productCategoryBasedDF['Warehouse'].tolist(),
      "OrderDemand": productCategoryBasedDF['OrderDemand'].tolist()
  }
  
  # ProductCode Based Analysis
  productCodeBasedDF = data[["OrderDemand", 'Warehouse', 'ProductCode', 'ProductCategory']] \
      .groupby(["ProductCode", "Warehouse", "ProductCategory"]) \
      .sum().reset_index().sort_values(by=['OrderDemand'], ascending=False)
  
  productCodeBaseJSON = {
      "ProductCode": productCodeBasedDF['ProductCode'].tolist(),
      "ProductCategory": productCodeBasedDF['ProductCategory'].tolist(),
      "Warehouse": productCodeBasedDF['Warehouse'].tolist(),
      "OrderDemand": productCodeBasedDF['OrderDemand'].tolist()
  }
  
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
--- 

[↑ 전체코드보기](https://github.com/bbak0105/AI_Project_Front/blob/main/src/views/dashboard/FileUploadBox.js)

