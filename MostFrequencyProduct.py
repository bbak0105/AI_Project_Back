
# def getData(inputValues, uploadedFile):
def getData():
    ##############################################################
    #################### Data Preparation
    ##############################################################
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    data = pd.read_csv("content/Historical_Product_Demand.csv")
    # data = pd.read_csv("./uploadFiles/{}".format(uploadedFile))

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

    # ProductCode Based Analysis
    productCodeBasedDF = data[["OrderDemand", 'Warehouse', 'ProductCode', 'ProductCategory']] \
        .groupby(["ProductCode", "Warehouse", "ProductCategory"]) \
        .sum().reset_index().sort_values(by=['OrderDemand'], ascending=False)

    print(productCodeBasedDF)

    # ProductCode Based Analysis
    productCodeJSON = {
        "index": data['ProductCode'].value_counts().index.tolist(),
        "count": pd.Series.tolist(data["ProductCode"].value_counts())
    }
    print(productCodeJSON)

getData()