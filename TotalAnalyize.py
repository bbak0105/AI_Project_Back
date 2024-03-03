
def getTotalAnalyizeList(uploadedFile):
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