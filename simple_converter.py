import gc
import pandas as pd
from datetime import datetime


def dummy_encode(df):
    columns_to_encode = list(df.select_dtypes(include=['category', 'object']))
    df = pd.get_dummies(df,prefix=columns_to_encode, columns=columns_to_encode)
    print(df)
    return df


def get_train_test_sets(frac=0.7):
    joined_file_name = './joined_table.csv'
    try:
        joined = pd.read_csv(joined_file_name)
    except:
        properties = pd.read_csv('./data/properties_2016.csv/properties_2016.csv')
        train = pd.read_csv('./data/train_2016_v2.csv/train_2016_v2.csv')

        joined = pd.merge(train, properties, how="left", on="parcelid")

        del properties
        del train
        gc.collect()

        joined["transactiondate"] = (joined["transactiondate"]
                                     .apply(lambda x: (pd.to_datetime(str(x), format='%Y-%m-%d')
                                                       - datetime(1960, 1, 1)).total_seconds() / (3600 * 24)))
        joined.to_csv(joined_file_name, index=False)

    joined = dummy_encode(joined)

    for name, values in joined.iteritems():
        # fill in nan fields
        if name == 'taxdelinquencyflag':
            joined[name] = joined[name].replace(to_replace="Y", value=1)
            joined[name] = joined[name].fillna(0)
            joined[name] = pd.to_numeric(joined[name])
        elif name == 'hashottuborspa' or name == 'fireplaceflag':
            joined[name] = joined[name].replace(to_replace=True, value=1)
            joined[name] = joined[name].fillna(0)
            joined[name] = pd.to_numeric(joined[name])
        elif (joined[name].dtype == 'object'):
            joined[name] = joined[name].fillna("No data")
            joined[name] = joined[name].apply(hash)
            joined[name] = joined[name].astype('float32')
        else:
            joined[name] = joined[name].fillna(0)

        if joined[name].dtype == 'float64':
            joined[name] = joined[name].astype('float32')
        elif joined[name].dtype == 'int64':
            joined[name] = joined[name].astype('float32')

    training_set = joined.sample(frac=frac)


    # temp set for all the testing data
    test_set = joined.loc[~joined.index.isin(training_set.index)]

    #further split it in to validation and tseting
    validation_set = test_set.sample(frac=0.5)
    testing_set = joined.loc[~joined.index.isin(test_set.index)]

    training_set.to_csv("training.csv", index=False)
    testing_set.to_csv("testing.csv", index=False)
    validation_set.to_csv("validation.csv", index=False)


    return training_set, testing_set, validation_set
