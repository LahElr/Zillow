import pickle
from typing import Any, List, Tuple, Union
import numpy
import pandas
import os
import gc
import utils

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor

data_path = utils.read_config("data.path")
processed_data_path = utils.read_config("data.processed_path")

assert os.path.exists(data_path), "The data path does not exist!"

if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)

r"""
functions loading data
"""


def load_train_data(year: int) -> pandas.DataFrame:
    if 0 <= year < 100:
        year = year + 2000
    data_file = os.path.join(data_path, f"train_{year}.csv")
    df = pandas.read_csv(data_file)
    return df


def load_property_data(year: int) -> pandas.DataFrame:
    if 0 <= year < 100:
        year = year + 2000
    data_file = os.path.join(data_path, f"properties_{year}.csv")
    df = pandas.read_csv(
        data_file,
        dtype={
            "propertycountylandusecode": str,  # category
            "propertyzoningdesc": str,
            "hashottuborspa": str,  # bool
            "fireplaceflag": str,  # bool
            "taxdelinquencyflag": str,  # bool
            "censustractandblock": str,
            "rawcensustractandblock": str,
        },
    )
    return df


r"""
functions processing single column
"""


def convert_true_to_float(df: pandas.DataFrame, col: str):
    df.loc[df[col] == "true", col] = "1"
    df.loc[df[col] == "Y", col] = "1"
    df[col] = df[col].astype(float)


category_definations = {
    "heatingorsystemtypeid": dict(zip(range(1, 26), range(0, 25))),  #
    "propertylandusetypeid": dict(
        zip(
            [
                31,
                46,
                47,
                246,
                247,
                248,
                260,
                261,
                262,
                263,
                264,
                265,
                266,
                267,
                268,
                269,
                270,
                271,
                273,
                274,
                275,
                276,
                279,
                290,
                291,
            ],  # lahelr: I don't know why formatter chose to make it vertical
            range(0, 25),
        )
    ),  #
    "storytypeid": dict(zip(range(1, 36), range(0, 35))),
    "airconditioningtypeid": dict(zip(range(1, 14), range(0, 13))),  #
    "architecturalstyletypeid": dict(zip(range(1, 28), range(0, 27))),  #
    "typeconstructiontypeid": dict(zip(range(1, 19), range(0, 18))),  #
    "buildingclasstypeid": dict(zip(range(1, 6), range(0, 5))),  #
    "regionidcounty": {
        2061: 0,
        3101: 1,
        1286: 2,
    },  # special values appeared in training data
    "fips": {6111: 0, 6037: 1, 6059: 2},  # special values appeared in training data
    # "propertyzoningdesc_id"
    # "propertycountylandusecode_id"
}


def convert_float_to_category(df: pandas.DataFrame, col: str):
    def category_map(x):
        try:
            return category_definations[col][x]
        except KeyError:
            if numpy.isnan(x):
                return -1
            else:
                return len(category_definations[col])

    df[col] = df[col].map(category_map)
    df[col] = df[col].astype(int).astype("category")


def convert_float_to_norm_cat(df: pandas.DataFrame, col: str):
    df[col] = df[col] - df[col].min()
    df.loc[df[col].isnull(), col] = -1
    df[col] = df[col].astype(int).astype("category")


def convert_category_to_one_hot(df: pandas.DataFrame, nan_as_category: bool = True):
    original_columns = list(df.columns)  # copy
    categorical_columns = [col for col in df.columns if df[col].dtype == "category"]
    df = pandas.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


class bin_count:
    r"""
    this class aims at calculating the majority of one column
    """

    def __init__(self):
        self.cluster_count = 3
        self.clusterer = KMeans(
            n_clusters=self.cluster_count, random_state=0, max_iter=10
        )  # only 10 iters is allowed, but in practice, it's still very slow

    def major(self, col):
        col = col.values  # to_numpy()
        col = col[~numpy.isnan(col)]  # filter nan away
        if 0 < len(col) <= self.cluster_count:
            # if there's no enough points to be clustered
            return (col[0] + col[-1]) / 2
        elif len(col) == 0:
            return numpy.nan  # if nothing left
        self.clusterer.fit(
            numpy.array(col).reshape(-1, 1)
        )  # may raise ConvergenceWarning due to duplicate points, but the result would be correct
        ret = self.clusterer.cluster_centers_.reshape(-1)  # the cluster centers
        ret_ind = numpy.argmax(
            numpy.bincount(self.clusterer.labels_)
        )  # the cluster with most points
        return ret[ret_ind]


def merge_columns(
    df: pandas.DataFrame,
    col1: str,
    col2: str,
    new_name: str,
    merge_method: str = "max",
    drop: bool = False,
    convert_null: Any = None,
) -> None:
    if convert_null:
        df["_col1"] = df[col1]
        df["_col2"] = df[col2]
        o_col1, o_col2 = col1, col2
        col1 = "_col1"
        col2 = "_col2"
        df.loc[df[col1].isnull(), col1] = convert_null
        df.loc[df[col2].isnull(), col2] = convert_null
    if merge_method == "max":
        df[new_name] = df[[col1, col2]].max(axis=1)
    elif merge_method == "min":
        df[new_name] = df[[col1, col2]].min(axis=1)
    elif merge_method == "mean":
        df[new_name] = df[[col1, col2]].mean(axis=1)
    elif merge_method == "sum":
        df[new_name] = df[col1] + df[col2]
    elif merge_method == "dif":
        df[new_name] = df[col1] - df[col2]
    else:
        raise ValueError(
            f"merge_columns: expected `merge_method` to be either 'min', 'max', 'mean', 'sum' or 'dif', but got {merge_method}."
        )
    if convert_null:
        df.drop([col1, col2], axis=1, inplace=True)
        col1, col2 = o_col1, o_col2
        df.loc[df[new_name] == convert_null, new_name] = numpy.nan
    if drop:
        if new_name == col1:
            df.drop([col2], axis=1, inplace=True)
        elif new_name == col2:
            df.drop([col1], axis=1, inplace=True)
        else:
            df.drop([col1, col2], axis=1, inplace=True)


def merge_aggregate_feature(df: pandas.DataFrame, group_col: str, agg_cols):
    df[f"{group_col}_groupcnt"] = df[group_col].map(df[group_col].value_counts())
    new_columns = []  # New feature columns added to the DataFrame

    for col in agg_cols:
        aggregates = df.groupby(group_col, as_index=False)[col].agg([numpy.mean])
        aggregates.columns = [group_col, f"{group_col}_{col}_mean"]
        # new_columns += list(aggregates.columns[1])
        new_columns += [f"{group_col}_{col}_mean"]
        df = pandas.merge(df, aggregates, how="left", on=group_col)

    for col in agg_cols:
        mean = df[f"{group_col}_{col}_mean"]
        diff = df[col] - mean
        df[f"{group_col}_{col}_diff"] = diff
        if col != "year_built":
            df[f"{group_col}_{col}_percent"] = diff / mean

    # Set the values of the new features to NaN if the groupcnt is too small (prevent overfitting)
    threshold = 100
    df.loc[df[f"{group_col}_groupcnt"] < threshold, new_columns] = numpy.nan

    # Drop the mean features since they turn out to be not useful
    df.drop([f"{group_col}_{col}_mean" for col in agg_cols], axis=1, inplace=True)

    gc.collect()
    return df


class AreaInfoCalculater:
    def __init__(self):
        with open("./models/position_clusterer.pkl", "rb") as pkl_file:
            self.clusterer = pickle.load(pkl_file)

    def __call__(self, latitude: float, longitude: float) -> Tuple[float, float]:
        position = numpy.array([[latitude, longitude]])
        try:
            area_id = self.clusterer.predict(position)
        except ValueError:
            return pandas.Series([numpy.nan, numpy.nan])
        area_id = area_id[0].astype(numpy.int32)

        cluster_center = self.clusterer.cluster_centers_[area_id]

        center_distence = numpy.linalg.norm(position[0] - cluster_center)
        center_distence = center_distence.astype(numpy.float32)

        return pandas.Series([area_id, center_distence])


r"""
functions mapping whole data frame
"""


def property_boolean_convert(df: pandas.DataFrame) -> None:
    for col in ["hashottuborspa", "fireplaceflag", "taxdelinquencyflag"]:
        convert_true_to_float(df, col)
    gc.collect()


def property_float32_convert(df: pandas.DataFrame) -> None:
    for col in df.columns:
        if df[col].dtype.name == "float64":
            df[col] = df[col].astype(numpy.float32)
    gc.collect()


def train_date_convert(df: pandas.DataFrame) -> None:
    dt = pandas.to_datetime(df.transactiondate).dt
    df["year"] = (dt.year - 2016).astype(int)
    df["month"] = (dt.month).astype(int)
    df["quarter"] = (dt.quarter).astype(int)
    df.drop(["transactiondate"], axis=1, inplace=True)


def property_category_convert(df: pandas.DataFrame) -> None:
    for col in category_definations.keys():
        convert_float_to_category(df, col)
    gc.collect()


r"""
data process
"""


def add_datetime_aggregate_features(df: pandas.DataFrame) -> None:
    r"""Special process to dates"""
    # Add temporary year/month/quarter columns
    dt = pandas.to_datetime(df["transactiondate"]).dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["quarter"] = dt.quarter

    # Median logerror for each year/month/quarter
    logerror_year = (
        df.groupby("year")["logerror"]
        .median()
        .to_frame()
        .rename(index=str, columns={"logerror": "logerror_year"})
    )
    logerror_month = (
        df.groupby("month")["logerror"]
        .median()
        .to_frame()
        .rename(index=str, columns={"logerror": "logerror_month"})
    )
    logerror_quarter = (
        df.groupby("quarter")["logerror"]
        .median()
        .to_frame()
        .rename(index=str, columns={"logerror": "logerror_quarter"})
    )

    logerror_year.index = logerror_year.index.map(int)
    logerror_month.index = logerror_month.index.map(int)
    logerror_quarter.index = logerror_quarter.index.map(int)

    #     # Drop the temporary columns
    #     df.drop(["year", "month", "quarter"], axis=1, errors="ignore", inplace=True)

    #     return logerror_year, logerror_month, logerror_quarter

    # def add_datetime_aggregate_features(df, logerror_year, logerror_month, logerror_quarter):
    #     # Add temporary year/month/quarter columns
    #     dt = pandas.to_datetime(df["transactiondate"]).dt
    #     df['year'] = dt.year
    #     df['month'] = dt.month
    #     df['quarter'] = dt.quarter

    # Join the aggregate features
    df = df.merge(how="left", right=logerror_year, on="year")
    df = df.merge(how="left", right=logerror_month, on="month")
    df = df.merge(how="left", right=logerror_quarter, on="quarter")

    # Drop the temporary columns
    df.drop(
        ["year", "month", "quarter", "transactiondate"],
        axis=1,
        errors="ignore",
        inplace=True,
    )
    return df


def KNN_aggregate(
    df: pandas.DataFrame, target_column: str, reference_columns: Union[str, List[str]]
):
    r"""This function will use KNN regressor to calculate the neighbor features"""
    assert type(target_column) not in (
        list,
        tuple,
    ), "Target cannot be multiple columns!"
    columns = []

    def __add_columns(cols):
        if type(cols) is list or type(cols) is tuple:
            columns.extend(list(cols))
        else:
            columns.append(cols)

    __add_columns(target_column)
    __add_columns(reference_columns)

    data = df[columns]
    data = data.dropna(axis=0, how="any")
    x = data[reference_columns].to_numpy().astype(numpy.float32)
    y = data[target_column].to_numpy().astype(numpy.float32)

    neighbor_regressor = KNeighborsRegressor(n_neighbors=5, weights="distance")
    neighbor_regressor.fit(x, y)

    def knn_calculate(*values):
        values = numpy.array([values])
        if numpy.isnan(values).any():
            return numpy.nan
        ret = neighbor_regressor.predict(values)
        return ret[0]

    def column_map(v):
        if type(reference_columns) in (list, tuple):
            ret = knn_calculate(*[df[_] for _ in reference_columns])
        else:
            ret = knn_calculate(v)
        return ret

    new_name = f"neighbor_mean_{target_column}_wrt_{'_'.join(reference_columns) if type(reference_columns) in (list, tuple) else reference_columns}"
    t = df[reference_columns]

    df[new_name] = df.apply(column_map)
    gc.collect()


r"""Special process to zoing codes and landuse codes
"""


def get_zoning_desc_code_df(
    property16: pandas.DataFrame, property17: pandas.DataFrame
) -> pandas.DataFrame:
    temp = property16.groupby("propertyzoningdesc")[
        "propertyzoningdesc"
    ].count()  # get counts for each discrete values
    zoning_codes = list(
        temp[temp >= 5000].index
    )  # filter out indices of codes with more than 5000 items, and those with no more than 5000 items become "others"
    temp = property17.groupby("propertyzoningdesc")["propertyzoningdesc"].count()
    zoning_codes += list(temp[temp >= 5000].index)

    zoning_codes = list(set(zoning_codes))
    df_zoning_codes = pandas.DataFrame(
        {
            "propertyzoningdesc": zoning_codes,
            "propertyzoningdescid": range(len(zoning_codes)),
        }
    )
    return df_zoning_codes


def get_landuse_code_df(
    property16: pandas.DataFrame, property17: pandas.DataFrame
) -> pandas.DataFrame:
    temp = property16.groupby("propertycountylandusecode")[
        "propertycountylandusecode"
    ].count()
    landuse_codes = list(temp[temp >= 300].index)
    temp = property17.groupby("propertycountylandusecode")[
        "propertycountylandusecode"
    ].count()
    landuse_codes += list(temp[temp >= 300].index)

    landuse_codes = list(set(landuse_codes))
    df_landuse_codes = pandas.DataFrame(
        {
            "propertycountylandusecode": landuse_codes,
            "propertycountylandusecodeid": range(len(landuse_codes)),
        }
    )
    return df_landuse_codes


def property_landuse_zoning_code_convert(
    property16: pandas.DataFrame, property17: pandas.DataFrame
) -> None:
    df_zoning_codes = get_zoning_desc_code_df(property16, property17)
    property16 = pandas.merge(
        property16, df_zoning_codes, how="left", on="propertyzoningdesc"
    )
    property17 = pandas.merge(
        property17, df_zoning_codes, how="left", on="propertyzoningdesc"
    )
    property16.drop(["propertyzoningdesc"], axis=1, inplace=True)
    property17.drop(["propertyzoningdesc"], axis=1, inplace=True)

    df_landuse_codes = get_landuse_code_df(property16, property17)
    property16 = pandas.merge(
        property16, df_landuse_codes, how="left", on="propertycountylandusecode"
    )
    property17 = pandas.merge(
        property17, df_landuse_codes, how="left", on="propertycountylandusecode"
    )
    property16.drop(["propertycountylandusecode"], axis=1, inplace=True)
    property17.drop(["propertycountylandusecode"], axis=1, inplace=True)

    for col in ["propertycountylandusecodeid", "propertyzoningdescid"]:
        convert_float_to_norm_cat(property16, col)
        convert_float_to_norm_cat(property17, col)

    gc.collect()
    return property16, property17


def property_feature_engineer(df: pandas.DataFrame) -> pandas.DataFrame:
    df["average_garage_size"] = df["garagetotalsqft"] / df["garagecarcnt"]

    # these 2 columns have almost same value and identical description, don't know why
    merge_columns(
        df,
        "bathroomcnt",
        "calculatedbathnbr",
        "bathroom_count",
        "max",
        drop=True,
        convert_null=-1,
    )
    # identical description, but several different values (72 items in train data)
    merge_columns(
        df,
        "finishedsquarefeet50",
        "finishedfloor1squarefeet",
        "finishedsquarefeet50",
        "max",
        drop=True,
        convert_null=-1,
    )

    # "censustractandblock" and "rawcensustractandblock", same meaning, 14 digits and 8 digits, str
    def census_letter_half_map(x):
        try:
            return int(x[8:])
        except (TypeError, ValueError, IndexError):
            return float("nan")

    df["census_track_letter_half"] = df["censustractandblock"].map(
        census_letter_half_map
    )
    df.rename(
        columns={"rawcensustractandblock": "census_track_front_half"}, inplace=True
    )
    df["census_track_letter_half"] = df["census_track_letter_half"].astype(
        numpy.float32
    )
    df["census_track_front_half"] = df["census_track_front_half"].astype(numpy.float32)
    df["censustractandblock"] = df["censustractandblock"].astype(numpy.float32)

    merge_columns(
        df,
        "bathroom_count",
        "threequarterbathnbr",
        "full_bath_room_count",
        "dif",
        drop=False,
        convert_null=None,
    )

    area_info_calculater = AreaInfoCalculater()
    df[["area_id", "area_center_distance"]] = df.apply(
        lambda x: area_info_calculater(x["latitude"], x["longitude"]), axis=1
    )

    df["structure_tax_value_ratio"] = (
        df["structuretaxvaluedollarcnt"] / df["taxvaluedollarcnt"]
    )
    df["land_tax_value_ratio"] = df["landtaxvaluedollarcnt"] / df["taxvaluedollarcnt"]

    KNN_aggregate(df, "taxvaluedollarcnt", ["latitude", "longitude"])
    KNN_aggregate(df, "finishedsquarefeet15", ["latitude", "longitude"])
    KNN_aggregate(df, "yearbuilt", ["latitude", "longitude"])
    KNN_aggregate(df, "roomcnt", ["latitude", "longitude"])

    # feature engineer following reference
    df["derived_room_cnt"] = df["bedroomcnt"] + df["bathroom_count"]

    df["property_tax_per_sqft"] = df["taxamount"] / df["calculatedfinishedsquarefeet"]

    # Rotated Coordinates
    df["location_1"] = df["latitude"] + df["longitude"]
    df["location_2"] = df["latitude"] - df["longitude"]
    df["location_3"] = df["latitude"] + 0.5 * df["longitude"]
    df["location_4"] = df["latitude"] - 0.5 * df["longitude"]

    # > 'finished_area_sqft' and 'total_area' cover only a strict subset of 'finished_area_sqft_calc' in terms of
    # > non-missing values. Also, when both fields are not null, the values are always the same.
    # > So we can probably drop 'finished_area_sqft' and 'total_area' since they are redundant
    # > If there're some patterns in when the values are missing, we can add two isMissing binary features
    df["missing_finished_area"] = (
        df["finishedsquarefeet12"].isnull().astype(numpy.float32)
    )
    df["missing_total_area"] = df["finishedsquarefeet15"].isnull().astype(numpy.float32)
    # df.drop(['finishedsquarefeet12', 'finishedsquarefeet15'], axis=1, inplace=True)
    df["missing_bathroom_cnt_calc"] = (
        df["bathroom_count"].isnull().astype(numpy.float32)
    )

    # Average area in sqft per room
    mask = df["roomcnt"] >= 1  # avoid dividing by zero
    df.loc[mask, "avg_area_per_room"] = (
        df.loc[mask, "calculatedfinishedsquarefeet"] / df.loc[mask, "roomcnt"]
    )

    # Use the derived room_cnt to calculate the avg area again
    mask = df["derived_room_cnt"] >= 1
    df.loc[mask, "derived_avg_area_per_room"] = (
        df.loc[mask, "calculatedfinishedsquarefeet"] / df.loc[mask, "derived_room_cnt"]
    )

    df = merge_aggregate_feature(
        df,
        "regionidzip",
        [
            "lotsizesquarefeet",
            "yearbuilt",
            "calculatedfinishedsquarefeet",
            "structuretaxvaluedollarcnt",
            "landtaxvaluedollarcnt",
            "taxamount",
            "property_tax_per_sqft",
        ],
    )

    return df


if __name__ == "__main__":
    utils.logger.info("Started.")
    property16 = load_property_data(16)
    property17 = load_property_data(17)

    utils.logger.info("Property data loaded.")
    # data type conversion following reference
    property_boolean_convert(property16)
    property_category_convert(property16)
    property_boolean_convert(property17)
    property_category_convert(property17)
    property16, property17 = property_landuse_zoning_code_convert(
        property16, property17
    )
    property_float32_convert(property16)
    property_float32_convert(property17)

    utils.logger.info("Type conversion and primary processing finished.")
    # print(property16.head())
    # print(property17.head())

    property16 = property_feature_engineer(property16)
    property17 = property_feature_engineer(property17)

    utils.logger.info("Property data engineering finished.")

    property16.to_csv(
        os.path.join(processed_data_path, "processed_property_16.csv"), index=False
    )
    property17.to_csv(
        os.path.join(processed_data_path, "processed_property_17.csv"), index=False
    )
    with open(
        os.path.join(processed_data_path, "processed_property_dtype.pkl"), "wb"
    ) as pkl_file:
        pickle.dump(
            {col: property16[col].dtype for col in property16.columns}, pkl_file
        )

    train16 = load_train_data(16)
    train17 = load_train_data(17)

    utils.logger.info("Train data loaded.")

    train16 = pandas.merge(train16, property16, how="left", on="parcelid")
    train17 = pandas.merge(train17, property17, how="left", on="parcelid")

    train = pandas.concat([train16, train17], axis=0, ignore_index=True)
    train["propertylandusetypeid"] = train["propertylandusetypeid"].astype("category")
    # it seems after merging, the dtype changed to int32 for unknown reason

    train_date_convert(train)

    utils.logger.info("Preprocessing finished.")

    train.to_csv(os.path.join(processed_data_path, "processed_data.csv"), index=False)
    train.head(10).to_csv(
        os.path.join(processed_data_path, "processed_data_example.csv"), index=False
    )
    with open(
        os.path.join(processed_data_path, "processed_data_dtype.pkl"), "wb"
    ) as pkl_file:
        pickle.dump({col: train[col].dtype for col in train.columns}, pkl_file)

    utils.logger.info("Saved processed data.")
