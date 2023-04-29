from collections import defaultdict
import pickle
import random
from catboost import CatBoostRegressor, Pool
import lightgbm
from sklearn.model_selection import train_test_split
import utils
import pandas
import numpy
import os
import matplotlib.pyplot as plt
import time


def drop_unuseful_features(df: pandas.DataFrame):
    # id and label (not features)
    unused_feature_list = ["parcelid", "logerror"]

    if utils.read_config("data.drop.missing"):
        # too many missing (LightGBM is robust against bad/unrelated features, so this step might not be needed)
        unused_feature_list.extend(
            [
                "buildingclasstypeid",
                "architecturalstyletypeid",
                "storytypeid",
                "finishedsquarefeet13",
                "basementsqft",
                "yardbuildingsqft26",
            ]
        )

    if utils.read_config("data.drop.unuseful"):
        # not useful
        unused_feature_list.extend(
            [
                "fireplaceflag",
                "decktypeid",
                "pooltypeid10",
                "typeconstructiontypeid",
                "regionidcounty",
                "fips",
            ]
        )

    if utils.read_config("data.drop.bad"):
        # really hurts performance
        unused_feature_list.extend(
            ["propertycountylandusecodeid", "propertyzoningdescid"]
        )

    return df.drop(unused_feature_list, axis=1, errors="ignore")


def param_combinations(param_space):
    r"""
    This function actually recursively dfs the whole parameter decision tree
    The reason why I traverse parameter space in a this complicated way
        is that I don't know the depth of the parameter space,
        so I can't just write a series of `for` caluses.
    """
    param_keys = list(param_space.keys())

    def __full_combine(l1, l2):
        ret = []
        for _e1 in l1:
            for _e2 in l2:
                ret.append([_e1] + _e2)
        return ret

    def __rec_search_sub_space(start_point):
        if start_point >= len(param_space) - 1:
            return [[_] for _ in param_space[param_keys[start_point]]]
        else:
            return __full_combine(
                param_space[param_keys[start_point]],
                __rec_search_sub_space(start_point + 1),
            )

    ret = __rec_search_sub_space(0)
    return [{k: v for k, v in zip(param_keys, r)} for r in ret]


def count_param_space(param_space):
    ret = 1
    for k in param_space:
        ret *= len(param_space[k])
    return ret


def transform_test_data(properties16, properties17, month=10):
    p16 = drop_unuseful_features(properties16)
    p17 = drop_unuseful_features(properties17)

    p16["year"] = 0
    p17["year"] = 1

    transform_test_data_month(p16, p17, month)

    p16["quarter"] = 4
    p17["quarter"] = 4

    return p16, p17


def transform_test_data_month(p16, p17, month=10):
    p16["month"] = month
    p17["month"] = month


def choose_test_params(param_space):
    ret = {}
    for k in param_space.keys():
        ret[k] = param_space[k][0]
    return ret


if __name__ == "__main__":
    utils.logger.info("Started...")

    numpy.random.seed(utils.read_config("seed"))
    random.seed(utils.read_config("seed"))

    with open(
        os.path.join(
            utils.read_config("data.processed_path"), "processed_data_dtype.pkl"
        ),
        "rb",
    ) as pkl_file:
        data_dtype_dict = pickle.load(pkl_file)

    train = pandas.read_csv(
        os.path.join(utils.read_config("data.processed_path"), "processed_data.csv"),
        dtype=data_dtype_dict,
    )
    utils.logger.info("Read training data finished.")

    if not utils.read_config("do_test"):
        train_features = drop_unuseful_features(train)
        utils.logger.info(f"Total feature count: {len(train_features.columns)}")
        train_target = train["logerror"].astype(numpy.float32)

        X = train_features.values
        y = train_target.values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=utils.read_config("data.train_test_ratio")
        )

        if utils.read_config("data.remove_outlier"):
            mask = abs(y) <= utils.read_config("data.outlier_threshold")
            X_train = X[mask, :]
            y_train = y[mask]

        if utils.read_config("model.name") == "catboost":
            X_train = pandas.DataFrame(
                {
                    train_features.columns[i]: pandas.Series(
                        X_train[:, i],
                        dtype=train_features.dtypes[train_features.columns[i]],
                    )
                    for i in range(len(train_features.columns))
                }
            )
            X_val = pandas.DataFrame(
                {
                    train_features.columns[i]: pandas.Series(
                        X_val[:, i],
                        dtype=train_features.dtypes[train_features.columns[i]],
                    )
                    for i in range(len(train_features.columns))
                }
            )

        utils.logger.info(
            f"X_train of {X_train.shape} shape; y_train of {y_train.shape}"
        )
        utils.logger.info(f"X_val of {X_val.shape} shape; y_val of {y_val.shape}")

        feature_names = [s for s in train_features.columns]

        categorical_indices = [
            str(col)
            for col in train_features.columns
            if train_features[col].dtype == "category"
        ]

        if utils.read_config("model.name") == "lightgbm":
            lightgbm_trainset = lightgbm.Dataset(
                X_train, label=y_train, feature_name=feature_names
            )

            lightgbm_valset = lightgbm.Dataset(
                X_val, label=y_val, feature_name=feature_names
            )

        param_space = utils.read_config(
            f"model.{utils.read_config('model.name')}_params"
        )

        param_space_size = count_param_space(param_space)

        for i, param_combination in enumerate(param_combinations(param_space)):
            utils.logger.info(
                f"Starting training {i}/{param_space_size} parameter combination..."
            )
            utils.logger.info(f"Parameter setting:")
            utils.logger.info(str(param_combination))

            if utils.read_config("model.name") == "lightgbm":
                model = lightgbm.train(
                    params=param_combination,
                    train_set=lightgbm_trainset,
                    verbose_eval=False,
                    valid_sets=[lightgbm_trainset, lightgbm_valset],
                    valid_names=["train", "val"],
                    categorical_feature=categorical_indices,
                )

                train_error = abs(model.predict(X_train) - y_train).mean()
                val_error = abs(model.predict(X_val) - y_val).mean()
                utils.save_lightgbm_model(
                    model, f"lightgbm_trainE{train_error:0.4}_valE{val_error:0.4}.model"
                )

                lightgbm.plot_importance(model)
                plt.savefig(
                    os.path.join(
                        utils.read_config("save_path"),
                        f"lightgbm_trainE{train_error:0.4}_valE{val_error:0.4}_importance.png",
                    ),
                    dpi=600,
                    bbox_inches="tight",
                )

                utils.logger.info(f"Train error: {train_error}")
                utils.logger.info(f"Val error: {val_error}")
                utils.logger.info("Model saved.")
                utils.logger.info("")

            elif utils.read_config("model.name") == "catboost":
                model = CatBoostRegressor(task_type="GPU", **param_combination)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    verbose=False,
                    cat_features=categorical_indices,
                )

                train_error = abs(model.predict(X_train) - y_train).mean()
                val_error = abs(model.predict(X_val) - y_val).mean()
                utils.save_catboost_model(
                    model, f"catboost_trainE{train_error:0.4}_valE{val_error:0.4}.model"
                )
                utils.logger.info(f"Train error: {train_error}")
                utils.logger.info(f"Val error: {val_error}")
                utils.logger.info("Model saved.")
                utils.logger.info("Now sleep 60 seconds to relif gpu load.")
                time.sleep(60)
                utils.logger.info("Wake up.")

    else:
        if utils.read_config("do_full_train_before_test"):
            train_features = drop_unuseful_features(train)
            utils.logger.info(f"Total feature count: {len(train_features.columns)}")
            train_target = train["logerror"].astype(numpy.float32)

            X = train_features.values
            y = train_target.values

            if utils.read_config("data.remove_outlier"):
                mask = abs(y) <= utils.read_config("data.outlier_threshold")
                if utils.read_config("model.name") == "catboost":
                    X_train = train_features.loc[mask, :]
                    y_train = train_target.loc[mask]
                else:
                    X_train = X[mask, :]
                    y_train = y[mask]

            utils.logger.info(
                f"X_train of {X_train.shape} shape; y_train of {y_train.shape}"
            )

            feature_names = [s for s in train_features.columns]

            categorical_indices = [
                str(col)
                for col in train_features.columns
                if train_features[col].dtype == "category"
            ]

            if utils.read_config("model.name") == "lightgbm":
                lightgbm_trainset = lightgbm.Dataset(
                    X_train, label=y_train, feature_name=feature_names
                )

            utils.logger.info(f"Starting training ...")
            utils.logger.info(f"Parameter setting:")
            param_combination = choose_test_params(
                utils.read_config(f"model.{utils.read_config('model.name')}_params")
            )
            utils.logger.info(str(param_combination))

            if utils.read_config("model.name") == "lightgbm":
                param_combination.pop("early_stopping_rounds")
                model = lightgbm.train(
                    params=param_combination,
                    train_set=lightgbm_trainset,
                    verbose_eval=False,
                    categorical_feature=categorical_indices,
                )

                lightgbm.plot_importance(model)
                plt.savefig(
                    os.path.join(
                        utils.read_config("save_path"),
                        "lightgbm_test_model_importance.png",
                    ),
                    dpi=600,
                    bbox_inches="tight",
                )
            elif utils.read_config("model.name") == "catboost":
                model = CatBoostRegressor(task_type="GPU", **param_combination)
                model.fit(
                    X_train, y_train, verbose=False, cat_features=categorical_indices
                )

            utils.logger.info("Testing model training finished!")

        # test only
        with open(
            os.path.join(
                utils.read_config("data.processed_path"), "processed_property_dtype.pkl"
            ),
            "rb",
        ) as pkl_file:
            property_dtype_dict = pickle.load(pkl_file)

        p16 = pandas.read_csv(
            os.path.join(
                utils.read_config("data.processed_path"), "processed_property_16.csv"
            ),
            dtype=property_dtype_dict,
        )
        p17 = pandas.read_csv(
            os.path.join(
                utils.read_config("data.processed_path"), "processed_property_17.csv"
            ),
            dtype=property_dtype_dict,
        )

        utils.logger.info("Testing data loaded")

        # Construct DataFrame for prediction results
        submission_2016 = pandas.DataFrame()
        submission_2017 = pandas.DataFrame()
        submission_2016["ParcelId"] = p16["parcelid"]
        submission_2017["ParcelId"] = p17["parcelid"]

        test_p16, test_p17 = transform_test_data(p16, p17)

        try:
            test_p16.drop("Unnamed: 0", inplace=True, axis=1)
            test_p17.drop("Unnamed: 0", inplace=True, axis=1)
        except KeyError:
            pass

        if utils.read_config("model.name") == "lightgbm":
            if not utils.read_config("do_full_train_before_test"):
                model = utils.load_lightgbm_model(utils.read_config("test_model"))
                utils.logger.info("Testing model loaded")

            pred_2016 = model.predict(test_p16)
            pred_2017 = model.predict(test_p17)

            submission_2016["201610"] = [float(format(x, ".4f")) for x in pred_2016]
            submission_2017["201710"] = [float(format(x, ".4f")) for x in pred_2017]
            utils.logger.info("Testing for October finished.")

            if utils.read_config("test.oct"):
                # The author of the reference claimed that 11 & 12 lead to unstable results,
                # probably due to the fact that there are few training examples for them
                submission_2016["201611"] = submission_2016["201610"]
                submission_2016["201612"] = submission_2016["201610"]

                submission_2017["201711"] = submission_2017["201710"]
                submission_2017["201712"] = submission_2017["201710"]

                utils.logger.info(
                    "All results are set to be the same with those of October, thus testing has finished."
                )

            else:
                transform_test_data_month(test_p16, test_p17, 11)
                pred_2016 = model.predict(test_p16)
                pred_2017 = model.predict(test_p17)
                submission_2016["201611"] = [float(format(x, ".4f")) for x in pred_2016]
                submission_2017["201711"] = [float(format(x, ".4f")) for x in pred_2017]

                utils.logger.info("Testing for November finished.")

                transform_test_data_month(test_p16, test_p17, 12)
                pred_2016 = model.predict(test_p16)
                pred_2017 = model.predict(test_p17)
                submission_2016["201612"] = [float(format(x, ".4f")) for x in pred_2016]
                submission_2017["201712"] = [float(format(x, ".4f")) for x in pred_2017]

                utils.logger.info("Testing for December finished.")

        elif utils.read_config("model.name") == "catboost":
            if not utils.read_config("do_full_train_before_test"):
                model = utils.load_catboost_model(utils.read_config("test_model"))
                utils.logger.info("Testing model loaded")

            categorical_indices = [
                str(col)
                for col in test_p16.columns
                if test_p16[col].dtype == "category"
            ]

            for cat_col in categorical_indices:
                test_p16[cat_col].fillna(0, inplace=True)
                test_p17[cat_col].fillna(0, inplace=True)
            pred_2016 = model.predict(test_p16)
            pred_2017 = model.predict(test_p17)

            submission_2016["201610"] = [float(format(x, ".4f")) for x in pred_2016]
            submission_2017["201710"] = [float(format(x, ".4f")) for x in pred_2017]
            utils.logger.info("Testing for October finished.")

            if utils.read_config("test.oct"):
                # The author of the reference claimed that 11 & 12 lead to unstable results,
                # probably due to the fact that there are few training examples for them
                submission_2016["201611"] = submission_2016["201610"]
                submission_2016["201612"] = submission_2016["201610"]

                submission_2017["201711"] = submission_2017["201710"]
                submission_2017["201712"] = submission_2017["201710"]

                utils.logger.info(
                    "All results are set to be the same with those of October, thus testing has finished."
                )

            else:
                transform_test_data_month(test_p16, test_p17, 11)
                pred_2016 = model.predict(test_p16)
                pred_2017 = model.predict(test_p17)
                submission_2016["201611"] = [float(format(x, ".4f")) for x in pred_2016]
                submission_2017["201711"] = [float(format(x, ".4f")) for x in pred_2017]

                utils.logger.info("Testing for November finished.")

                transform_test_data_month(test_p16, test_p17, 12)
                pred_2016 = model.predict(test_p16)
                pred_2017 = model.predict(test_p17)
                submission_2016["201612"] = [float(format(x, ".4f")) for x in pred_2016]
                submission_2017["201712"] = [float(format(x, ".4f")) for x in pred_2017]

                utils.logger.info("Testing for December finished.")

        submission = submission_2016.merge(
            how="inner", right=submission_2017, on="ParcelId"
        )

        utils.logger.info(f"Length of submission DataFrame: {len(submission)}")
        submission.to_csv(
            os.path.join(utils.read_config("save_path"), "submission.csv"), index=False
        )
