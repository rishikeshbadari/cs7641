import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Dataset:

    def __init__(self, src, merge=False) -> None:
        """
        Creates a Dataset object

        Args:
            src (str|list(str)): the path (or list of paths) to the csv file(s)
            merge (bool, optional): if we have to merge the csv in the src list.
                Defaults to False.
        """
        with open("Datasets/noc_team.json", "r") as file:
            self.nocs = json.load(file)
        if merge:
            # Load the dictionary of the cities
            with open("Datasets/cities.json", "r") as file:
                self.cities = json.load(file)
            # Load all the subdatasets
            dfs = []
            # Fix the city column
            for path in src:
                # List of rows to drop
                self.youth = []
                df = pd.read_csv(path)
                # Handle empty CSVs (the last ones)
                if len(df) == 0:
                    continue
                df["City"] = df.apply(self.get_city, axis=1)
                # Drop the youth entries
                bfr = len(df)
                df = df.drop(self.youth)
                print(f"Dropping {bfr - len(df)} youth entries")
                # Add the current df to the list
                dfs.append(df)
            print()
            # Merge the data in one dataframe
            df = pd.concat(dfs, ignore_index=True)
            self.src = src[0].rpartition("_")[0] + "_" + src[-1].rpartition("_")[-1]
            self.src = self.src.replace("/splits", "")
        else:
            df = pd.read_csv(src)
            self.src = src
        self.labels = df["Medal"]
        self.datapoints = df  # df.iloc[:, :-1].copy()
        self.label_encoders = {}
        self.map = {}

    def get_city(self, row):
        """
        Get the city for the given row

        Args:
            row (pd.row): a row of the dataframe

        Raises:
            Exception: custom exception for debugging

        Returns:
            str: the city's name
        """
        try:
            year = row["Year"]
            # Very special case, intercalated games in 1906
            if year == 1906:
                return "Athens"
            season = row["Season"]
            return self.cities[season][str(year)]
        except:
            # Very special case again, the equestrian events of the 1956's games
            # were done in Stockholm instead of Melbourne due to some laws
            if year == 1956:
                return "Stockholm"
            # Some athletes competed in youth Olympics, we want to remove these
            # entries, we return a dummy value to avoid an exception
            elif year in np.arange(2010, 2023, 2):
                self.youth.append(row.name)
                return "YOUTH OLYMPICS - DELETED"
            # For debugging
            raise Exception(f"Row: {row}")

    def get_team(self, noc):
        return self.nocs[noc]

    def clean(self, method="drop"):
        """
        Cleans the dataset by replacing the NaNs values or dropping the
        lines. The behavior depends on the chosen method

        Args:
            method (str, optional): The method to use to replace the NaNs.
                One of "drop", "median", "mean". Defaults to "drop".
        """
        # Fix the missing 'Team' entry
        msk = self.datapoints["Team"].isna()
        self.datapoints.loc[msk, "Team"] = self.datapoints.loc[msk, "NOC"].apply(self.get_team)

        # Replace the Nan in the medal field by a string
        msk = self.datapoints["Medal"].isna()
        self.datapoints.loc[msk, "Medal"] = "None"

        print(f"NA before cleaning (df size: {self.datapoints.shape[0]}):")
        print(f"{self.datapoints.isna().sum()}\n")
        if method == "drop":
            self.datapoints = self.datapoints.dropna()
        else:
            # Only numerical values
            for column in ["Age", "Height", "Weight"]:

                if method == "median":
                    self.datapoints[column] = self.datapoints[column].fillna(
                        self.datapoints[column].median()
                    )
                elif method == "mean":
                    self.datapoints[column] = self.datapoints[column].fillna(
                        self.datapoints[column].mean()
                    )

        print(f"NA after cleaning (df size: {self.datapoints.shape[0]}):")
        print(f"{self.datapoints.isna().sum()}\n")

    def prepare(self):
        """
        Updates the dataset by replacing the string labels by encoded integers
        """
        # Ensure some columns are of the correct datatype
        self.datapoints["ID"] = self.datapoints["ID"].astype(int)
        self.datapoints["Year"] = self.datapoints["Year"].astype(int)

        # Encode all the strings to integers
        for column in [
            "Name",
            "Sex",
            "Team",
            "NOC",
            "Games",
            "Season",
            "City",
            "Sport",
            "Event",
            "Medal",
        ]:
            le = LabelEncoder()
            dtp = self.datapoints[column]
            tfd = le.fit_transform(dtp)
            unq = set(dtp)
            unq = list(unq)
            tfq = le.transform(unq)
            self.datapoints[column] = tfd
            self.label_encoders[column] = le
            self.map[column] = (unq, tfq.tolist())
            print(f"{column}: {len(le.classes_)} classes")

        # First convert heights and weights to string, for some reason pandas thinks it's float64
        self.datapoints["Height"] = self.datapoints["Height"].astype(str)
        self.datapoints["Weight"] = self.datapoints["Weight"].astype(str)

        # Some data is not well stored, and we ended up with weights in the height column
        # So we find these values, store them into the weight column and replace their
        # height by a NaN
        kg_mask = self.datapoints["Height"].str.contains("kg")
        self.datapoints.loc[kg_mask, "Weight"] = self.datapoints.loc[kg_mask, "Height"]
        self.datapoints.loc[kg_mask, "Height"] = None
        # Finally, we remove the 'cm' tag before converting the heights back to float
        self.datapoints["Height"] = self.datapoints["Height"].str.replace("cm", "").astype(float)

        # For some weights, we have a range of weights,either separated by an hyphen or a comma
        # We take the first one (arbitrary choice)
        self.datapoints["Weight"] = self.datapoints["Weight"].str.split("-").str[0]
        self.datapoints["Weight"] = self.datapoints["Weight"].str.split(",").str[0]
        # Remove the 'kg' tag and convert back to float
        self.datapoints["Weight"] = self.datapoints["Weight"].str.replace("kg", "").astype(float)
        print()

    def save(self):
        """
        Saves a new csv file containing the current dataset.
        Appends '_new' to the file name.

        Also saves a json file containing the mapping between
        the originals and encoded labels
        """
        new_src = self.src.rpartition(".")[0]
        self.datapoints.to_csv(new_src + "_new.csv", index=False)
        with open(new_src + ".json", "w") as file:
            json.dump(self.map, file, indent=4)

    def get_correlation(self, save=False):
        """
        Get the correlation matrix and creates a plot

        Returns:
            ndarray: the correlation matrix
        """
        corr_mat = self.datapoints.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        if save:
            plt.savefig("Figures/correlation_matrix.png")
        plt.show()
        return corr_mat

    def run_pca(self, n_comp=None):
        """
        Run a PCA with the specified number of components

        Args:
            n_comp (int, optional): the number of components. Defaults to None.

        Returns:
            pca_data (ndarray): the dataset after the PCA
        """
        scaler = StandardScaler()
        ctrd_data = scaler.fit_transform(self.datapoints)

        self.pca = PCA(n_components=n_comp)
        pca_data = self.pca.fit_transform(ctrd_data)

        return pca_data


if __name__ == "__main__":
    # Load the original dataset, clean the data, encode the strings
    # and save it to a new file
    dataset = Dataset("Datasets/kaggle/athlete_events.csv")
    dataset.clean()
    dataset.prepare()
    dataset.save()

    # Merge the scrapped datasets into a single one, encode the strings
    # and save it to a new file
    scraping_dataset = Dataset(
        [
            "Datasets/splits/olympics_athletes_1_10000.csv",
            "Datasets/splits/olympics_athletes_10001_20000.csv",
            "Datasets/splits/olympics_athletes_20001_30000.csv",
            "Datasets/splits/olympics_athletes_30001_40000.csv",
            "Datasets/splits/olympics_athletes_40001_50000.csv",
            "Datasets/splits/olympics_athletes_50001_60000.csv",
            "Datasets/splits/olympics_athletes_60001_70000.csv",
            "Datasets/splits/olympics_athletes_70001_80000.csv",
            "Datasets/splits/olympics_athletes_80001_90000.csv",
            "Datasets/splits/olympics_athletes_90001_100000.csv",
            "Datasets/splits/olympics_athletes_100001_110000.csv",
            "Datasets/splits/olympics_athletes_110001_120000.csv",
            "Datasets/splits/olympics_athletes_120001_130000.csv",
            "Datasets/splits/olympics_athletes_130001_140000.csv",
            "Datasets/splits/olympics_athletes_140001_150000.csv",
            "Datasets/splits/olympics_athletes_150001_160000.csv",  # no data after 150,000
            "Datasets/splits/olympics_athletes_160001_170000.csv",
            "Datasets/splits/olympics_athletes_170001_180000.csv",
            "Datasets/splits/olympics_athletes_180001_190000.csv",
            "Datasets/splits/olympics_athletes_190001_200000.csv",
        ],
        merge=True,
    )
    scraping_dataset.clean()
    scraping_dataset.prepare()
    scraping_dataset.save()
    corr_mat = scraping_dataset.get_correlation()
    print(corr_mat)

    exit()

    # TODO: comment this part
    dataset.run_pca()

    abs_loadings = np.abs(dataset.pca.components_)
    sum_abs_loadings = np.sum(abs_loadings, axis=0)
    features_idx = np.argsort(sum_abs_loadings)[::-1]
    print(
        f"""Best features: {list(zip(
            dataset.datapoints.columns.to_numpy()[features_idx],
            dataset.pca.explained_variance_ratio_))}"""
    )

    cum_var_ratio = np.cumsum(dataset.pca.explained_variance_ratio_)
    print(f"Cumulative variance: {cum_var_ratio}")
    n_comp = np.argmax(cum_var_ratio > 0.95) + 1

    pca_data = dataset.run_pca(n_comp)
