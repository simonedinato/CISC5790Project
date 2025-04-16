import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from pandasgui import show
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Import data
dataset_raw = 'alzheimers_disease_data.csv'
dataset_oversampled = 'alzheimers_disease_data_oversampled.xlsx'


# DF build
dataset_raw_df = pd.read_csv(dataset_raw)
dataset_oversampled_df = pd.read_excel(dataset_oversampled)
# show(dataset_test_df)


if __name__ == '__main__':

    # Preproccessing
    df_clean = dataset_raw_df.drop(columns=['PatientID', 'DoctorInCharge'])  # Drop IDs if present
    df_clean = dataset_oversampled_df.drop(columns=['PatientID', 'DoctorInCharge'])  # Drop IDs if present
    scaler = StandardScaler()
    # print(df_clean.columns)
    # show(df_clean) ===== 2,149x33

    # # See if there is any missing data
    # df_clean = df_clean.dropna()
    # show(df_clean) ===== 2,149x33
    # # No Missing Data


    x = df_clean.drop(columns=['Diagnosis'])
    y = df_clean['Diagnosis']

    # before
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)

    x_scaled = scaler.fit_transform(x)

    ks = [1, 3, 5, 10, 25, 100, 200] # run K max sqrt(n)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    highest_accuracy = 0
    best_k = 0
    for k in range(1,200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        print(f"k={k} - Accuracy = {accuracy}")


        if(accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k

        # Compute confusion matrix
        if(k == 25):
            cm = confusion_matrix(y, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
            plt.title(f'Confusion Matrix (k={k})')
            plt.show()

    print(f"Best k = {best_k} -  Accuracy = {highest_accuracy}")



    print("WIP!")