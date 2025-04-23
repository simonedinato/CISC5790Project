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

# Other Import Data
dataset_cleaned = 'alzheimers_disease_cleaned.csv'
dataset_updated = 'updated_data.csv'


# DF build
dataset_raw_df = pd.read_csv(dataset_raw)
dataset_oversampled_df = pd.read_excel(dataset_oversampled)
dataset_cleaned_df = pd.read_csv(dataset_cleaned)
dataset_updated_df = pd.read_csv(dataset_updated)
# show(dataset_test_df)


if __name__ == '__main__':

    # Preproccessing
    df_clean = dataset_raw_df.drop(columns=['PatientID', 'DoctorInCharge'])  # Drop IDs if present

    # df_clean = dataset_updated_df.drop(columns=['PatientID'])  # Drop IDs if present
    scaler = StandardScaler()
    # print(df_clean.columns)
    # show(df_clean) ===== 2,149x33

    # # See if there is any missing data
    # df_clean = df_clean.dropna()
    # show(df_clean) ===== 2,149x33
    # # No Missing Data


    x = df_clean.drop(columns=['Diagnosis'])
    y = df_clean['Diagnosis']

    x_scaled = scaler.fit_transform(x)

    ks = [1, 3, 5, 10, 25, 100, 200] # run K max sqrt(n)
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    highest_accuracy = 0
    accuracies = []

    best_k = 0
    for k in range(1,200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        # print(f"k={k} - Accuracy = {accuracy}")
        accuracies.append(accuracy)


        if(accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
            best_cm = confusion_matrix(y, y_pred)

        # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 200), accuracies, color='green')
    plt.title('5 Fold Accuracy vs. K in KNN - Raw Dataset')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.show()
    print(f"5 Fold Performance: Best k = {best_k} - Accuracy = {highest_accuracy}")
    ConfusionMatrixDisplay(confusion_matrix=best_cm).plot(cmap='Greens')
    plt.title(f'5 Fold Confusion Matrix Raw Dataset (k={best_k})')
    plt.show()

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    highest_accuracy = 0
    accuracies = []

    best_k = 0
    for k in range(1, 200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        # print(f"k={k} - Accuracy = {accuracy}")
        accuracies.append(accuracy)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
            best_cm = confusion_matrix(y, y_pred)

        # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 200), accuracies, color='green')
    plt.title('10 Fold Accuracy vs. K in KNN - Raw Dataset')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.show()

    ConfusionMatrixDisplay(confusion_matrix=best_cm).plot(cmap='Greens')
    plt.title(f'10 Fold Confusion Matrix Raw Dataset (k={best_k})')
    plt.show()

        # Compute confusion matrix
        # if(k == 103):
        #     cm = confusion_matrix(y, y_pred)
        #     ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
        #     plt.title(f'Confusion Matrix (k={k})')
        #     plt.show()

    print(f"10 Fold Performance: Best k = {best_k} - Accuracy = {highest_accuracy}")


    # Over Sampled Data
    df_clean = dataset_oversampled_df.drop(columns=['PatientID', 'DoctorInCharge'])  # Drop IDs if present
    scaler = StandardScaler()

    x = df_clean.drop(columns=['Diagnosis'])
    y = df_clean['Diagnosis']

    x_scaled = scaler.fit_transform(x)

    ks = [1, 3, 5, 10, 25, 100, 200]  # run K max sqrt(n)
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    highest_accuracy = 0
    accuracies = []

    best_k = 0
    for k in range(1, 200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        # print(f"k={k} - Accuracy = {accuracy}")
        accuracies.append(accuracy)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
            best_cm = confusion_matrix(y, y_pred)

        # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 200), accuracies, color='purple')
    plt.title('5 Fold Accuracy vs. K in KNN - Oversampled Data')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.show()
    print(f"5 Fold Performance: Best k = {best_k} - Accuracy = {highest_accuracy}")

    ConfusionMatrixDisplay(confusion_matrix=best_cm).plot(cmap='Purples')
    plt.title(f'5 Fold Confusion Matrix Oversampled Data(k={best_k})')
    plt.show()

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    highest_accuracy = 0
    accuracies = []

    best_k = 0
    for k in range(1, 200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        # print(f"k={k} - Accuracy = {accuracy}")
        accuracies.append(accuracy)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
            best_cm = confusion_matrix(y, y_pred)

        # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 200), accuracies, color='purple')
    plt.title('10 Fold Accuracy vs. K in KNN')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.show()
    print(f"10 Fold Performance: Best k = {best_k} - Accuracy = {highest_accuracy}")

    ConfusionMatrixDisplay(confusion_matrix=best_cm).plot(cmap='Purples')
    plt.title(f'10 Fold Confusion Matrix (k={best_k})')
    plt.show()


    df_clean = dataset_cleaned_df  # .drop(columns=['PatientID', 'DoctorInCharge'])  # Drop IDs if present

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    highest_accuracy = 0
    accuracies = []

    best_k = 0
    for k in range(1, 200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        # print(f"k={k} - Accuracy = {accuracy}")
        accuracies.append(accuracy)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
            best_cm = confusion_matrix(y, y_pred)

        # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 200), accuracies, color='red')
    plt.title('5 Fold Accuracy vs. K in KNN - Oversampled Data - Cleaned Data')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.show()
    print(f"5 Fold Performance: Best k = {best_k} - Accuracy = {highest_accuracy}")

    ConfusionMatrixDisplay(confusion_matrix=best_cm).plot(cmap='Reds')
    plt.title(f'5 Fold Confusion Matrix Oversampled Data(k={best_k}) - Cleaned Data')
    plt.show()

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    highest_accuracy = 0
    accuracies = []

    best_k = 0
    for k in range(1, 200):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, x_scaled, y, cv=cv)

        accuracy = accuracy_score(y, y_pred) * 100
        # print(f"k={k} - Accuracy = {accuracy}")
        accuracies.append(accuracy)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
            best_cm = confusion_matrix(y, y_pred)

        # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 200), accuracies, color='red')
    plt.title('10 Fold Accuracy vs. K in KNN - Cleaned Data')
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.show()

    ConfusionMatrixDisplay(confusion_matrix=best_cm).plot(cmap='Reds')
    plt.title(f'10 Fold Confusion Matrix (k={best_k}) - Cleaned Data')
    plt.show()
